"""Associative recall: adaptation and evaluation of a pretrained checkpoint.

A prompt is a sequence of key-value pairs followed by one query key:
    [k_1, v_1, ..., k_N, v_N, k_query]
The model must output the value paired with the query key. Keys and values are
drawn from disjoint token ranges of the real vocabulary. Only the last position
is supervised. Strict accuracy is the argmax over the full vocabulary;
candidate accuracy is the argmax restricted to the N values in the prompt.

In finetune mode (default) the pretrained model is first adapted on the task
(uniform query positions), then evaluated. Options:

  --n_pairs N         number of key-value pairs per prompt (memory capacity).
                      The key/value pools are enlarged if N exceeds
                      --n_keys / --n_values.
  --distance_mode     query-distance axis at evaluation time:
        uniform         query pair sampled uniformly,
        fixed           every query sits --distance pairs back from the end,
        stratified      each batch is spread evenly over the recency buckets
                        given by --dist_edges; accuracy is reported per bucket.
      Distance d counts pairs between the queried pair and the query position
      (d = 0 is the most recent pair).
  --seed / --eval_seed
      --seed drives the adaptation data order and selects the checkpoint (pass
      the pretraining seed). --eval_seed drives the evaluation prompts and is
      fixed by default, so all models see the same prompts.

One CSV row per distance bucket is appended to logs/recall_results.csv. For
uniform mode a single row with dist_lo = dist_hi = -1 is written.

Examples:
    python recall.py --model trissm --seed 42 --n_pairs 10
    python recall.py --model fp_ssm --seed 42 --n_pairs 50 \\
        --distance_mode stratified --dist_edges 0,1,2,4,8,16,32,50 \\
        --experiment distance
"""
import argparse
import time

import torch
import torch.nn as nn

from configs import RECALL_CSV, REGISTRY, append_csv, resolve_ckpt
from train import count_params

RECALL_CSV_FIELDS = ["experiment", "arch", "seed", "n_pairs",
                     "distance_mode", "dist_lo", "dist_hi",
                     "n_prompts", "strict_correct", "strict_acc",
                     "cand_correct", "cand_acc",
                     "ft_steps", "ft_lr", "batch_size", "eval_seed",
                     "ckpt", "timestamp"]


def sample_kv_pool(key_lo, key_hi, val_lo, val_hi, pool_size, gen):
    """Sample key tokens from [key_lo, key_hi) and value tokens from
    [val_lo, val_hi)."""
    key_perm = torch.randperm(key_hi - key_lo, generator=gen)[:pool_size]
    keys = (key_perm + key_lo).long()
    val_perm = torch.randperm(val_hi - val_lo, generator=gen)[:pool_size]
    vals = (val_perm + val_lo).long()
    return keys, vals


def make_buckets(distance_mode, n_pairs, distance, dist_edges):
    """List of (lo, hi) pair-distance ranges [lo, hi) clamped to [0, n_pairs].
    Uniform mode returns the single sentinel bucket (-1, -1)."""
    if distance_mode == "uniform":
        return [(-1, -1)]
    if distance_mode == "fixed":
        d = min(distance, n_pairs - 1)
        return [(d, d + 1)]
    edges = sorted(set(int(e) for e in dist_edges.split(",")))
    edges = [e for e in edges if 0 <= e <= n_pairs]
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != n_pairs:
        raise ValueError(
            f"--dist_edges must start at 0 and end at n_pairs ({n_pairs}) "
            f"after clamping; got {edges}")
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def make_batch(keys, vals, n_pairs, batch_size, gen, device, buckets):
    """Build one batch of recall prompts. Returns (x, y, bucket_ids).

    x[b] = [k_1, v_1, ..., k_N, v_N, k_query]; only position -1 is supervised.
    Examples are assigned to buckets round-robin; bucket_ids[b] indexes into
    `buckets`. In the uniform sentinel bucket the query pair is uniform.
    """
    seq = 2 * n_pairs + 1
    x = torch.zeros(batch_size, seq, dtype=torch.long)
    y = torch.full((batch_size, seq), -100, dtype=torch.long)
    ki = torch.randperm(len(keys), generator=gen)[:n_pairs]
    ks = keys[ki]
    vi = torch.randint(0, len(vals), (n_pairs,), generator=gen)
    vs = vals[vi]
    x[:, 0:2 * n_pairs:2] = ks.unsqueeze(0).expand(batch_size, -1)
    x[:, 1:2 * n_pairs:2] = vs.unsqueeze(0).expand(batch_size, -1)
    bucket_ids = []
    for b in range(batch_size):
        lo, hi = buckets[b % len(buckets)]
        if lo < 0:  # uniform
            qi = torch.randint(0, n_pairs, (1,), generator=gen).item()
        else:
            hi_c = min(hi, n_pairs)
            lo_c = min(lo, n_pairs - 1)
            d = torch.randint(lo_c, hi_c, (1,), generator=gen).item()
            qi = n_pairs - 1 - d
        bucket_ids.append(b % len(buckets))
        x[b, -1] = ks[qi]
        y[b, -1] = vs[qi]
    return x.to(device), y.to(device), bucket_ids


@torch.no_grad()
def eval_recall(model, keys, vals, n_pairs, batch_size, n_batches,
                gen, device, buckets):
    """Per-bucket strict and candidate accuracy counts."""
    model.eval()
    stats = [{"strict": 0, "cand": 0, "total": 0} for _ in buckets]
    for _ in range(n_batches):
        x, y, bucket_ids = make_batch(keys, vals, n_pairs, batch_size,
                                      gen, device, buckets)
        logits = model(x, last_only=True)  # (B, 1, V)
        pred = logits[:, -1, :].argmax(dim=-1)
        for b in range(pred.size(0)):
            s = stats[bucket_ids[b]]
            s["total"] += 1
            tgt = y[b, -1].item()
            s["strict"] += int(pred[b].item() == tgt)
            presented = x[b, 1:2 * n_pairs:2]
            sub_logits = logits[b, -1, :][presented]
            pick = presented[sub_logits.argmax()].item()
            s["cand"] += int(pick == tgt)
    return stats


def finetune_recall(model, keys, vals, n_pairs, batch_size, ft_steps,
                    ft_lr, ft_warmup, gen, device, vocab_size,
                    log_every=250):
    """Adapt the model on recall prompts: uniform query positions, AdamW
    (betas 0.9/0.95, weight decay 1e-2), gradient clipping at 1.0, linear
    warmup over ft_warmup steps."""
    model.train()
    uniform_bucket = [(-1, -1)]
    optim = torch.optim.AdamW(model.parameters(), lr=ft_lr,
                              betas=(0.9, 0.95), weight_decay=1e-2)
    for step in range(ft_steps):
        if ft_warmup and step < ft_warmup:
            lr = ft_lr * step / ft_warmup
            for g in optim.param_groups:
                g["lr"] = lr
        x, y, _ = make_batch(keys, vals, n_pairs, batch_size, gen, device,
                             uniform_bucket)
        logits = model(x, last_only=True)  # (B, 1, V)
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, vocab_size), y[:, -1].view(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % log_every == 0 or step == ft_steps - 1:
            pred = logits[:, -1, :].argmax(dim=-1)
            acc = (pred == y[:, -1]).float().mean().item()
            print(f"  [ft] step {step:4d}/{ft_steps} | loss {loss.item():.4f}"
                  f" | acc {acc:.3f}")
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Key-value associative recall")
    p.add_argument("--model", choices=list(REGISTRY.keys()), required=True)
    p.add_argument("--ckpt", default=None,
                   help="Default: out/<model>_s<seed>/model.pt")
    p.add_argument("--ckpt_root", default="out")
    p.add_argument("--seed", type=int, default=0,
                   help="Drives the adaptation data order and checkpoint "
                        "resolution. Pass the pretraining seed.")
    p.add_argument("--eval_seed", type=int, default=12345,
                   help="Drives the evaluation prompts; fixed across models.")
    p.add_argument("--experiment", default="headline",
                   help="Tag written to the CSV (headline | capacity | "
                        "distance | configs).")
    p.add_argument("--log_csv", default=RECALL_CSV)
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--n_keys", type=int, default=30)
    p.add_argument("--n_values", type=int, default=30)
    p.add_argument("--n_pairs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_batches", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--mode", choices=["zeroshot", "finetune"],
                   default="finetune")
    p.add_argument("--ft_steps", type=int, default=3000)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--ft_warmup", type=int, default=100)
    p.add_argument("--distance_mode",
                   choices=["uniform", "fixed", "stratified"],
                   default="uniform")
    p.add_argument("--distance", type=int, default=0,
                   help="fixed mode: pair-distance of every query.")
    p.add_argument("--dist_edges", default="0,1,2,4,8,16,32,50",
                   help="stratified mode: comma-separated bucket edges; "
                        "must span 0..n_pairs after clamping.")
    p.add_argument("--checkpoint", action="store_true",
                   help="Gradient-checkpoint the SSM blocks during adaptation "
                        "(for long prompts). Enabled automatically when the "
                        "prompt has 128 or more tokens.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt_path = args.ckpt or resolve_ckpt(args.ckpt_root, args.model,
                                          args.seed)
    if ckpt_path is None:
        raise SystemExit(
            f"ERROR: no checkpoint for model={args.model} seed={args.seed} "
            f"under {args.ckpt_root}/. Train it first (train.py).")
    print(f"model = {args.model}  ckpt = {ckpt_path}  mode = {args.mode}")

    pool_size = max(args.n_keys, args.n_values, args.n_pairs)
    if pool_size > max(args.n_keys, args.n_values):
        print(f"note: n_pairs={args.n_pairs} exceeds the key/value pool; "
              f"enlarging pool to {pool_size}.")

    gen = torch.Generator().manual_seed(args.seed)
    key_lo, key_hi = 50, args.vocab_size // 2
    val_lo, val_hi = args.vocab_size // 2, args.vocab_size
    keys, vals = sample_kv_pool(key_lo, key_hi, val_lo, val_hi, pool_size, gen)
    print(f"KV pool = {pool_size} tokens  n_pairs = {args.n_pairs}  "
          f"seq_len required = {2 * args.n_pairs + 1}")
    assert 2 * args.n_pairs + 1 <= args.seq_len, "prompt exceeds seq_len"

    buckets = make_buckets(args.distance_mode, args.n_pairs,
                           args.distance, args.dist_edges)
    if args.distance_mode == "stratified":
        print(f"distance buckets (pairs): {buckets}")

    model = REGISTRY[args.model]["factory"](args.vocab_size)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if hasattr(model, "use_checkpoint") and (args.checkpoint
                                             or 2 * args.n_pairs + 1 >= 128):
        model.use_checkpoint = True
        print("gradient checkpointing ON")
    total, ternary = count_params(model)
    print(f"params = {total/1e6:.3f}M  ternary = {ternary/1e6:.3f}M")

    if args.mode == "finetune":
        print(f"\nAdapting on recall task: {args.ft_steps} steps @ "
              f"lr {args.ft_lr} (uniform query distances)")
        ft_gen = torch.Generator().manual_seed(args.seed * 1000 + 1)
        model = finetune_recall(model, keys, vals, args.n_pairs,
                                args.batch_size, args.ft_steps, args.ft_lr,
                                args.ft_warmup, ft_gen, device,
                                args.vocab_size)

    print(f"\nEvaluating recall over {args.n_batches} batches "
          f"({args.n_batches * args.batch_size} prompts), "
          f"distance_mode = {args.distance_mode}...")
    eval_gen = torch.Generator().manual_seed(args.eval_seed)
    stats = eval_recall(model, keys, vals, args.n_pairs, args.batch_size,
                        args.n_batches, eval_gen, device, buckets)

    print(f"\n=== {args.model} ({args.mode}, "
          f"seed {args.seed}, N={args.n_pairs}) ===")
    for (lo, hi), s in zip(buckets, stats):
        if s["total"] == 0:
            continue
        strict = s["strict"] / s["total"]
        cand = s["cand"] / s["total"]
        tag = "all (uniform)" if lo < 0 else f"d in [{lo},{hi}) pairs"
        print(f"  {tag:<22} n={s['total']:5d}  strict={strict*100:6.2f}%  "
              f"candidate={cand*100:6.2f}%")
        append_csv(args.log_csv, RECALL_CSV_FIELDS, {
            "experiment": args.experiment,
            "arch": args.model,
            "seed": args.seed,
            "n_pairs": args.n_pairs,
            "distance_mode": args.distance_mode,
            "dist_lo": lo,
            "dist_hi": hi,
            "n_prompts": s["total"],
            "strict_correct": s["strict"],
            "strict_acc": f"{strict * 100:.2f}",
            "cand_correct": s["cand"],
            "cand_acc": f"{cand * 100:.2f}",
            "ft_steps": args.ft_steps,
            "ft_lr": args.ft_lr,
            "batch_size": args.batch_size,
            "eval_seed": args.eval_seed,
            "ckpt": ckpt_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    print(f"chance level: strict {100/args.vocab_size:.4f}%  "
          f"candidate {100/args.n_pairs:.2f}%")
    print(f"Rows appended to {args.log_csv}")


if __name__ == "__main__":
    main()
