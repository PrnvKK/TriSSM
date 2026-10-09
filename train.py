"""Train one (architecture, seed) cell on TinyStories.

All models use the same data, token budget, optimizer and LR schedule
(batch_size * gradient_accumulation_steps * seq_len * max_iters = 40.96M
tokens by default). Checkpoints go to out/<arch>_s<seed>/ (model.pt holds the
weights, ckpt.pt additionally holds optimizer state for --resume). On
completion one summary row is appended to logs/train_results.csv.

Examples:
    python train.py --arch trissm --seed 42
    python train.py --arch v2t_deep_fp --seed 42 --experiment configs --amp
    python train.py --arch trissm --seed 42 --resume
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from bitlinear import BitLinear
from configs import REGISTRY, TRAIN_CSV, VARIANCE_CSV, append_csv

TRAIN_CSV_FIELDS = ["experiment", "arch", "seed", "out_dir",
                    "params", "ternary", "bpp",
                    "final_train_loss", "final_val_loss", "final_ppl",
                    "max_iters", "wall_min", "timestamp"]

VARIANCE_CSV_FIELDS = ["experiment", "arch", "seed", "step", "variance"]


def count_params(model):
    """Return (total, ternary): all parameters, and those in BitLinear weights."""
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(m.weight.numel() for m in model.modules()
                  if isinstance(m, BitLinear))
    return total, ternary


def effective_bits_per_param(total, ternary):
    """Ternary weights count as 1.58 bits, all other parameters as 16 bits."""
    return (ternary * 1.58 + (total - ternary) * 16) / total


def get_lr(it, warmup, decay_iters, lr, min_lr):
    """Linear warmup followed by cosine decay to min_lr."""
    if it < warmup:
        return lr * it / warmup
    if it > decay_iters:
        return min_lr
    ratio = (it - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def get_batch(data, seq_len, batch_size, device):
    """Sample a random batch of (input, next-token target) windows."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_len]).astype(np.int64)) for i in ix])
    if device == "cuda":
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, seq_len, batch_size,
                  device, eval_iters, vocab_size):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, seq_len, batch_size, device)
            logits = model(X)
            losses[k] = nn.CrossEntropyLoss()(logits.view(-1, vocab_size),
                                              Y.view(-1)).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def collect_variance(model):
    """Mean over layers of the sequence-mixing variance statistic: hidden-state
    variance for SSM blocks, pre-softmax attention-score variance for
    Transformer blocks. Returns None if nothing was logged."""
    vals = []
    for block in model.blocks:
        sub = getattr(block, "ssm", None) or getattr(block, "attn", None)
        if sub is None:
            continue
        stats = getattr(sub, "stats", None)
        if not stats:
            continue
        v = stats.get("var_hidden_state")
        if v is None:
            v = stats.get("var_attn_score")
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return torch.stack(vals).mean().item()


def main():
    p = argparse.ArgumentParser(description="Train one (architecture, seed) cell.")
    p.add_argument("--arch", choices=list(REGISTRY.keys()), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment", default="headline",
                   help="Tag written to the CSV (headline | configs | ...).")
    p.add_argument("--out_dir", default=None,
                   help="Default: out/<arch>_s<seed>/")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--log_csv", default=TRAIN_CSV)
    p.add_argument("--max_iters", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=5e-5)
    p.add_argument("--warmup_iters", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=1e-1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--resume", action="store_true",
                   help="Resume from <out_dir>/ckpt.pt.")
    p.add_argument("--no_save", action="store_true",
                   help="Skip checkpoint writes.")
    p.add_argument("--amp", action="store_true",
                   help="fp16 autocast with GradScaler (CUDA only).")
    p.add_argument("--variance_steps", type=int, default=500,
                   help="Log the per-step attention/hidden variance for the "
                        "first N steps (0 disables).")
    p.add_argument("--variance_csv", default=VARIANCE_CSV)
    p.add_argument("--variance_only", action="store_true",
                   help="Stop once the first --variance_steps iterations are "
                        "logged; write no checkpoint and no training row. The "
                        "learning-rate schedule of the full run is kept.")
    args = p.parse_args()

    if args.variance_only:
        args.no_save = True
    if args.out_dir is None:
        args.out_dir = os.path.join("out", f"{args.arch}_s{args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise SystemExit(f"ERROR: {train_path} / {val_path} not found. "
                         "Run prepare_data.py first.")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"train tokens = {len(train_data):,}  val tokens = {len(val_data):,}")

    model = REGISTRY[args.arch]["factory"](args.vocab_size).to(device)
    total, ternary = count_params(model)
    bpp = effective_bits_per_param(total, ternary)
    print(f"arch = {args.arch} ({REGISTRY[args.arch]['label']})  seed = {args.seed}")
    print(f"params = {total/1e6:.3f}M  ternary = {ternary/1e6:.3f}M  "
          f"eff_bits/param = {bpp:.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"amp = {'fp16' if use_amp else 'off'}")

    iter_num = 0
    val_loss = 1e9
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = ckpt["iter_num"]
        val_loss = ckpt["val_loss"]
        print(f"  resumed at iter {iter_num} val_loss {val_loss:.4f}")

    arch_meta = {
        "model": args.arch,
        "label": REGISTRY[args.arch]["label"],
        "seed": args.seed,
        "experiment": args.experiment,
        "params_total": total,
        "params_ternary": ternary,
        "bits_per_param": bpp,
        "vocab_size": args.vocab_size,
    }

    tokens_at_halt = (args.batch_size * args.gradient_accumulation_steps
                      * args.seq_len * args.max_iters)
    print(f"training budget = {args.max_iters} iters  "
          f"~ {tokens_at_halt/1e6:.2f}M tokens seen at halt\n")

    t_start = time.time()
    t0 = t_start
    while iter_num <= args.max_iters:
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters,
                    args.learning_rate, args.min_lr) \
            if args.warmup_iters < args.max_iters else args.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args.seq_len,
                                   args.batch_size, device, args.eval_iters,
                                   args.vocab_size)
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

            if not args.no_save:
                val_loss = losses["val"]
                print(f"saving checkpoint to {args.out_dir}...")
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num,
                            "val_loss": val_loss,
                            "arch_meta": arch_meta}, ckpt_path)
                torch.save({"model": model.state_dict(),
                            "iter_num": iter_num,
                            "arch_meta": arch_meta},
                           os.path.join(args.out_dir, "model.pt"))

        optimizer.zero_grad(set_to_none=True)
        lossf = 0.0
        for _ in range(args.gradient_accumulation_steps):
            X, Y = get_batch(train_data, args.seq_len, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                logits = model(X)
            loss = nn.CrossEntropyLoss()(logits.view(-1, args.vocab_size),
                                         Y.view(-1))
            loss = loss / args.gradient_accumulation_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            lossf += loss.item()

        if use_amp:
            scaler.unscale_(optimizer)
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if args.variance_steps and iter_num < args.variance_steps:
            v = collect_variance(model)
            if v is not None:
                append_csv(args.variance_csv, VARIANCE_CSV_FIELDS, {
                    "experiment": args.experiment,
                    "arch": args.arch,
                    "seed": args.seed,
                    "step": iter_num,
                    "variance": f"{v:.6e}",
                })

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0:
            tps = (args.batch_size * args.gradient_accumulation_steps
                   * args.seq_len) / dt
            mem = (f" | GPU {torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
                   if device == "cuda" else "")
            print(f"iter {iter_num:5d} | loss {lossf:.4f} | lr {lr:e} | "
                  f"{dt*1000:.0f}ms | {tps:.0f} tok/s{mem}")
        iter_num += 1

        if args.variance_only and iter_num >= args.variance_steps:
            print(f"\nVariance probe complete: {iter_num} steps logged to "
                  f"{args.variance_csv}.")
            return

    # The CSV row marks the cell as finished for the run_* drivers, so it is
    # written only after the loop completes.
    final = estimate_loss(model, train_data, val_data, args.seq_len,
                          args.batch_size, device, args.eval_iters,
                          args.vocab_size)
    wall_min = (time.time() - t_start) / 60.0
    final_ppl = float(np.exp(final["val"]))
    print(f"\nTraining complete. Final val loss {final['val']:.4f} "
          f"(ppl {final_ppl:.2f}). Wall time {wall_min:.1f} min.")

    if not args.no_save:
        torch.save({"model": model.state_dict(),
                    "iter_num": args.max_iters,
                    "arch_meta": arch_meta},
                   os.path.join(args.out_dir, "model.pt"))

    append_csv(args.log_csv, TRAIN_CSV_FIELDS, {
        "experiment": args.experiment,
        "arch": args.arch,
        "seed": args.seed,
        "out_dir": args.out_dir,
        "params": total,
        "ternary": ternary,
        "bpp": f"{bpp:.2f}",
        "final_train_loss": f"{final['train']:.4f}",
        "final_val_loss": f"{final['val']:.4f}",
        "final_ppl": f"{final_ppl:.2f}",
        "max_iters": args.max_iters,
        "wall_min": f"{wall_min:.1f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Row appended to {args.log_csv}")


if __name__ == "__main__":
    main()
