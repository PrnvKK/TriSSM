"""Validation perplexity of trained checkpoints across seeds.

Each checkpoint is evaluated on random windows of val.bin (--eval_iters batches
of --batch_size sequences). Prints per-seed loss and perplexity, the mean and
sample standard deviation across seeds, and the loss and perplexity change from
the full-precision model to its ternary counterpart within each family (when
all four headline models are present).

Example:
    python eval_perplexity.py --seeds 42,43,44
"""
import argparse
import math
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn

from configs import HEADLINE_ARCHES, REGISTRY, append_csv, resolve_ckpt
from train import count_params, effective_bits_per_param, get_batch

FAMILIES = {
    "SSM": ("fp_ssm", "trissm"),
    "Transformer": ("fp_transformer", "bitnet_transformer"),
}

CSV_FIELDS = ["arch", "seed", "val_loss", "ppl", "params", "ternary", "bpp",
              "eval_iters", "timestamp"]


@torch.no_grad()
def eval_val_perplexity(model, val_data, seq_len, batch_size, device,
                        vocab_size, eval_iters=200):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, seq_len, batch_size, device)
        logits = model(X)
        losses[k] = nn.CrossEntropyLoss()(logits.view(-1, vocab_size),
                                          Y.view(-1)).item()
    mean_loss = losses.mean().item()
    return mean_loss, math.exp(mean_loss)


def mean_std(values):
    values = [float(v) for v in values]
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def main():
    p = argparse.ArgumentParser(
        description="Validation perplexity of trained checkpoints.")
    p.add_argument("--models", default=",".join(HEADLINE_ARCHES),
                   help="Comma-separated registry keys.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--ckpt_root", default="out",
                   help="Directory holding <model>_s<seed>/model.pt.")
    p.add_argument("--data_dir", default="data",
                   help="Directory holding val.bin.")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the validation window sampling.")
    p.add_argument("--log_csv", default=os.path.join("logs", "ppl_results.csv"),
                   help="Per-checkpoint rows are appended here. Empty to skip.")
    args = p.parse_args()

    archs = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [a for a in archs if a not in REGISTRY]
    if unknown:
        p.error(f"unknown model(s): {', '.join(unknown)}")
    seeds = [int(s) for s in args.seeds.split(",")]

    val_path = os.path.join(args.data_dir, "val.bin")
    if not os.path.exists(val_path):
        raise SystemExit(f"ERROR: {val_path} not found. Run prepare_data.py first.")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    torch.manual_seed(args.seed)

    results = {}  # (arch, seed) -> (val_loss, ppl)
    for arch in archs:
        for seed in seeds:
            path = resolve_ckpt(args.ckpt_root, arch, seed)
            if path is None:
                print(f"  [skip] {arch} seed {seed}: no checkpoint under "
                      f"{args.ckpt_root}/")
                continue
            model = REGISTRY[arch]["factory"](args.vocab_size)
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            model.to(device)
            total, ternary = count_params(model)
            loss, ppl = eval_val_perplexity(
                model, val_data, args.seq_len, args.batch_size,
                device, args.vocab_size, args.eval_iters)
            results[(arch, seed)] = (loss, ppl)
            print(f"  {arch:<20} seed {seed}: val_loss = {loss:.4f}  "
                  f"ppl = {ppl:.2f}", flush=True)
            if args.log_csv:
                append_csv(args.log_csv, CSV_FIELDS, {
                    "arch": arch, "seed": seed,
                    "val_loss": f"{loss:.4f}", "ppl": f"{ppl:.2f}",
                    "params": total, "ternary": ternary,
                    "bpp": f"{effective_bits_per_param(total, ternary):.2f}",
                    "eval_iters": args.eval_iters,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    if not results:
        raise SystemExit("ERROR: no checkpoints found.")

    seeds_present = sorted({s for (_, s) in results})

    print("\nPer-model validation perplexity")
    print(f"{'Model':<22}" + "".join(f"  seed {s}" for s in seeds_present)
          + "    mean +/- std")
    for arch in archs:
        vals = [results[(arch, s)] for s in seeds_present if (arch, s) in results]
        if not vals:
            continue
        cells = "".join(f"  {ppl:7.2f}" for _, ppl in vals)
        m, sd = mean_std([ppl for _, ppl in vals])
        print(f"{arch:<22}{cells}    {m:6.2f} +/- {sd:.2f}")

    penalty_rows = []
    for seed in seeds_present:
        if not all((a, seed) in results
                   for pair in FAMILIES.values() for a in pair):
            continue
        row = {"seed": seed}
        for fam, (fp_arch, tern_arch) in FAMILIES.items():
            fp_loss, fp_ppl = results[(fp_arch, seed)]
            tern_loss, tern_ppl = results[(tern_arch, seed)]
            row[f"{fam}_ppl_inc"] = 100.0 * (tern_ppl / fp_ppl - 1.0)
            row[f"{fam}_dloss"] = tern_loss - fp_loss
        penalty_rows.append(row)
    if penalty_rows:
        print("\nTernary vs full precision within family")
        print(f"{'Seed':<6}{'SSM ppl +%':>12}{'Transf ppl +%':>15}"
              f"{'SSM dLoss':>12}{'Transf dLoss':>14}")
        for r in penalty_rows:
            print(f"{r['seed']:<6}{r['SSM_ppl_inc']:>12.1f}"
                  f"{r['Transformer_ppl_inc']:>15.1f}"
                  f"{r['SSM_dloss']:>12.4f}{r['Transformer_dloss']:>14.4f}")


if __name__ == "__main__":
    main()
