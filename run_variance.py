"""Driver: per-step variance probe.

Runs train.py in --variance_only mode for the headline models over the
requested seeds. Each cell trains from scratch for --variance_steps iterations
(default 500) with the learning-rate schedule of the full run, appends one
variance row per step to logs/variance.csv (hidden-state variance for SSMs,
pre-softmax attention-score variance for Transformers), then exits without
writing a checkpoint or a training row.

Completed cells (>= --variance_steps rows) are skipped. A partial cell is
cleared and re-run from step 0. Requires data/train.bin and data/val.bin
(run prepare_data.py first).

Usage:
    python run_variance.py --dry_run
    python run_variance.py                                  # seeds 42,43,44
    python run_variance.py --archs fp_transformer,bitnet_transformer
"""
import argparse
import csv
import os

from configs import HEADLINE_ARCHES, VARIANCE_CSV, finish, run_cell

FIELDS = ["experiment", "arch", "seed", "step", "variance"]


def count_cell(path, arch, seed):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r.get("arch") == arch and r.get("seed") == str(seed))


def drop_cell(path, arch, seed):
    """Remove a partial cell's rows so it can be re-run from step 0."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows
            if not (r.get("arch") == arch and r.get("seed") == str(seed))]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", default=",".join(HEADLINE_ARCHES),
                   help="Comma-separated models to probe.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--variance_steps", type=int, default=500)
    p.add_argument("--variance_csv", default=VARIANCE_CSV)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    archs = args.archs.split(",")
    for a in archs:
        if a not in HEADLINE_ARCHES:
            raise SystemExit(f"unknown arch: {a}")
    failures = []

    for seed in [int(s) for s in args.seeds.split(",")]:
        for arch in archs:
            have = count_cell(args.variance_csv, arch, seed)
            if have >= args.variance_steps:
                print(f"skip (done): variance {arch} seed {seed} "
                      f"({have} steps)")
                continue
            if have:
                print(f"clearing partial cell: variance {arch} seed {seed} "
                      f"({have}/{args.variance_steps} steps)")
                if not args.dry_run:
                    drop_cell(args.variance_csv, arch, seed)
            run_cell(["train.py", "--arch", arch, "--seed", str(seed),
                      "--experiment", "headline", "--variance_only",
                      "--variance_steps", str(args.variance_steps),
                      "--variance_csv", args.variance_csv],
                     args.dry_run, failures)

    finish("run_variance.py", failures)


if __name__ == "__main__":
    main()
