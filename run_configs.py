"""Driver: parameter-matched Transformer configurations.

Trains each ~5M-parameter Transformer configuration in configs.py (v2t_deep,
v2t_mid, v2t_shallow, v2t_widehead) in full-precision (_fp) and ternary (_tern)
form, then runs the headline recall cell (N=10 pairs, uniform query distance,
3000-step adaptation) on each. This checks whether Transformer recall results
depend on the particular depth/width/head configuration.

Usage:
    python run_configs.py --dry_run
    python run_configs.py                                  # all configs, seeds 42,43,44
    python run_configs.py --configs v2t_deep,v2t_shallow   # subset
    python run_configs.py --seeds 42 --skip_train          # recall cells only
"""
import argparse

from configs import (CONFIG_ARCHES, finish, recall_csv_has, resolve_ckpt,
                     run_cell, train_csv_has)

CONFIG_FAMILIES = ["v2t_deep", "v2t_mid", "v2t_shallow", "v2t_widehead"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--configs", default=",".join(CONFIG_FAMILIES))
    p.add_argument("--n_pairs", type=int, default=10)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_recall", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    archs = [f"{cfg}_{prec}"
             for cfg in args.configs.split(",")
             for prec in ("fp", "tern")]
    for a in archs:
        if a not in CONFIG_ARCHES:
            raise SystemExit(f"unknown config arch: {a}")

    failures = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        if not args.skip_train:
            for arch in archs:
                if train_csv_has(experiment="configs", arch=arch, seed=seed) \
                        or resolve_ckpt("out", arch, seed):
                    print(f"skip (done): train {arch} seed {seed}")
                    continue
                run_cell(["train.py", "--arch", arch, "--seed", str(seed),
                          "--experiment", "configs", "--amp",
                          "--eval_interval", "250", "--eval_iters", "20"],
                         args.dry_run, failures)
        if not args.skip_recall:
            for arch in archs:
                if recall_csv_has(experiment="configs", arch=arch, seed=seed,
                                  n_pairs=args.n_pairs,
                                  distance_mode="uniform"):
                    print(f"skip (done): recall {arch} seed {seed}")
                    continue
                run_cell(["recall.py", "--model", arch, "--seed",
                          str(seed), "--experiment", "configs",
                          "--n_pairs", str(args.n_pairs)],
                         args.dry_run, failures)

    finish("run_configs.py", failures)


if __name__ == "__main__":
    main()
