"""Driver: multi-seed headline experiment.

Training: for each seed, train the headline models (Transformers first, SSMs
last). Cells already present in logs/train_results.csv are skipped.

Recall: for each seed, run the headline recall cell (N=10 pairs, uniform query
distance, 3000-step adaptation). Cells already present in
logs/recall_results.csv are skipped.

Cells run sequentially. A half-finished training run can be continued with
    python train.py --arch trissm --seed 43 --resume

Usage:
    python run_seeds.py --dry_run          # print the commands
    python run_seeds.py                    # seeds 42,43,44
    python run_seeds.py --skip_train       # recall cells only
"""
import argparse

from configs import (HEADLINE_ARCHES, finish, recall_csv_has, run_cell,
                     train_csv_has)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_seeds", default="42,43,44",
                   help="Seeds to pretrain.")
    p.add_argument("--eval_seeds", default="42,43,44",
                   help="Seeds to evaluate for recall.")
    p.add_argument("--models", default=",".join(HEADLINE_ARCHES))
    p.add_argument("--n_pairs", type=int, default=10)
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_recall", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    models = args.models.split(",")
    failures = []

    if not args.skip_train:
        for seed in [int(s) for s in args.train_seeds.split(",")]:
            for arch in models:
                if train_csv_has(experiment="headline", arch=arch, seed=seed):
                    print(f"skip (done): train {arch} seed {seed}")
                    continue
                run_cell(["train.py", "--arch", arch, "--seed", str(seed),
                          "--experiment", "headline"], args.dry_run, failures)

    if not args.skip_recall:
        for seed in [int(s) for s in args.eval_seeds.split(",")]:
            for arch in models:
                if recall_csv_has(experiment="headline", arch=arch,
                                  seed=seed, n_pairs=args.n_pairs,
                                  distance_mode="uniform"):
                    print(f"skip (done): recall {arch} seed {seed}")
                    continue
                run_cell(["recall.py", "--model", arch, "--seed",
                          str(seed), "--experiment", "headline",
                          "--n_pairs", str(args.n_pairs)],
                         args.dry_run, failures)

    finish("run_seeds.py", failures)


if __name__ == "__main__":
    main()
