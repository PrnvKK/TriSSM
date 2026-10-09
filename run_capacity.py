"""Driver: memory-capacity sweep.

Runs recall.py for each (model, seed) over the number of key-value pairs in
--ns. Evaluation only: it loads checkpoints produced by train.py (see
run_seeds.py). N=10 is the headline setting and is covered by run_seeds.py, so
it is not in the default list.

Usage:
    python run_capacity.py --dry_run
    python run_capacity.py                          # headline models, seeds 42,43,44
    python run_capacity.py --models trissm,fp_ssm --ns 50,100
"""
import argparse

from configs import HEADLINE_ARCHES, finish, recall_csv_has, run_cell


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(HEADLINE_ARCHES))
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--ns", default="5,20,50,100",
                   help="Numbers of key-value pairs.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    models = args.models.split(",")
    ns = sorted(int(n) for n in args.ns.split(","))
    failures = []

    for seed in [int(s) for s in args.seeds.split(",")]:
        for arch in models:
            for n in ns:  # ascending N: cheap cells first
                if recall_csv_has(experiment="capacity", arch=arch,
                                  seed=seed, n_pairs=n,
                                  distance_mode="uniform"):
                    print(f"skip (done): capacity {arch} seed {seed} N={n}")
                    continue
                run_cell(["recall.py", "--model", arch, "--seed",
                          str(seed), "--experiment", "capacity",
                          "--n_pairs", str(n)], args.dry_run, failures)

    finish("run_capacity.py", failures)


if __name__ == "__main__":
    main()
