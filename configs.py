"""Model registry and shared helpers for the training, recall and driver scripts.

Registry keys:
    trissm              ternary SSM (TriSSM)
    fp_ssm              full-precision SSM, same architecture
    bitnet_transformer  ternary Transformer (attention and MLP)
    fp_transformer      full-precision Transformer
    qkvo_transformer    ternary Q/K/V/O projections, full-precision MLP
    v2t_<cfg>_fp / v2t_<cfg>_tern
                        parameter-matched Transformer configurations, in
                        full-precision and ternary form:
        v2t_deep      d_model=128, n_layers=8, n_heads=4, d_ff=1152
        v2t_mid       d_model=160, n_layers=6, n_heads=5, d_ff=960
        v2t_shallow   d_model=224, n_layers=2, n_heads=8, d_ff=1120
        v2t_widehead  d_model=192, n_layers=4, n_heads=2, d_ff=768
"""
import csv
import os
import subprocess
import sys

from model import TriSSM
from baselines import (TransformerLM, bitnet_transformer, fp_transformer,
                       qkvo_transformer)

ROOT = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Headline models.
# -----------------------------------------------------------------------------
MAIN_ARCHES = {
    "trissm": {
        "factory": lambda v: TriSSM(v, d_model=256, n_layers=12,
                                    d_state=16, quantized=True),
        "label": "TriSSM (1.58b)",
    },
    "fp_ssm": {
        "factory": lambda v: TriSSM(v, d_model=256, n_layers=12,
                                    d_state=16, quantized=False),
        "label": "FP SSM (16b)",
    },
    "bitnet_transformer": {
        "factory": lambda v: bitnet_transformer(v, d_model=192, n_layers=4,
                                                n_heads=8, d_ff=768),
        "label": "Ternary Transformer (1.58b)",
    },
    "fp_transformer": {
        "factory": lambda v: fp_transformer(v, d_model=192, n_layers=4,
                                            n_heads=8, d_ff=768),
        "label": "FP Transformer (16b)",
    },
}

# -----------------------------------------------------------------------------
# Parameter-matched Transformer configurations (~5M parameters).
# -----------------------------------------------------------------------------
_CONFIG_SPECS = {
    "v2t_deep":     dict(d_model=128, n_layers=8, n_heads=4,  d_ff=1152),
    "v2t_mid":      dict(d_model=160, n_layers=6, n_heads=5,  d_ff=960),
    "v2t_shallow":  dict(d_model=224, n_layers=2, n_heads=8,  d_ff=1120),
    "v2t_widehead": dict(d_model=192, n_layers=4, n_heads=2,  d_ff=768),
}

CONFIG_ARCHES = {}
for _name, _spec in _CONFIG_SPECS.items():
    for _suffix, _quant in (("_fp", False), ("_tern", True)):
        CONFIG_ARCHES[_name + _suffix] = {
            "factory": (lambda spec, q: (lambda v: TransformerLM(
                v, quantized=q, **spec)))(_spec, _quant),
            "label": (f"Transformer {_name[4:]} "
                      f"({'1.58b' if _quant else '16b'}) "
                      f"d{_spec['d_model']} L{_spec['n_layers']} "
                      f"h{_spec['n_heads']}"),
        }

# -----------------------------------------------------------------------------
# Control: ternary Q/K/V/O with a full-precision MLP.
# -----------------------------------------------------------------------------
QKVO_ARCHES = {
    "qkvo_transformer": {
        "factory": lambda v: qkvo_transformer(v, d_model=192, n_layers=4,
                                              n_heads=8, d_ff=768),
        "label": "QKVO-ternary Transformer (14.24b)",
    },
}

REGISTRY = {**MAIN_ARCHES, **CONFIG_ARCHES, **QKVO_ARCHES}

# Headline models, fastest to train first.
HEADLINE_ARCHES = ["bitnet_transformer", "fp_transformer", "trissm", "fp_ssm"]

LOG_DIR = "logs"
TRAIN_CSV = os.path.join(LOG_DIR, "train_results.csv")
RECALL_CSV = os.path.join(LOG_DIR, "recall_results.csv")
VARIANCE_CSV = os.path.join(LOG_DIR, "variance.csv")


def resolve_ckpt(ckpt_root, arch, seed):
    """Path of the trained checkpoint for (arch, seed), or None if absent."""
    path = os.path.join(ckpt_root, f"{arch}_s{seed}", "model.pt")
    return path if os.path.exists(path) else None


def append_csv(path, fieldnames, row):
    """Append one dict row to a CSV, writing the header if the file is new."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _csv_has(path, **match):
    """True if any row matches all key=value pairs (compared as strings)."""
    return any(all(row.get(k) == str(v) for k, v in match.items())
               for row in read_csv_rows(path))


def train_csv_has(**match):
    return _csv_has(TRAIN_CSV, **match)


def recall_csv_has(**match):
    return _csv_has(RECALL_CSV, **match)


def run_cell(cmd, dry_run, failures):
    """Run `python <script> <args>` from the repository root. `cmd` is
    [script, *args]; failed commands are appended to `failures`."""
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    rc = subprocess.run([sys.executable, os.path.join(ROOT, cmd[0])] + cmd[1:]).returncode
    if rc != 0:
        failures.append(" ".join(cmd))
        print(f"cell failed (rc={rc}): {' '.join(cmd)}", flush=True)


def finish(name, failures):
    """Print the driver summary and exit non-zero if any cell failed."""
    print(f"\n=== {name} finished ===")
    if failures:
        print(f"{len(failures)} failed cell(s):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("All cells completed (or skipped as done).")
