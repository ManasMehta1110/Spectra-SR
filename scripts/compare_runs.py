"""Compare several training runs' epoch logs on weight-independent metrics.

Built for the loss-weight sweep, where the obvious comparison is the wrong one: each arm
optimises a DIFFERENT objective (that is the point of the sweep), so `val_total_loss` is not
comparable across arms -- arm C can post a larger total loss purely because its spectral terms
carry bigger coefficients, while being the better model. Only metrics whose definition does not
depend on the loss weights can be compared, and this script reports exactly those:

    val_psnr, val_ssim_metric, val_rmse   -- fidelity
    val_sam_degrees                        -- spectral angle, in degrees (lower is better)
    val_downstream_improvement             -- NDVI-classification agreement vs bicubic

Reports the best epoch per arm per metric, not the last, since a run can peak mid-way; and
reports the control arm's own drift so a change can be attributed to the swept variable rather
than to the additional training every arm received.

Usage:
    python scripts/compare_runs.py checkpoints/sweep_a checkpoints/sweep_b --labels a b
"""
from __future__ import annotations

import argparse
import json
import os

# (key, lower_is_better)
METRICS = [
    ("val_psnr", False),
    ("val_ssim_metric", False),
    ("val_rmse", True),
    ("val_sam_degrees", True),
    ("val_downstream_improvement", False),
]


def load_epochs(run_dir: str) -> list:
    """Epoch-boundary records only. Intra-epoch step records are scored on a small fixed subset
    of the held-out set rather than all of it, so mixing them in would compare arms on different
    denominators."""
    path = os.path.join(run_dir, "epoch_log.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no epoch_log.jsonl in {run_dir}")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if not r.get("step_eval")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--control", default=None,
                   help="Label of the control arm; its drift is reported separately so a change "
                        "can be attributed to the swept variable rather than to extra training.")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/\\")) for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise SystemExit(f"{len(labels)} labels for {len(args.run_dirs)} run dirs")

    runs = {label: load_epochs(d) for label, d in zip(labels, args.run_dirs)}
    for label, rows in runs.items():
        print(f"{label:<16} {len(rows)} epochs")
    print()

    width = 14
    print(f"{'metric':<28}" + "".join(f"{label:>{width}}" for label in labels))
    print("-" * (28 + width * len(labels)))
    for key, lower_is_better in METRICS:
        cells = []
        for label in labels:
            values = [r[key] for r in runs[label] if key in r]
            best = min(values) if lower_is_better else max(values)
            cells.append(best)
        best_overall = min(cells) if lower_is_better else max(cells)
        row = f"{key:<28}"
        for value in cells:
            mark = "*" if value == best_overall else " "
            row += f"{value:>{width - 1}.4f}{mark}"
        print(row)
    print("\n(* = best arm for that metric; each cell is that arm's BEST epoch, not its last.)")

    if args.control and args.control in runs:
        print(f"\nControl arm '{args.control}' drift (first -> best epoch) -- any arm that only "
              f"matches this much movement has not beaten 'more training':")
        rows = runs[args.control]
        for key, lower_is_better in METRICS:
            values = [r[key] for r in rows if key in r]
            if not values:
                continue
            best = min(values) if lower_is_better else max(values)
            print(f"  {key:<28}{values[0]:>10.4f} -> {best:>10.4f}  ({best - values[0]:+.4f})")


if __name__ == "__main__":
    main()
