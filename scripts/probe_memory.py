"""Measure peak training VRAM for a config/batch-size grid, so the big run's settings are chosen
from measurement rather than from hope.

Why this exists: FULL has never been trained. It OOM'd on the 4.3GB dev laptop even at batch=4,
which tells us nothing useful about a 16GB T4 or a 40GB A100 -- and "launch a 10-hour Colab job
and find out at minute three" is an expensive way to discover a batch size. Run this on the
target machine first; it takes about a minute per cell.

Measures a full training step (forward + loss + backward + optimizer step), not just a forward
pass, because backward is what actually peaks: activations for every attention block stay live
until their gradients are consumed. A forward-only probe would under-report by roughly 2-3x and
would be worse than no measurement at all, since it reads as authoritative.

Usage:
    python scripts/probe_memory.py                          # default grid
    python scripts/probe_memory.py --configs full --batch-sizes 4 8 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from spectra_sr.degradation import DegradationOperator
from spectra_sr.losses import SpectraCombinedLoss
from spectra_sr.model import SpectraHATCore
from spectra_sr.uncertainty import UncertaintyHead, heteroscedastic_nll_loss
from train_pretrain import CONFIGS  # noqa: E402


def probe(cfg, batch_size: int, device, steps: int = 3) -> dict:
    """One cell of the grid. Returns peak allocated/reserved MiB, or an oom flag.

    Runs several steps rather than one: the first step allocates the optimizer state lazily
    (Adam creates its moment buffers on first `.step()`), so a single-step probe misses roughly
    2 x model-size of permanent memory and would under-report the steady state.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        model = SpectraHATCore(cfg).to(device)
        head = UncertaintyHead(n_bands=cfg.n_bands).to(device)
        degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale).to(device)
        criterion = SpectraCombinedLoss(degradation).to(device)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=2e-4)

        lr_size = cfg.train_patch_size
        hr_size = lr_size * cfg.scale
        for _ in range(steps):
            lr_img = torch.rand(batch_size, cfg.n_bands, lr_size, lr_size, device=device)
            hr_img = torch.rand(batch_size, cfg.n_bands, hr_size, hr_size, device=device)

            optimizer.zero_grad()
            pred = model(lr_img)
            loss_terms = criterion(pred, hr_img, lr_img)
            residual = lr_img - degradation.forward(pred)
            residual_up = torch.nn.functional.interpolate(
                residual, size=pred.shape[-2:], mode="nearest")
            log_var = head(pred.detach(), residual_up)
            total = loss_terms["total"] + heteroscedastic_nll_loss(pred.detach(), hr_img, log_var)
            total.backward()
            optimizer.step()

        peak_alloc = torch.cuda.max_memory_allocated(device) / 2 ** 20
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 20
        n_params = sum(p.numel() for p in model.parameters())
        result = {"ok": True, "peak_alloc_mib": peak_alloc, "peak_reserved_mib": peak_reserved,
                  "params": n_params}
    except torch.cuda.OutOfMemoryError:
        result = {"ok": False, "peak_alloc_mib": None, "peak_reserved_mib": None, "params": None}

    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["colab_realistic", "full"],
                   choices=list(CONFIGS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--out", default="results/memory_probe.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device -- this probe measures GPU memory and has nothing to report.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_mib = props.total_memory / 2 ** 20
    print(f"device: {props.name}  total: {total_mib:,.0f} MiB\n")

    rows = []
    print(f"{'config':<18}{'batch':>7}{'patch':>7}{'params':>12}"
          f"{'peak MiB':>12}{'% of GPU':>10}")
    print("-" * 66)
    for config_name in args.configs:
        cfg = CONFIGS[config_name]
        for batch_size in args.batch_sizes:
            r = probe(cfg, batch_size, device, steps=args.steps)
            r.update({"config": config_name, "batch_size": batch_size,
                      "train_patch_size": cfg.train_patch_size})
            rows.append(r)
            if r["ok"]:
                pct = 100 * r["peak_reserved_mib"] / total_mib
                print(f"{config_name:<18}{batch_size:>7}{cfg.train_patch_size:>7}"
                      f"{r['params']:>12,}{r['peak_reserved_mib']:>12,.0f}{pct:>9.1f}%")
            else:
                print(f"{config_name:<18}{batch_size:>7}{cfg.train_patch_size:>7}"
                      f"{'--':>12}{'OOM':>12}{'--':>10}")
                break  # larger batches on this config will also OOM

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": props.name, "total_mib": total_mib, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")
    print("Pick the largest batch that stays under ~85% of GPU -- validation allocates on top of "
          "training peak, and fragmentation over a long run needs headroom.")


if __name__ == "__main__":
    main()
