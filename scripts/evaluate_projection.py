"""Stage 5 gate: does the data-consistency projection actually help, or does it hurt?

`project_to_data_consistency` implements x <- x + A_dagger(y - A(x)), which makes the output
provably consistent with the real Sentinel-2 observation *under the operator A*. That caveat is
the whole question. Stage 0's operator has NOT been fitted to real Sentinel-2 -- it runs on a
placeholder sigma=1.0 -- so the projection enforces consistency with a guessed forward model, not
the true one. Enforcing the wrong constraint can be worse than enforcing none.

This measures both halves of the trade, on held-out data the model never saw:

  1. CONSISTENCY  ||A(x) - y||  -- must improve, or the projection is not doing its job at all.
  2. ACCURACY     PSNR against the true HR -- may improve or degrade. If A were exact this would
                  improve; the amount it degrades is a direct measure of how wrong A is.

Reported per lambda so the Tikhonov regularisation strength can be chosen on evidence.
The plan's own gate for this stage is "re-degraded output reproduces real LR input within
numerical precision" -- criterion 1. Criterion 2 decides whether shipping it is worth the cost.

Usage:
    python scripts/evaluate_projection.py --checkpoint checkpoints/pretrain_run10/checkpoint_epoch19.pt
"""
import argparse
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_sr.degradation import DegradationOperator
from spectra_sr.model import SpectraHATCore
from spectra_sr.projection import project_to_data_consistency
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.utils import DEVICE, logger
from train_pretrain import CONFIGS  # noqa: E402


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1))
    return float(10 * torch.log10(1.0 / (mse + 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--n-tiles", type=int, default=100)
    p.add_argument("--res-scale", type=float, default=None)
    p.add_argument("--lambdas", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1",
                   help="Which SEN2NAIP release --sen2naip-dir points at. Selects tile geometry, "
                        "HR scaling and whether radiometric calibration is applied. Wrong values "
                        "fail SILENTLY: v1's /255 HR rule applied to v2's uint16 reflectance "
                        "overshoots by ~40x, so every metric would be computed against nonsense "
                        "without anything raising.")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    cfg = CONFIGS[ckpt["config"]]
    rs = args.res_scale if args.res_scale is not None else ckpt.get("res_scale")
    if rs is not None and rs != cfg.res_scale:
        from dataclasses import replace
        cfg = replace(cfg, res_scale=rs)
    model = SpectraHATCore(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"{args.checkpoint} (epoch {ckpt['epoch']}, res_scale={cfg.res_scale})")

    degradation = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale).to(DEVICE)
    with torch.no_grad():
        degradation.log_sigma.fill_(torch.log(torch.tensor(1.0)))

    _, val_rois = _split_train_val_rois(args.sen2naip_dir, 0.2)
    ds = SEN2NAIPCrossSensorDataset(args.sen2naip_dir,
                                     hr_patch_size=cfg.train_patch_size * cfg.scale,
                                     crops_per_file=1, roi_list=val_rois, seed=1,
                                     variant=args.sen2naip_variant)
    n = min(args.n_tiles, len(val_rois))

    base_psnr, base_cons = [], []
    proj = {lam: {"psnr": [], "cons": []} for lam in args.lambdas}

    for i in range(n):
        lo, hi = ds[i]
        lo, hi = lo.unsqueeze(0).to(DEVICE), hi.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(lo)
            base_psnr.append(psnr(pred, hi))
            base_cons.append(float((degradation.forward(pred) - lo).abs().mean()))
            for lam in args.lambdas:
                projected = project_to_data_consistency(pred, lo, degradation,
                                                         tikhonov_lambda=lam)
                proj[lam]["psnr"].append(psnr(projected, hi))
                proj[lam]["cons"].append(float((degradation.forward(projected) - lo).abs().mean()))

    b_p, b_c = float(np.mean(base_psnr)), float(np.mean(base_cons))
    print()
    print("=" * 74)
    print(f"Stage 5 projection, {n} held-out tiles")
    print("=" * 74)
    print(f"{'setting':<18}{'PSNR vs HR':>13}{'d PSNR':>10}{'|A(x)-y|':>13}{'consistency':>15}")
    print(f"{'no projection':<18}{b_p:>13.3f}{'--':>10}{b_c:>13.6f}{'--':>15}")
    for lam in args.lambdas:
        pp, pc = float(np.mean(proj[lam]["psnr"])), float(np.mean(proj[lam]["cons"]))
        print(f"{'lambda=' + f'{lam:g}':<18}{pp:>13.3f}{pp - b_p:>+10.3f}"
              f"{pc:>13.6f}{(b_c - pc) / b_c * 100:>14.1f}%")
    print("=" * 74)
    print("consistency column = % reduction in ||A(x)-y||; positive is better.")
    print("d PSNR negative means the projection trades accuracy for consistency, which is the")
    print("expected cost when A is a placeholder rather than fitted to real Sentinel-2.")


if __name__ == "__main__":
    main()
