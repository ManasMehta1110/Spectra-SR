"""Visualise the complete delivered product, not just the super-resolved image.

`visualize_predictions.py` shows LR / bicubic / SR / truth. That is the accuracy story. This
adds the two panels that carry the PS's other requirements: the calibrated per-pixel uncertainty
map, and the error map it is supposed to predict.

Placing predicted uncertainty directly beside actual error is the honest presentation -- if the
uncertainty map is meaningful the two should light up in the same places, and if it is not, that
is visible at a glance rather than hidden behind an ECE number.

Usage:
    python scripts/visualize_product.py --checkpoint checkpoints/pretrain_run10/checkpoint_epoch19.pt \
        --res-scale 0.2 --recalibration 1.1907 --n-samples 6 --out-dir <dir>
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_sr.inference import load_for_inference, super_resolve
from spectra_sr.metrics import _match_radiometry
from spectra_sr.model import SpectraHATCore
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.utils import DEVICE, logger
from train_pretrain import CONFIGS  # noqa: E402

GAP = 6  # px of separator between panels


def rgb(x: torch.Tensor) -> np.ndarray:
    a = x[:3].detach().cpu().clamp(0, 1).numpy()
    return (np.transpose(a, (1, 2, 0)) * 255).astype(np.uint8)


def heat(x: torch.Tensor, vmax: float) -> np.ndarray:
    """Single-channel magnitude -> perceptually ordered dark->amber->white ramp. Shared `vmax`
    across the uncertainty and error panels so the two are directly comparable by eye; scaling
    each to its own range would make any uncertainty map look like any error map."""
    v = (x.detach().cpu().numpy() / max(vmax, 1e-8)).clip(0, 1)
    stops = np.array([[10, 14, 24], [40, 30, 80], [150, 55, 70], [225, 130, 40], [255, 245, 220]],
                     dtype=np.float32)
    pos = np.linspace(0, 1, len(stops))
    out = np.stack([np.interp(v, pos, stops[:, c]) for c in range(3)], axis=-1)
    return out.astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=6)
    p.add_argument("--res-scale", type=float, default=None)
    p.add_argument("--recalibration", type=float, default=1.0)
    p.add_argument("--no-projection", action="store_true")
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1",
                   help="Which SEN2NAIP release --sen2naip-dir points at. Selects tile geometry, "
                        "HR scaling and whether radiometric calibration is applied. Wrong values "
                        "fail SILENTLY: v1's /255 HR rule applied to v2's uint16 reflectance "
                        "overshoots by ~40x, so every metric would be computed against nonsense "
                        "without anything raising.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, head, degradation, cfg = load_for_inference(args.checkpoint, CONFIGS, DEVICE)
    if args.res_scale is not None and args.res_scale != cfg.res_scale:
        cfg = replace(cfg, res_scale=args.res_scale)
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model = SpectraHATCore(cfg).to(DEVICE).eval()
        model.load_state_dict(ckpt["model"])
    logger.info(f"res_scale={cfg.res_scale} recalibration={args.recalibration} "
                f"projection={'off' if args.no_projection else 'on'}")

    _, val_rois = _split_train_val_rois(args.sen2naip_dir, 0.2)
    ds = SEN2NAIPCrossSensorDataset(args.sen2naip_dir,
                                     hr_patch_size=cfg.train_patch_size * cfg.scale,
                                     crops_per_file=1, roi_list=val_rois, seed=1,
                                     variant=args.sen2naip_variant)

    for i in range(min(args.n_samples, len(val_rois))):
        lo, hi = ds[i]
        lo, hi = lo.unsqueeze(0).to(DEVICE), hi.unsqueeze(0).to(DEVICE)
        r = super_resolve(lo, model, head, degradation,
                          apply_projection=not args.no_projection,
                          uncertainty_recalibration=args.recalibration, run_checks=False)
        with torch.no_grad():
            bic = F.interpolate(lo, size=hi.shape[-2:], mode="bicubic", align_corners=False)
            lo_up = F.interpolate(lo, size=hi.shape[-2:], mode="nearest")
            err = (r.image - hi).abs().mean(dim=1)[0]
            unc = r.uncertainty_std.mean(dim=1)[0]

        vmax = float(max(err.max(), unc.max()))
        panels = [rgb(_match_radiometry(lo_up, hi)[0]), rgb(_match_radiometry(bic, hi)[0]),
                  rgb(r.image[0]), rgb(hi[0]), heat(unc, vmax), heat(err, vmax)]
        h, w = panels[0].shape[:2]
        sep = np.full((h, GAP, 3), 245, dtype=np.uint8)
        strip = []
        for j, pan in enumerate(panels):
            strip.append(pan)
            if j < len(panels) - 1:
                strip.append(sep)
        Image.fromarray(np.concatenate(strip, axis=1)).save(
            os.path.join(args.out_dir, f"{val_rois[i]}_product.png"))
        print(f"  {val_rois[i]}  psnr-order: LR | bicubic | SR | truth | uncertainty | actual error")

    print(f"\nWrote {min(args.n_samples, len(val_rois))} panels to {args.out_dir}")
    print("Uncertainty and error panels share one colour scale, so they are directly comparable.")


if __name__ == "__main__":
    main()
