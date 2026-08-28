"""Rank the product-visualization gallery by PSNR gain over bicubic, so picking PPT images means
scrolling a short sorted list instead of 65 unordered files.

Deliberately computed independently of visualize_product.py rather than parsed from its log --
the gain is exactly what a PPT slide needs to defend ("this tile: +N dB over bicubic"), and
computing it directly from the same checkpoint/settings means the ranking can't drift out of
sync with a future run of the visualization script.

Usage:
    python scripts/rank_gallery_for_ppt.py --checkpoint checkpoints/pretrain_run10/checkpoint_best.pt \
        --res-scale 0.2 --recalibration 1.1907 --n-tiles 65
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spectra_sr.inference import load_for_inference, super_resolve
from spectra_sr.metrics import compute_metrics
from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset, _split_train_val_rois
from spectra_sr.utils import DEVICE, logger
from train_pretrain import CONFIGS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--n-tiles", type=int, default=65)
    p.add_argument("--res-scale", type=float, default=None)
    p.add_argument("--recalibration", type=float, default=1.0)
    p.add_argument("--sen2naip-variant", choices=["v1", "v2"], default="v1")
    args = p.parse_args()

    model, head, degradation, cfg = load_for_inference(args.checkpoint, CONFIGS, DEVICE)
    if args.res_scale is not None and args.res_scale != cfg.res_scale:
        from dataclasses import replace
        cfg = replace(cfg, res_scale=args.res_scale)
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        from spectra_sr.model import SpectraHATCore
        model = SpectraHATCore(cfg).to(DEVICE).eval()
        model.load_state_dict(ckpt["model"])

    _, val_rois = _split_train_val_rois(args.sen2naip_dir, 0.2)
    ds = SEN2NAIPCrossSensorDataset(args.sen2naip_dir,
                                    hr_patch_size=cfg.train_patch_size * cfg.scale,
                                    crops_per_file=1, roi_list=val_rois, seed=1,
                                    variant=args.sen2naip_variant)

    rows = []
    n = min(args.n_tiles, len(val_rois))
    for i in range(n):
        roi = val_rois[i]
        lo, hi = ds[i]
        lo, hi = lo.unsqueeze(0).to(DEVICE), hi.unsqueeze(0).to(DEVICE)
        r = super_resolve(lo, model, head, degradation, apply_projection=True,
                          uncertainty_recalibration=args.recalibration, run_checks=False)
        with torch.no_grad():
            # No post-hoc _match_radiometry on either side. That function's docstring calls
            # matching a lone baseline against the TRUE hr_target "LEGACY AND UNFAIR" -- it leaks
            # ground-truth statistics into whichever side receives it and was the exact root
            # cause of Bug 4 (docs/findings.md), which inverted a real result the same way. The
            # valid comparison relies on the calibration this dataset already applies UPSTREAM,
            # from train-split statistics only (SEN2NAIPCrossSensorDataset's
            # radiometric_calibration, on by default for v1): once `lo` is calibrated, bicubic
            # and the model both start from the same corrected input and neither needs a
            # per-sample fix against the answer.
            bic = F.interpolate(lo, size=hi.shape[-2:], mode="bicubic", align_corners=False)
            m_sr = compute_metrics(r.image, hi)
            m_bic = compute_metrics(bic, hi)
        rows.append({
            "roi": roi, "index": i,
            "sr_psnr": m_sr.psnr, "bic_psnr": m_bic.psnr, "gain_db": m_sr.psnr - m_bic.psnr,
            "sr_ssim": m_sr.ssim, "bic_ssim": m_bic.ssim,
            "consistency_error": r.consistency_error,
        })
        if (i + 1) % 20 == 0:
            logger.info(f"  scored {i + 1}/{n}")

    rows.sort(key=lambda x: x["gain_db"], reverse=True)

    print(f"\n{'rank':<5}{'ROI':<12}{'gain (dB)':>10}{'SR PSNR':>10}{'bicubic PSNR':>14}"
         f"{'SR SSIM':>10}{'consist. err':>14}")
    print("-" * 75)
    for rank, row in enumerate(rows, 1):
        print(f"{rank:<5}{row['roi']:<12}{row['gain_db']:>+10.2f}{row['sr_psnr']:>10.2f}"
             f"{row['bic_psnr']:>14.2f}{row['sr_ssim']:>10.4f}{row['consistency_error']:>14.5f}")

    print(f"\ntop 10 for the PPT (largest, most defensible gains):")
    print(", ".join(r["roi"] for r in rows[:10]))
    print(f"\nmedian-gain tiles (representative, not cherry-picked):")
    mid = len(rows) // 2
    print(", ".join(r["roi"] for r in rows[mid - 5:mid + 5]))
    print(f"\nworst 5 (for honest Q&A prep -- know your failure cases):")
    print(", ".join(r["roi"] for r in rows[-5:]))


if __name__ == "__main__":
    main()
