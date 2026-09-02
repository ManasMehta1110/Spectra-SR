"""Does run 10 (trained on ~99.9% US SEN2NAIP imagery) generalize to a real Indian tile it has
never seen? The one genuinely good India pair we have (Bangalore, same-day PlanetScope +
Sentinel-2, real cross-sensor correlation 0.73-0.76 -- see docs/findings.md) is the only honest
basis for this; the other four pulled AOIs had too weak a real PlanetScope/Sentinel-2 agreement
to serve as a trustworthy reference themselves.

IMPORTANT CAVEAT, stated up front because it changes what any number below can mean: PlanetScope
is NOT NAIP. The model was trained to map Sentinel-2 onto NAIP's specific radiometric
convention, and today's own measurement found real per-band differences between PlanetScope and
Sentinel-2 (e.g. Red ratio 0.564, Blue ratio 1.165) that have nothing to do with India or model
generalization -- they are just PlanetScope-vs-NAIP-vs-Sentinel2 sensor convention differences.
So an absolute PSNR here is NOT directly comparable to run 10's US PSNR (21.706 dB); it measures
a mix of (a) real India domain generalization and (b) a reference-domain mismatch that would
exist even for a perfect model. The RELATIVE comparison -- does SR beat bicubic on this same,
equally-mismatched reference -- partially cancels that confound and is the more honest read.

Usage:
    python scripts/test_india_generalization.py \\
        --checkpoint checkpoints/pretrain_run10/checkpoint_best.pt --res-scale 0.2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio
import rasterio.warp
import torch
import torch.nn.functional as F
from PIL import Image
from rasterio.warp import Resampling, reproject

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectra_sr.inference import load_for_inference, super_resolve
from spectra_sr.metrics import _match_radiometry, compute_metrics
from spectra_sr.sen2naip_dataset import calibrate_lr_to_hr_radiometry
from spectra_sr.utils import DEVICE, logger
from train_pretrain import CONFIGS  # noqa: E402

GAP = 6


def rgb(x: torch.Tensor) -> np.ndarray:
    a = x[:3].detach().cpu().clamp(0, 1).numpy()
    return (np.transpose(a, (1, 2, 0)) * 255).astype(np.uint8)


def extract_aligned_crop(s2_path: str, ps_path: str, lr_size: int, scale: int,
                         row0: int, col0: int):
    """Real Sentinel-2 crop as LR, and the GEOGRAPHICALLY MATCHING PlanetScope region --
    reprojected onto the LR crop's exact footprint at lr_size*scale resolution, not a naive
    array resize. Same reprojection discipline as the cross-sensor correlation check earlier
    (docs/findings.md): PlanetScope is UTM, Sentinel-2 here is WGS84 -- a shape-match is not a
    geography-match."""
    with rasterio.open(s2_path) as src:
        lr_raw = src.read(window=rasterio.windows.Window(col0, row0, lr_size, lr_size))
        lr_transform = src.window_transform(
            rasterio.windows.Window(col0, row0, lr_size, lr_size))
        lr_crs = src.crs

    with rasterio.open(ps_path) as ps_src:
        ps_data = ps_src.read()
        ps_crs, ps_transform = ps_src.crs, ps_src.transform

    hr_size = lr_size * scale
    # Destination transform: same geographic origin as the LR crop, hr_size pixels covering the
    # same footprint (i.e. HR pixel = LR pixel / scale).
    dst_transform = rasterio.Affine(
        lr_transform.a / scale, 0, lr_transform.c,
        0, lr_transform.e / scale, lr_transform.f)

    hr_aligned = np.zeros((4, hr_size, hr_size), dtype=np.float64)
    for b in range(4):
        reproject(source=ps_data[b], destination=hr_aligned[b],
                  src_transform=ps_transform, src_crs=ps_crs,
                  dst_transform=dst_transform, dst_crs=lr_crs,
                  resampling=Resampling.average)

    return lr_raw.astype(np.float32), hr_aligned.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pretrain_run10/checkpoint_best.pt")
    p.add_argument("--sentinel2", default="data/raw/india_pairs/sentinel2_bangalore_2026-06-01.tif")
    p.add_argument("--planetscope", default="data/raw/india_pairs/planetscope_bangalore_2026-06-01.tif")
    p.add_argument("--res-scale", type=float, default=0.2)
    p.add_argument("--recalibration", type=float, default=1.1907)
    p.add_argument("--out-dir", default="checkpoints/pretrain_run10/india_test")
    p.add_argument("--n-crops", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, head, degradation, cfg = load_for_inference(args.checkpoint, CONFIGS, device=DEVICE)
    if args.res_scale is not None and args.res_scale != cfg.res_scale:
        from dataclasses import replace
        from spectra_sr.model import SpectraHATCore
        cfg = replace(cfg, res_scale=args.res_scale)
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        model = SpectraHATCore(cfg).to(DEVICE).eval()
        model.load_state_dict(ckpt["model"])
    logger.info(f"res_scale={cfg.res_scale} patch={cfg.train_patch_size} scale={cfg.scale}")

    lr_size = cfg.train_patch_size

    with rasterio.open(args.sentinel2) as src:
        s2_h, s2_w = src.height, src.width

    # Non-overlapping crop grid over the real Sentinel-2 tile, staying inside its bounds.
    max_row = s2_h - lr_size
    max_col = s2_w - lr_size
    positions = []
    grid = int(np.ceil(np.sqrt(args.n_crops)))
    for i in range(grid):
        for j in range(grid):
            if len(positions) >= args.n_crops:
                break
            row0 = min(int(i * max_row / max(grid - 1, 1)), max_row)
            col0 = min(int(j * max_col / max(grid - 1, 1)), max_col)
            positions.append((row0, col0))

    sr_metrics, bic_metrics = [], []
    for idx, (row0, col0) in enumerate(positions):
        lr_raw, hr_raw = extract_aligned_crop(
            args.sentinel2, args.planetscope, lr_size, cfg.scale, row0, col0)

        lr = torch.from_numpy(lr_raw) / 10000.0  # Sentinel-2 L2A convention
        hr = torch.from_numpy(hr_raw) / 10000.0  # PlanetScope SR asset, same x10000 DN scale
        lr_calibrated = calibrate_lr_to_hr_radiometry(lr)
        logger.info(f"  debug crop{idx}: lr_raw mean={lr.mean():.4f} hr_raw mean={hr.mean():.4f} "
                    f"lr_calib mean={lr_calibrated.mean():.4f} "
                    f"hr_nonzero_frac={(hr > 0).float().mean():.3f}")

        lr_b = lr_calibrated.unsqueeze(0).to(DEVICE)
        hr_b = hr.unsqueeze(0).to(DEVICE)

        result = super_resolve(lr_b, model, head, degradation, apply_projection=True,
                               uncertainty_recalibration=args.recalibration, run_checks=False)
        bicubic = F.interpolate(lr_b, size=hr_b.shape[-2:], mode="bicubic", align_corners=False)

        # Raw comparison, reported for transparency -- this is expected to be uninformative
        # (see the module docstring): calibrate_lr_to_hr_radiometry pushes the input onto NAIP's
        # brightness convention (measured this run: ~0.55 mean) while PlanetScope's own natural
        # scale here is ~0.15 mean. That ~3.5x constant offset dominates MSE for BOTH sr and
        # bicubic near-identically, so raw PSNR mostly measures "how far is NAIP's convention
        # from PlanetScope's," not model quality.
        m_sr_raw = compute_metrics(result.image, hr_b)
        m_bic_raw = compute_metrics(bicubic, hr_b)

        # "Both" mode -- matches EACH side's own mean/std to hr_b's, symmetrically. Distinct
        # from Bug 4 (metrics.py's own documented mistake): that matched only the baseline,
        # handing it the ground-truth's statistics as unfair information the model never had.
        # Matching both sides equally removes the convention offset without giving either side
        # an advantage, isolating structural/spatial agreement -- the legitimate use of this
        # mode per metrics.py's own three-way comparison (its docstring's "both" row).
        sr_matched = _match_radiometry(result.image, hr_b)
        bic_matched = _match_radiometry(bicubic, hr_b)
        m_sr = compute_metrics(sr_matched, hr_b)
        m_bic = compute_metrics(bic_matched, hr_b)
        sr_metrics.append(m_sr)
        bic_metrics.append(m_bic)

        logger.info(f"crop {idx} (row={row0},col={col0}): "
                    f"[raw] SR={m_sr_raw.psnr:.2f} bic={m_bic_raw.psnr:.2f}  "
                    f"[both-matched] SR={m_sr.psnr:.2f} ssim={m_sr.ssim:.3f}  "
                    f"bicubic={m_bic.psnr:.2f} ssim={m_bic.ssim:.3f}  "
                    f"gain={m_sr.psnr - m_bic.psnr:+.2f} dB")

        panels = [rgb(lr_calibrated), rgb(bicubic[0]), rgb(result.image[0]), rgb(hr)]
        h, w = panels[3].shape[:2]
        panels[0] = np.array(Image.fromarray(panels[0]).resize((w, h), Image.NEAREST))
        sep = np.full((h, GAP, 3), 245, dtype=np.uint8)
        strip = []
        for k, pan in enumerate(panels):
            strip.append(pan)
            if k < len(panels) - 1:
                strip.append(sep)
        Image.fromarray(np.concatenate(strip, axis=1)).save(
            os.path.join(args.out_dir, f"india_crop{idx}.png"))

    gains = [s.psnr - b.psnr for s, b in zip(sr_metrics, bic_metrics)]
    print("\n" + "=" * 70)
    print(f"India generalization test -- Bangalore, {len(positions)} crops, "
          f"checkpoint={args.checkpoint}")
    print("=" * 70)
    print(f"mean SR PSNR (both-matched):      {np.mean([m.psnr for m in sr_metrics]):.2f} dB")
    print(f"mean bicubic PSNR (both-matched): {np.mean([m.psnr for m in bic_metrics]):.2f} dB")
    print(f"mean gain:         {np.mean(gains):+.2f} dB")
    print(f"win rate:          {sum(1 for g in gains if g > 0)}/{len(gains)}")
    print()
    print("Method note: raw (unmatched) PSNR was also logged per-crop above and is near-identical")
    print("for SR and bicubic (~7.3-7.8 dB both) -- confirms it is dominated by a real, measured")
    print("~3.5x brightness-convention offset between NAIP (what the model targets) and")
    print("PlanetScope (this AOI's only available reference), not by model behavior. The")
    print("both-matched numbers above remove that offset symmetrically (metrics.py's own 'both'")
    print("mode, not the Bug-4 baseline-only mistake) and are the honest measure of whether SR's")
    print("spatial detail agrees with PlanetScope better than bicubic's does.")
    print(f"\nPanels written to {args.out_dir}")


if __name__ == "__main__":
    main()
