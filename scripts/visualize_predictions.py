"""Qualitative check, not just aggregate metrics: save real LR / bicubic / SR / true-HR
comparison images from a trained checkpoint on held-out val tiles.

Built specifically because every diagnosis so far (pretrain_run1-4) has been aggregate numbers
(PSNR/SSIM/downstream %) -- we've never actually looked at what the model produces. This answers
something the metrics can't: is the model basically learning to approximate bicubic (a common
failure mode when the loss stack is this blur-tolerant), or doing something visibly different
that the NDVI-threshold downstream check just isn't sensitive to.

Usage:
    python scripts/visualize_predictions.py --checkpoint checkpoints/pretrain_run4/checkpoint_best.pt \
        --naip-dir data/raw/naip --out-dir checkpoints/pretrain_run4/viz --n-samples 3
"""
import argparse
import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image

from spectra_sr.degradation import DegradationOperator
from spectra_sr.metrics import _match_radiometry
from spectra_sr.model import SpectraHATCore
from spectra_sr.utils import DEVICE

# Same deterministic split logic (and CONFIGS registry) as train_pretrain.py -- imported directly
# so these are genuinely the same held-out files the checkpoint never trained on.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_pretrain import CONFIGS, _split_train_val_files  # noqa: E402


def to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """(C,H,W) float tensor in [0,1], C>=3 -- take first 3 bands as R,G,B (NAIP band order) and
    clamp/convert for saving. Not a radiometrically correct visualization, just a real one."""
    rgb = x[:3].detach().cpu().clamp(0, 1).numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    return (rgb * 255).astype(np.uint8)


def _iter_naip_synthetic(args, cfg, hr_patch_size):
    """Yields (name, lr, hr) on DEVICE. LR is synthesized from HR by Stage 0's operator, matching
    how the naip_synthetic training path builds its pairs."""
    _, val_files = _split_train_val_files(args.naip_dir, args.val_fraction)
    print(f"Held-out val files ({len(val_files)}): {[os.path.basename(f) for f in val_files]}")

    op = DegradationOperator(n_bands=cfg.n_bands, scale=cfg.scale).to(DEVICE)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))  # same fixed val sigma as training

    for f in val_files:
        with rasterio.open(f) as src:
            if src.width < hr_patch_size or src.height < hr_patch_size or src.count < cfg.n_bands:
                continue
            hr_raw = src.read(window=rasterio.windows.Window(0, 0, hr_patch_size, hr_patch_size))
        hr = torch.from_numpy(hr_raw.astype(np.float32)).unsqueeze(0).to(DEVICE) / 255.0
        hr = hr[:, :cfg.n_bands]
        with torch.no_grad():
            lr = op.simulate(hr)
        yield os.path.splitext(os.path.basename(f))[0], lr, hr


def _iter_sen2naip(args, cfg, hr_patch_size):
    """Yields (name, lr, hr) on DEVICE from real SEN2NAIP cross-sensor pairs -- no degradation
    simulation, lr.tif IS the real Sentinel-2 observation. Same normalization constants and
    LR/HR crop alignment as SEN2NAIPCrossSensorDataset, so what's visualized is genuinely what
    the model trained/validated on."""
    from spectra_sr.sen2naip_dataset import NATIVE_SCALE, _split_train_val_rois

    _, val_rois = _split_train_val_rois(args.sen2naip_dir, args.val_fraction)
    print(f"Held-out val ROIs ({len(val_rois)}), showing first {args.n_samples}")
    lr_patch_size = hr_patch_size // NATIVE_SCALE

    for roi in val_rois:
        roi_dir = os.path.join(args.sen2naip_dir, roi)
        with rasterio.open(os.path.join(roi_dir, "lr.tif")) as src:
            lr_raw = src.read(window=rasterio.windows.Window(0, 0, lr_patch_size, lr_patch_size))
        with rasterio.open(os.path.join(roi_dir, "hr.tif")) as src:
            hr_raw = src.read(window=rasterio.windows.Window(0, 0, hr_patch_size, hr_patch_size))
        lr = torch.from_numpy(lr_raw.astype(np.float32)).unsqueeze(0).to(DEVICE) / 10000.0
        hr = torch.from_numpy(hr_raw.astype(np.float32)).unsqueeze(0).to(DEVICE) / 255.0
        yield roi, lr, hr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--naip-dir", default="data/raw/naip")
    p.add_argument("--sen2naip-dir", default="data/raw/sen2naip/cross-sensor/cross-sensor")
    p.add_argument("--data-source", choices=["naip_synthetic", "sen2naip"],
                   default="naip_synthetic")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    cfg = CONFIGS[ckpt["config"]]
    model = SpectraHATCore(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (config={ckpt['config']}, epoch={ckpt['epoch']}, "
          f"val_total_loss={ckpt['val_total_loss']:.4f}, data_source={args.data_source})")

    hr_patch_size = cfg.train_patch_size * cfg.scale
    source = (_iter_sen2naip if args.data_source == "sen2naip" else _iter_naip_synthetic)

    n_done = 0
    for name, lr, hr in source(args, cfg, hr_patch_size):
        if n_done >= args.n_samples:
            break
        with torch.no_grad():
            pred = model(lr)
            bicubic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
            # Same radiometric matching the downstream metric applies (metrics._match_radiometry)
            # -- without it the real-Sentinel-2 bicubic panel renders far darker than the NAIP HR
            # panel, and the comparison reads as a brightness difference rather than the sharpness
            # difference actually being judged.
            bicubic_shown = _match_radiometry(bicubic, hr)
            lr_shown = _match_radiometry(
                F.interpolate(lr, size=hr.shape[-2:], mode="nearest"), hr)

        panels = [lr_shown[0], bicubic_shown[0], pred[0], hr[0]]
        combined = np.concatenate([to_uint8_rgb(v) for v in panels], axis=1)
        Image.fromarray(combined).save(os.path.join(args.out_dir, f"{name}_comparison.png"))
        print(f"  saved {name}_comparison.png  (order: LR-nearest | bicubic | SR output | true HR)")
        n_done += 1

    if n_done == 0:
        print("No usable val samples found -- nothing saved.")
    else:
        print(f"\nDone: {n_done} comparison image(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
