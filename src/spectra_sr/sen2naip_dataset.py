"""Real (not synthetic) Sentinel-2/HR pairs from the SEN2NAIP cross-sensor dataset (Ostertagova
et al., Scientific Data 2024; isp-uv-es/SEN2NAIP on Hugging Face). 2,851 same-day-acquired
Sentinel-2 L2A / NAIP pairs, US-based (NAIP coverage) but a REAL sensor-degradation pair, not our
own DegradationOperator's synthetic guess -- directly tests whether that synthetic degradation
model itself was contributing to the blur-plateau problem confirmed via visualize_predictions.py
(pretrain_run2/3/4 all converged to bicubic-equivalent output).

Every ROI_XXXX directory has a fixed-size hr.tif (484x484x4, uint8, 2.5m) and lr.tif (121x121x4,
int32 Sentinel-2 L2A surface reflectance x10000, 10m) -- confirmed via direct inspection of the
real downloaded files, not assumed from documentation. No degradation simulation needed here:
lr.tif *is* the real Sentinel-2 observation, already paired with its real HR reference.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch

from .utils import logger

# SEN2NAIP's lr.tif files carry a 4th band without a matching ExtraSamples tag, so GDAL emits
# "Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel"
# on EVERY open. The files read correctly (verified: all 2,851 pairs, consistent 4-band shapes) --
# this is a metadata-tag nit in the published dataset, not a data problem. Left unsuppressed it
# emits two warning lines per sample, which buried the actual per-epoch training results in
# hundreds of megabytes of noise.
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

HR_TILE_SIZE = 484
LR_TILE_SIZE = 121
NATIVE_SCALE = 4  # 484/121 -- matches this project's scale_factor exactly, not a coincidence:
                  # SEN2NAIP was built for the same 10m Sentinel-2 -> 2.5m NAIP 4x SR task.

# Cross-sensor radiometric calibration constants, per band (B02/B03/B04/B08 order), measured over
# a 300-ROI sample of the TRAIN split only (never val -- these are applied at inference too, so
# deriving them from held-out data would leak).
#
# Why this is required, not cosmetic: SpectraHATCore predicts `bicubic(lr) + residual`
# (model.py:375-377). That skip assumes lr and hr share a radiometric scale. Real Sentinel-2 and
# real NAIP do NOT -- per-band means differ by 2.0-5.3x and stds by 1.5-2.7x. Without calibration
# the residual branch has to synthesize that entire affine transform before it can model any
# detail, and it demonstrably fails: pretrain_run5 epoch 3 produced washed-out, color-shifted
# output with almost no structure, PSNR 20.7 vs ~28 on same-scale data, and scored 9 points BELOW
# a bicubic baseline. This is the "cross-sensor radiometric calibration" step the plan's own
# pairing protocol (Section 4.3) requires "from the first batch of data onward".
LR_BAND_MEAN = (0.12571, 0.10533, 0.07370, 0.27888)
LR_BAND_STD = (0.07874, 0.05099, 0.04068, 0.08075)
HR_BAND_MEAN = (0.48517, 0.47471, 0.38936, 0.55600)
HR_BAND_STD = (0.15607, 0.12043, 0.11127, 0.11993)

# Per-release geometry and radiometry. SEN2NAIPv2's cross-sensor release differs from v1 in ways
# that are silent rather than loud if ignored -- every one of these was measured from the real
# archives, not assumed:
#
#   * tiles are 520/130 px, not 484/121;
#   * HR is uint16 in Sentinel-2 reflectance units, NOT uint8 NAIP. Dividing v2's HR by 255 (the
#     v1 rule) overshoots by ~40x and silently trains on nonsense -- nothing raises, the images
#     are simply wrong;
#   * HR has ALREADY been harmonized to the Sentinel-2 radiometric scale. Measured over the
#     first 20 ROIs, per-band HR/LR mean ratios are 1.000/0.999/0.999/1.000 -- max deviation
#     0.07%, against v1's 2.0-5.3x per-band mismatch. Applying v1's affine calibration to v2
#     would therefore INTRODUCE the very error it exists to remove.
DATASET_VARIANTS = {
    "v1": {
        "hr_tile_size": 484, "lr_tile_size": 121,
        "hr_divisor": 255.0,     # NAIP uint8
        "lr_divisor": 10000.0,   # Sentinel-2 L2A reflectance, scaled x10000
        "radiometric_calibration": True,
    },
    "v2": {
        "hr_tile_size": 520, "lr_tile_size": 130,
        "hr_divisor": 10000.0,   # already in Sentinel-2 reflectance units
        "lr_divisor": 10000.0,
        "radiometric_calibration": False,
    },
}


def calibrate_lr_to_hr_radiometry(lr: torch.Tensor) -> torch.Tensor:
    """Map real Sentinel-2 reflectance onto the NAIP radiometric scale with a fixed per-band
    affine transform. Fixed constants (not per-sample statistics matched against the paired HR)
    -- a per-sample match would leak the ground truth into the model's own input, which would be
    cheating at inference time where no HR exists. `lr` is (C, H, W) or (B, C, H, W)."""
    shape = (-1, 1, 1) if lr.dim() == 3 else (1, -1, 1, 1)
    lr_mean = torch.tensor(LR_BAND_MEAN, dtype=lr.dtype, device=lr.device).reshape(shape)
    lr_std = torch.tensor(LR_BAND_STD, dtype=lr.dtype, device=lr.device).reshape(shape)
    hr_mean = torch.tensor(HR_BAND_MEAN, dtype=lr.dtype, device=lr.device).reshape(shape)
    hr_std = torch.tensor(HR_BAND_STD, dtype=lr.dtype, device=lr.device).reshape(shape)
    return (lr - lr_mean) / lr_std * hr_std + hr_mean


def filter_rois_by_quality(root_dir: str, rois: List[str], qa1_max: Optional[float] = None,
                            qa2_max: Optional[float] = None) -> List[str]:
    """Filter ROIs on SEN2NAIP's own published per-pair quality metadata.

    QA1 is spatial misalignment between the LR/HR pair in pixels (MAE over ground-control points
    matched by LightGlue/DISK); QA2 is spectral angle distance in degrees. The dataset authors
    already excluded QA1>1px and QA2>2deg, but the surviving distribution is still wide -- median
    QA1 is 0.680 px across all 2,851 pairs.

    Why tightening this matters more than it looks: super-resolution is asked to place
    high-frequency detail precisely. If the HR target's edges sit ~0.68 px from where the LR
    input implies, and that offset varies per tile, then the L1/Charbonnier-optimal prediction
    for an edge is a BLURRED edge -- the objective is mathematically paying the model to smear.
    That is the same blur-regression failure seen across pretrain_run1-5, and no gradient-loss
    weight can outvote it, because it fights the data rather than the penalty. Trading tile count
    for alignment quality is therefore a direct attack on the root cause, not a data-hygiene nicety.
    """
    if qa1_max is None and qa2_max is None:
        return rois
    kept = []
    for roi in rois:
        meta_path = os.path.join(root_dir, roi, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        qa1, qa2 = meta.get("QA1"), meta.get("QA2")
        if qa1_max is not None and (qa1 is None or qa1 > qa1_max):
            continue
        if qa2_max is not None and (qa2 is None or qa2 > qa2_max):
            continue
        kept.append(roi)
    logger.info(f"Quality filter (QA1<={qa1_max}, QA2<={qa2_max}): kept {len(kept)}/{len(rois)} ROIs")
    return kept


def _split_train_val_rois(root_dir: str, val_fraction: float = 0.2,
                           qa1_max: Optional[float] = None,
                           qa2_max: Optional[float] = None) -> Tuple[List[str], List[str]]:
    """Same deterministic-split-by-sorted-name discipline as
    train_pretrain.py's _split_train_val_files -- not re-randomized per seed, so the same ROIs
    are always held out regardless of run.

    Quality filtering is applied BEFORE the split, so train and val are drawn from the same
    (filtered) distribution -- validating on badly-registered pairs the model was never trained
    on would measure the registration noise rather than the model.
    """
    rois = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ROI_")
    )
    rois = filter_rois_by_quality(root_dir, rois, qa1_max, qa2_max)
    if len(rois) < 5:
        raise ValueError(f"Only {len(rois)} ROI dirs in {root_dir} -- too few for a val split.")
    n_val = max(1, round(len(rois) * val_fraction))
    val_rois = rois[::len(rois) // n_val][:n_val]
    train_rois = [r for r in rois if r not in val_rois]
    return train_rois, val_rois


class SEN2NAIPCrossSensorDataset(torch.utils.data.Dataset):
    """Random aligned crops from real SEN2NAIP cross-sensor pairs. Crop offset is drawn in LR
    pixel space, then scaled by NATIVE_SCALE for the HR read -- keeps the two reads exactly
    aligned without needing per-file reprojection (both tifs already share the same footprint,
    confirmed via identical `bounds` on the demo pair)."""

    def __init__(self, root_dir: str, hr_patch_size: int = 384, crops_per_file: int = 20,
                 roi_list: Optional[List[str]] = None, seed: Optional[int] = None,
                 radiometric_calibration: Optional[bool] = None, variant: str = "v1",
                 deterministic: bool = False):
        if variant not in DATASET_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(DATASET_VARIANTS)}, got {variant!r}")
        spec = DATASET_VARIANTS[variant]
        hr_tile_size, lr_tile_size = spec["hr_tile_size"], spec["lr_tile_size"]
        # An explicit argument still wins, so the flag remains testable; but the DEFAULT must come
        # from the variant, because the right answer differs between releases and a silently
        # wrong default here produces plausible-looking, badly-scaled imagery rather than an error.
        if radiometric_calibration is None:
            radiometric_calibration = spec["radiometric_calibration"]

        if hr_patch_size % NATIVE_SCALE != 0:
            raise ValueError(f"hr_patch_size ({hr_patch_size}) must be divisible by "
                              f"NATIVE_SCALE ({NATIVE_SCALE})")
        lr_patch_size = hr_patch_size // NATIVE_SCALE
        if hr_patch_size > hr_tile_size or lr_patch_size > lr_tile_size:
            raise ValueError(f"hr_patch_size={hr_patch_size} (lr={lr_patch_size}) exceeds the "
                              f"{variant} tile size (hr={hr_tile_size}, lr={lr_tile_size})")

        self.rois = roi_list if roi_list is not None else sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("ROI_")
        )
        if not self.rois:
            raise ValueError(f"No ROI_* directories found in {root_dir}")
        self.root_dir = root_dir
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size
        self.crops_per_file = crops_per_file
        self.radiometric_calibration = radiometric_calibration
        self.variant = variant
        self.hr_tile_size = hr_tile_size
        self.lr_tile_size = lr_tile_size
        self.hr_divisor = spec["hr_divisor"]
        self.lr_divisor = spec["lr_divisor"]
        self.seed = seed
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rois) * self.crops_per_file

    def __getitem__(self, idx: int):
        roi = self.rois[idx % len(self.rois)]
        roi_dir = os.path.join(self.root_dir, roi)

        max_lr_xy = self.lr_tile_size - self.lr_patch_size
        if self.deterministic:
            # A crop offset that is a pure function of (seed, idx), not a draw consumed from
            # self.rng's shared, advancing stream. Matters specifically because this dataset is
            # constructed ONCE and reused across every epoch of a training run: with the shared
            # stream, epoch 0's validation pass consumes hundreds of draws from self.rng, so
            # epoch 1's call to the SAME idx lands wherever that stream left off and reads a
            # DIFFERENT crop of the SAME ROI -- silently mixing crop-sampling noise into every
            # epoch-to-epoch validation comparison (confirmed: this dataset's val_dataset is
            # built once before train_pretrain.py's epoch loop, not once per epoch). A per-item
            # generator seeded by (seed, idx) makes ds[idx] return the identical crop no matter
            # how many other items were read first, so "epoch 0 vs epoch N" compares the same
            # pixels every time. Intentionally NOT the default: training benefits from crops
            # varying across epochs (implicit augmentation), so only validation/eval callers
            # should opt in.
            item_rng = np.random.default_rng((self.seed or 0, idx))
            x0_lr = int(item_rng.integers(0, max_lr_xy + 1))
            y0_lr = int(item_rng.integers(0, max_lr_xy + 1))
        else:
            x0_lr = int(self.rng.integers(0, max_lr_xy + 1))
            y0_lr = int(self.rng.integers(0, max_lr_xy + 1))
        x0_hr, y0_hr = x0_lr * NATIVE_SCALE, y0_lr * NATIVE_SCALE

        with rasterio.open(os.path.join(roi_dir, "lr.tif")) as src:
            lr_raw = src.read(window=rasterio.windows.Window(
                x0_lr, y0_lr, self.lr_patch_size, self.lr_patch_size))
        with rasterio.open(os.path.join(roi_dir, "hr.tif")) as src:
            hr_raw = src.read(window=rasterio.windows.Window(
                x0_hr, y0_hr, self.hr_patch_size, self.hr_patch_size))

        # Real Sentinel-2 L2A surface reflectance, scaled x10000 -- same convention already
        # defined (but unused until now) as Config.reflectance_scale.
        lr = torch.from_numpy(lr_raw.astype(np.float32)) / self.lr_divisor
        hr = torch.from_numpy(hr_raw.astype(np.float32)) / self.hr_divisor
        if self.radiometric_calibration:
            lr = calibrate_lr_to_hr_radiometry(lr)
        return lr, hr


def synthetic_component_files(root_dir: str, era: str = "both",
                              val_fraction: float = 0.2) -> Tuple[List[str], List[str]]:
    """Train/val file lists over the SEN2NAIP *synthetic* component.

    That component is structurally NOTHING like the cross-sensor one this module's Dataset
    class reads, which is why it gets a plain file-list helper rather than a Dataset of its own:
    a synthetic ROI holds `early/<naip>.tif` and `late/<naip>.tif` and its metadata carries
    `s2_id: null`, `QA1: null`, `QA2: null` -- there is NO Sentinel-2 imagery in it at all. It is
    HR NAIP only (1100x1100, 4-band uint8, 2.5m), and the LR side has to be synthesized.

    Synthesizing LR from HR NAIP is exactly what `NAIPPretrainDataset` already does, including
    the uint8 rescale, the band-count reconciliation and the per-sample sigma randomization. So
    this returns file lists to hand that class via its `file_list=` argument instead of
    duplicating tested crop/degradation logic here.

    Note what this data is and is not. Because the LR side is simulated by our own operator, a
    model trained on it learns to invert a degradation we chose -- an easier and different task
    than the real cross-sensor one. Its role is PRETRAINING volume (plan Section 5, phase 3:
    "trained on synthetic pairs, then fine-tuned on real Tier 2 pairs"); the real cross-sensor
    pairs remain the only honest basis for reported accuracy.

    `era` selects which acquisitions to use: "early", "late", or "both". "both" treats the two
    dates as independent HR tiles, which roughly doubles the file count -- they share a footprint
    but are separate acquisitions a decade apart, so their surface content genuinely differs.

    The split is by ROI, never by file: with era="both" the two acquisitions of one ROI cover the
    SAME ground, so splitting per-file would put a 2011 image of a field in train and a 2021
    image of that same field in val, and the resulting val score would be inflated by
    near-duplicate leakage rather than measuring generalization.
    """
    if era not in ("early", "late", "both"):
        raise ValueError(f"era must be 'early', 'late' or 'both', got {era!r}")

    # Each shard extracts to its OWN subdirectory (synthetic_01.zip -> `synthetic_1/ROI_*/...`),
    # so the natural root -- the directory the shards were extracted into -- contains shard dirs
    # rather than ROI dirs. Accept either, because pointing at the shard parent is the obvious
    # thing to do and would otherwise fail with a confusing "no ROI dirs" error after a 10 GB
    # download. ROI ids are drawn from one global space (shard 01 alone spans 1..104577), so
    # merging shards cannot collide.
    roi_dirs = {}  # ROI name -> absolute path, across however many shards are present
    for entry in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        if entry.startswith("ROI_"):
            roi_dirs[entry] = path
        else:
            for sub in sorted(os.listdir(path)):
                if sub.startswith("ROI_") and os.path.isdir(os.path.join(path, sub)):
                    roi_dirs[sub] = os.path.join(path, sub)

    rois = sorted(roi_dirs)
    if len(rois) < 5:
        raise ValueError(f"Only {len(rois)} ROI dirs found under {root_dir} (searched it and one "
                         f"level below, for extracted shard subdirectories) -- too few for a val "
                         f"split.")

    # Same deterministic strided split as _split_train_val_rois, so the held-out set is stable
    # across runs and independent of seed.
    n_val = max(1, round(len(rois) * val_fraction))
    val_rois = set(rois[::len(rois) // n_val][:n_val])

    eras = ("early", "late") if era == "both" else (era,)
    train_files, val_files = [], []
    for roi in rois:
        target = val_files if roi in val_rois else train_files
        for sub in eras:
            target.extend(sorted(glob.glob(os.path.join(roi_dirs[roi], sub, "*.tif"))))

    logger.info(f"SEN2NAIP synthetic ({era}): {len(train_files)} train files from "
                f"{len(rois) - len(val_rois)} ROIs, {len(val_files)} val files from "
                f"{len(val_rois)} ROIs")
    return train_files, val_files
