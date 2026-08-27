"""Stage 1 -- preprocessing. Plan Section 5, phase 2.

Adapted from optical_guided_sr.preprocessing (pattern reused, code rewritten for Sentinel-2's
band set): fill-masking, rasterio.warp.reproject-based co-registration, and -- critically -- the
"group files by parsed granule ID, never by list position" discipline that file's docstring
documents as a real prior bug (positional pairing silently drifts out of alignment the moment
region/band counts differ) -- carried forward into acquire_naip.py/acquire_sentinel2.py's own
filename-based (not index-based) pairing.

Two real pairing regimes, per plan Section 4.1's data tiers:
- Pretrain tier: no co-registration needed at all -- take real HR imagery (NAIP, or later
  archival India-relevant sources), synthesize the LR side via Stage 0's DegradationOperator.
  See spectra_sr.degradation.DegradationOperator.simulate/forward.
- Fine-tune/validate tiers: real Sentinel-2 paired against real independently-sourced HR
  reference (once PlanetScope or another Indian source lands) -- THIS needs `coregister`, since
  the two rasters come from different sensors, different native grids, and (for now, testing)
  different acquisition dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from rasterio.windows import from_bounds
from skimage.registration import phase_cross_correlation


def mask_fill_and_scale(arr: np.ndarray, fill_value: float = 0,
                         scale_factor: Optional[float] = None) -> np.ndarray:
    """Sentinel-2 L2A's fill/scale convention -- confirmed against a real downloaded scene
    (spectra_sr's acquire_sentinel2.py writes nodata as 0, DNs scaled x10000 per the standard
    L2A convention), NOT assumed to match HLS's -9999/0.0001 convention from
    optical_guided_sr.preprocessing._mask_fill (config.py flagged this as needing confirmation;
    confirmed here rather than left as an assumption).
    """
    arr = arr.astype(np.float32)
    arr[arr == fill_value] = np.nan
    if scale_factor is not None:
        arr = arr * scale_factor
    return arr


@dataclass(frozen=True)
class CoregistrationResult:
    output_path: str
    residual_px: Tuple[float, float]  # (row_shift, col_shift) from the phase-correlation pass


def crop_reference_to_footprint(hr_path: str, lr_reference_path: str, out_path: str,
                                 pad_fraction: float = 0.25) -> str:
    """Crop `lr_reference_path` down to the (padded) real-world footprint of `hr_path`, writing
    the result to `out_path`. Real production need, not just a test helper: pairing a small HR
    patch against a full multi-km LR scene means almost the entire reprojected destination is
    empty, which makes both training pairs and any residual/consistency check meaningless --
    found this the hard way testing coregister() against a real NAIP-patch/full-Sentinel-2-scene
    pair, where the reported "residual" was actually just detecting where the small nonzero
    patch happened to land in a mostly-empty array, not a real misalignment measurement.

    `pad_fraction=0.25` gives 25% padding on each side, so downstream co-registration has some
    real margin to search within rather than an exact, zero-slack crop.
    """
    with rasterio.open(hr_path) as hr:
        hr_bounds_wgs84 = transform_bounds(hr.crs, "EPSG:4326", *hr.bounds)

    with rasterio.open(lr_reference_path) as ref:
        ref_bounds_wgs84 = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
        left, bottom, right, top = hr_bounds_wgs84
        pad_x = (right - left) * pad_fraction
        pad_y = (top - bottom) * pad_fraction
        left, right = left - pad_x, right + pad_x
        bottom, top = bottom - pad_y, top + pad_y
        # Clip to the reference scene's own extent -- padding shouldn't request pixels outside
        # what the reference actually covers.
        left = max(left, ref_bounds_wgs84[0])
        bottom = max(bottom, ref_bounds_wgs84[1])
        right = min(right, ref_bounds_wgs84[2])
        top = min(top, ref_bounds_wgs84[3])

        window = from_bounds(left, bottom, right, top, transform=ref.transform)
        window = window.round_lengths().round_offsets()
        data = ref.read(window=window)
        profile = ref.profile.copy()
        profile.update(height=window.height, width=window.width,
                        transform=ref.window_transform(window))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return out_path


def coregister(hr_path: str, lr_reference_path: str, out_path: str,
               upsample_factor: int = 20) -> CoregistrationResult:
    """Reproject `hr_path` onto `lr_reference_path`'s grid (CRS/transform/shape), then measure
    residual sub-pixel misalignment via phase cross-correlation on the reprojected result versus
    the reference -- logged, not silently ignored, per plan Section 4.3's pairing protocol
    (residual >= 0.2 LR px should be treated as a bad pair upstream of this function, not fixed
    here; this function reports the residual, the caller decides whether to keep the pair).

    `upsample_factor=20` gives sub-pixel precision (1/20 px) in `phase_cross_correlation` --
    matches the plan's stated <0.2px target granularity.
    """
    with rasterio.open(lr_reference_path) as ref:
        ref_transform, ref_crs = ref.transform, ref.crs
        ref_width, ref_height = ref.width, ref.height
        ref_data = ref.read(1).astype(np.float32)

    with rasterio.open(hr_path) as src:
        src_data = src.read()
        target_profile = src.profile.copy()
        target_profile.update(
            crs=ref_crs, transform=ref_transform, width=ref_width, height=ref_height,
            driver="GTiff",
        )
        dest = np.zeros((src.count, ref_height, ref_width), dtype=src_data.dtype)
        for band in range(src.count):
            reproject(
                source=src_data[band], destination=dest[band],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

    with rasterio.open(out_path, "w", **target_profile) as dst:
        dst.write(dest)

    # Residual sub-pixel shift: compare the reprojected HR's first band against the LR reference
    # band, both now on the same grid/shape. NaN-safe: phase_cross_correlation doesn't tolerate
    # NaNs, so fill with the finite mean rather than 0 (0 would bias the correlation toward
    # whatever fraction of the image is nodata).
    a = dest[0].astype(np.float32)
    b = ref_data
    a = np.nan_to_num(a, nan=np.nanmean(a))
    b = np.nan_to_num(b, nan=np.nanmean(b))
    shift, _error, _diffphase = phase_cross_correlation(b, a, upsample_factor=upsample_factor)

    return CoregistrationResult(output_path=out_path, residual_px=(float(shift[0]), float(shift[1])))
