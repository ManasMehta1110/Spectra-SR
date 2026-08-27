"""Tests against REAL downloaded data (data/raw/), not synthetic arrays -- Stage 1's whole job is
handling real sensor/file quirks, so its tests should too. Skipped (not failed) if the relevant
real files aren't present, e.g. on a machine that hasn't run the acquisition scripts yet.
"""
import glob
import os

import numpy as np
import pytest
import rasterio
import rasterio.warp

from spectra_sr.preprocessing import coregister, crop_reference_to_footprint, mask_fill_and_scale

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2_PUNJAB = glob.glob(os.path.join(REPO_ROOT, "data/raw/sentinel2/*.tif"))
S2_CA = glob.glob(os.path.join(REPO_ROOT, "data/raw/sentinel2_ca/*.tif"))

requires_s2 = pytest.mark.skipif(not S2_PUNJAB, reason="no real Sentinel-2 file in data/raw/sentinel2/")
requires_matched_pair = pytest.mark.skipif(
    not S2_CA, reason="no real Sentinel-2 California scene in data/raw/sentinel2_ca/"
)  # naip_primary_file fixture handles the "no NAIP file" skip case


@requires_s2
def test_mask_fill_and_scale_on_real_sentinel2_scene():
    with rasterio.open(S2_PUNJAB[0]) as src:
        raw = src.read()

    scaled = mask_fill_and_scale(raw, fill_value=0, scale_factor=1 / 10000)

    # Real L2A reflectance should land in a physically sensible range -- not [0, 65535] raw DNs,
    # and not everything collapsed to 0/1 the way the first (buggy) Sentinel-2 pull was.
    finite = scaled[~np.isnan(scaled)]
    assert finite.min() >= 0
    assert finite.max() < 2.0, "scaled reflectance should be roughly in [0, ~1.5], not raw DNs"
    assert finite.mean() > 0.01, "shouldn't have collapsed toward zero"


@requires_s2
def test_mask_fill_and_scale_actually_masks_zeros():
    with rasterio.open(S2_PUNJAB[0]) as src:
        raw = src.read()
    raw = raw.copy()
    raw[0, 0, 0] = 0  # force a real fill pixel

    scaled = mask_fill_and_scale(raw, fill_value=0, scale_factor=1 / 10000)
    assert np.isnan(scaled[0, 0, 0])


@requires_matched_pair
def test_crop_reference_to_footprint_matches_hr_extent(tmp_path, naip_primary_file):
    """A real bug this caught: coregistering a small (~300m) NAIP patch directly against a full
    (~5.5km) Sentinel-2 scene reprojects almost entirely into empty space, and the "residual"
    from phase correlation just detects where the tiny nonzero patch landed in a mostly-blank
    array -- 185px/-84px, not a real misalignment measurement. Cropping the reference to the
    HR patch's own footprint first is what makes the residual measurement meaningful."""
    cropped_path = str(tmp_path / "cropped_ref.tif")
    crop_reference_to_footprint(naip_primary_file, S2_CA[0], cropped_path, pad_fraction=0.25)

    with rasterio.open(naip_primary_file) as hr, rasterio.open(cropped_path) as cropped:
        hr_bounds_wgs84 = rasterio.warp.transform_bounds(hr.crs, "EPSG:4326", *hr.bounds)
        # Cropped reference should be small (matching the HR patch plus padding), not the full
        # multi-km Sentinel-2 scene.
        assert cropped.width < 100
        assert cropped.height < 100
        # And should genuinely bound/overlap the HR patch's real footprint, not some other area.
        assert cropped.bounds.left <= hr_bounds_wgs84[0]
        assert cropped.bounds.right >= hr_bounds_wgs84[2]


@requires_matched_pair
def test_coregister_real_pair_produces_matching_grid_and_reasonable_residual(tmp_path, naip_primary_file):
    """The real, honest test: crop the Sentinel-2 scene down to the NAIP patch's real footprint
    first (see test above), then reproject NAIP onto that cropped grid. Both over the same
    California AOI, different sensors/dates -- see conversation: this is a code-correctness
    check using whatever real matching pair is available, not a claim that California is part
    of the actual validation set."""
    cropped_ref_path = str(tmp_path / "cropped_ref.tif")
    crop_reference_to_footprint(naip_primary_file, S2_CA[0], cropped_ref_path, pad_fraction=0.25)

    out_path = str(tmp_path / "coregistered.tif")
    result = coregister(naip_primary_file, cropped_ref_path, out_path)

    with rasterio.open(cropped_ref_path) as ref, rasterio.open(result.output_path) as out:
        assert out.width == ref.width
        assert out.height == ref.height
        assert out.crs == ref.crs

    # Now that both cover the same real footprint, the residual should be small -- real
    # cross-sensor/cross-date imagery still won't be pixel-perfect, but it shouldn't be the
    # hundred-plus-pixel garbage from comparing against a mismatched-scale reference.
    row_shift, col_shift = result.residual_px
    assert np.isfinite(row_shift) and np.isfinite(col_shift)
    assert abs(row_shift) < 10 and abs(col_shift) < 10, (
        f"expected a small residual now the footprints match, got {result.residual_px}"
    )
