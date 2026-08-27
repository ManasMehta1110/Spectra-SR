import glob
import os

import pytest
import torch

from spectra_sr.dataset import NAIPPretrainDataset
from spectra_sr.degradation import DegradationOperator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAIP_DIR = os.path.join(REPO_ROOT, "data/raw/naip")
requires_naip = pytest.mark.skipif(
    not glob.glob(os.path.join(NAIP_DIR, "*.tif")), reason="no real NAIP file in data/raw/naip/"
)


@requires_naip
def test_len_matches_files_times_crops_per_file():
    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=10, seed=0)
    assert len(ds) == len(ds.files) * 10


@requires_naip
def test_item_shapes_are_consistent_with_scale():
    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=5, seed=0)
    lr, hr = ds[0]
    assert hr.shape == (4, 64, 64)
    assert lr.shape == (4, 16, 16)


@requires_naip
def test_crops_are_actually_randomized_not_fixed():
    """A dataset that always returns the same crop would silently defeat the point of
    `crops_per_file` -- verify two draws of the same index actually differ."""
    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=5, seed=0)
    _, hr_a = ds[0]
    _, hr_b = ds[0]
    assert not torch.equal(hr_a, hr_b), "expected different random crops across repeated draws"


@requires_naip
def test_excludes_in_progress_full_tile_temp_files(tmp_path):
    """Real bug this caught: acquire_naip.py's temporary "_full_*.tif" download-in-progress
    files can be sitting in the same directory as finished patches (a concurrent acquisition
    run was still writing one when this was first hit), and a naive "*.tif" glob picks them up
    as if they were valid, causing a real GDAL read error on the incomplete file."""
    import shutil
    real_file = glob.glob(os.path.join(NAIP_DIR, "*.tif"))[0]
    shutil.copy(real_file, tmp_path / "_full_should_be_excluded.tif")
    shutil.copy(real_file, tmp_path / "real_patch.tif")

    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(str(tmp_path), op, hr_patch_size=64, crops_per_file=1, seed=0)
    assert len(ds.files) == 1
    assert "_full_" not in ds.files[0]


@requires_naip
def test_sigma_domain_randomization_varies_within_range():
    """Real point of this feature: pretraining shouldn't be trained against one single guessed
    degradation strength (risk of learning habits tuned to a wrong assumption, per the team's
    own prior degradation-modeling finding) -- verify sigma actually varies draw-to-draw and
    stays within the configured range, not just that the dataset runs."""
    op = DegradationOperator(n_bands=4, scale=4)
    sigma_range = (0.5, 2.5)
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=20, seed=0,
                              sigma_range=sigma_range)

    observed_sigmas = []
    for i in range(10):
        ds[i]  # triggers the in-place randomization as a side effect
        observed_sigmas.extend(op.log_sigma.exp().tolist())

    assert len(set(round(s, 4) for s in observed_sigmas)) > 1, "sigma should vary across draws"
    assert all(sigma_range[0] <= s <= sigma_range[1] for s in observed_sigmas)


@requires_naip
def test_sigma_range_none_disables_randomization():
    """sigma_range=None should keep whatever sigma the operator was constructed/fit with,
    unchanged -- needed for the fine-tune tier later, where the operator will be fit to real
    data via fit_to_pairs() and must NOT be randomized away from that fitted value."""
    op = DegradationOperator(n_bands=4, scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.23)))
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=5, seed=0,
                              sigma_range=None)
    ds[0]
    assert torch.allclose(op.log_sigma.exp(), torch.full((4,), 1.23), atol=1e-4)


def test_undersized_files_are_skipped_not_fatal(tmp_path):
    """Real bug this caught: acquire_naip.py's bbox-intersection cropping can legitimately
    produce a patch smaller than requested (hit this on 4 of 51 real files, all near one
    state-border AOI where the real overlap was narrow in one dimension) -- one undersized file
    used to hard-crash NAIPPretrainDataset's constructor, killing an entire training run over a
    single bad tile. Build a real small image plus a real valid-size one and verify only the
    valid one survives, with the dataset still usable."""
    import numpy as np
    import rasterio

    small_path = str(tmp_path / "small.tif")
    valid_path = str(tmp_path / "valid.tif")
    profile = {"driver": "GTiff", "dtype": "uint8", "count": 4}

    with rasterio.open(small_path, "w", **profile, width=200, height=150) as dst:
        dst.write(np.zeros((4, 150, 200), dtype=np.uint8))
    with rasterio.open(valid_path, "w", **profile, width=512, height=512) as dst:
        dst.write(np.zeros((4, 512, 512), dtype=np.uint8))

    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(str(tmp_path), op, hr_patch_size=384, crops_per_file=5, seed=0)

    assert len(ds.files) == 1
    assert "valid.tif" in ds.files[0]
    lr, hr = ds[0]  # should not raise -- the surviving file is genuinely usable
    assert hr.shape == (4, 384, 384)


@requires_naip
def test_dataloader_batches_real_data_end_to_end():
    op = DegradationOperator(n_bands=4, scale=4)
    ds = NAIPPretrainDataset(NAIP_DIR, op, hr_patch_size=64, crops_per_file=8, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    lr_batch, hr_batch = next(iter(loader))
    assert lr_batch.shape == (4, 4, 16, 16)
    assert hr_batch.shape == (4, 4, 64, 64)
    assert torch.isfinite(lr_batch).all() and torch.isfinite(hr_batch).all()
