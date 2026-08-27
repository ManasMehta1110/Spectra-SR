import glob
import os

import pytest
import torch

from spectra_sr.sen2naip_dataset import (
    HR_BAND_MEAN, HR_TILE_SIZE, LR_TILE_SIZE, NATIVE_SCALE, SEN2NAIPCrossSensorDataset,
    _split_train_val_rois, calibrate_lr_to_hr_radiometry,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEN2NAIP_DIR = os.path.join(REPO_ROOT, "data/raw/sen2naip/cross-sensor/cross-sensor")
requires_sen2naip = pytest.mark.skipif(
    not glob.glob(os.path.join(SEN2NAIP_DIR, "ROI_*")),
    reason="no real SEN2NAIP cross-sensor data in data/raw/sen2naip/",
)


@requires_sen2naip
def test_len_matches_rois_times_crops():
    ds = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=10, seed=0)
    assert len(ds) == len(ds.rois) * 10


@requires_sen2naip
def test_item_shapes_are_consistent_with_scale():
    ds = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=5, seed=0)
    lr, hr = ds[0]
    assert hr.shape == (4, 64, 64)
    assert lr.shape == (4, 16, 16)  # 64 / NATIVE_SCALE (4)


@requires_sen2naip
def test_crops_are_actually_randomized_not_fixed():
    ds = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=5, seed=0)
    _, hr_a = ds[0]
    _, hr_b = ds[0]
    assert not torch.equal(hr_a, hr_b), "expected different random crops across repeated draws"


@requires_sen2naip
def test_values_are_real_reflectance_and_naip_ranges_not_garbage():
    """Real bug-catcher: a wrong normalization constant (e.g. /255 on the LR side, which is
    actually x10000-scaled Sentinel-2 reflectance) would silently produce values wildly outside
    a plausible [0,1]-ish range rather than crashing -- worth asserting the real range directly."""
    ds = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=5, seed=0,
                                     radiometric_calibration=False)
    lr, hr = ds[0]
    assert 0.0 <= hr.min() and hr.max() <= 1.0
    assert 0.0 <= lr.min() and lr.max() < 2.0  # real S2 reflectance can exceed 1.0 (snow/cloud/etc)


@requires_sen2naip
def test_radiometric_calibration_moves_lr_onto_the_hr_scale():
    """The real bug this fixes: SpectraHATCore predicts bicubic(lr) + residual, which assumes lr
    and hr share a radiometric scale. Real Sentinel-2 vs. real NAIP differ by 2-5x per band, and
    pretrain_run5 confirmed the cost for real -- washed-out structureless output, 9 points below a
    bicubic baseline. Verify calibration actually closes the gap rather than just running."""
    uncal = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=1, seed=0,
                                        radiometric_calibration=False)
    cal = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=1, seed=0,
                                      radiometric_calibration=True)

    lr_uncal, _ = uncal[0]
    lr_cal, _ = cal[0]
    hr_means = torch.tensor(HR_BAND_MEAN)

    gap_uncal = (lr_uncal.mean(dim=(-2, -1)) - hr_means).abs().mean()
    gap_cal = (lr_cal.mean(dim=(-2, -1)) - hr_means).abs().mean()
    assert gap_cal < gap_uncal / 2, (
        f"calibration should substantially close the LR/HR radiometric gap "
        f"(uncalibrated {gap_uncal:.4f} -> calibrated {gap_cal:.4f})"
    )


def test_calibration_uses_fixed_constants_not_per_sample_hr_statistics():
    """Guards against a subtle but serious form of cheating: matching each LR sample to ITS OWN
    paired HR's statistics would leak the ground truth into the model's input, and would silently
    inflate every validation number while being impossible to reproduce at real inference time
    (where no HR exists). Two different synthetic inputs must be transformed by the same fixed
    affine map, independent of any HR."""
    a = torch.full((4, 8, 8), 0.1)
    b = torch.full((4, 8, 8), 0.2)
    out_a = calibrate_lr_to_hr_radiometry(a)
    out_b = calibrate_lr_to_hr_radiometry(b)
    # Same fixed scale factor applied to both, so equal input deltas produce equal output deltas.
    delta = (out_b - out_a)[:, 0, 0]
    expected = torch.tensor([0.1]) * (
        torch.tensor([0.15607, 0.12043, 0.11127, 0.11993])
        / torch.tensor([0.07874, 0.05099, 0.04068, 0.08075])
    )
    assert torch.allclose(delta, expected, atol=1e-4)


@requires_sen2naip
def test_hr_patch_size_must_be_divisible_by_native_scale():
    with pytest.raises(ValueError, match="divisible"):
        SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=65, crops_per_file=1)


@requires_sen2naip
def test_hr_patch_size_exceeding_real_tile_size_raises():
    with pytest.raises(ValueError, match="exceeds"):
        SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=HR_TILE_SIZE + 4, crops_per_file=1)


@requires_sen2naip
def test_split_train_val_rois_is_deterministic_and_disjoint():
    train_a, val_a = _split_train_val_rois(SEN2NAIP_DIR, val_fraction=0.2)
    train_b, val_b = _split_train_val_rois(SEN2NAIP_DIR, val_fraction=0.2)
    assert train_a == train_b and val_a == val_b, "split must be deterministic across calls"
    assert set(train_a).isdisjoint(set(val_a)), "train/val ROIs must never overlap"
    all_rois = set(
        d for d in os.listdir(SEN2NAIP_DIR)
        if os.path.isdir(os.path.join(SEN2NAIP_DIR, d)) and d.startswith("ROI_")
    )
    assert set(train_a) | set(val_a) == all_rois, "every real ROI must land in train or val"
    assert val_a, "val split should never be empty for a real-sized dataset"


@requires_sen2naip
def test_dataloader_batches_real_data_end_to_end():
    ds = SEN2NAIPCrossSensorDataset(SEN2NAIP_DIR, hr_patch_size=64, crops_per_file=8, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    lr_batch, hr_batch = next(iter(loader))
    assert lr_batch.shape == (4, 4, 16, 16)
    assert hr_batch.shape == (4, 4, 64, 64)
    assert torch.isfinite(lr_batch).all() and torch.isfinite(hr_batch).all()
