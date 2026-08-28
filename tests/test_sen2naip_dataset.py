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


def _make_synthetic_tree(root, n_rois=10):
    """Minimal stand-in for the SEN2NAIP synthetic component's layout: ROI_*/early|late/*.tif."""
    import numpy as np
    import rasterio
    for i in range(n_rois):
        for era in ("early", "late"):
            d = os.path.join(root, f"ROI_{i:04d}", era)
            os.makedirs(d)
            path = os.path.join(d, f"{era}__tile{i}.tif")
            with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=4,
                               dtype="uint8") as dst:
                dst.write(np.full((4, 8, 8), i, dtype=np.uint8))
    return root


def test_synthetic_split_is_by_roi_so_paired_eras_never_straddle_it(tmp_path):
    """The two eras of one ROI cover the SAME ground a decade apart. Splitting per-file would
    put the 2011 image of a field in train and the 2021 image of that same field in val, and the
    val score would then be inflated by near-duplicate leakage rather than measuring
    generalization. The split must therefore be by ROI."""
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    root = _make_synthetic_tree(str(tmp_path / "synthetic"))
    train, val = synthetic_component_files(root, era="both", val_fraction=0.2)

    def rois(paths):
        return {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths}

    assert rois(train) & rois(val) == set(), "an ROI appeared on both sides of the split"
    assert len(train) + len(val) == 20  # 10 ROIs x 2 eras
    # Both eras of any given ROI land together.
    for roi in rois(val):
        assert sum(1 for p in val if roi in p) == 2


def test_synthetic_era_selection_halves_the_file_count(tmp_path):
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    root = _make_synthetic_tree(str(tmp_path / "synthetic"))
    both_train, both_val = synthetic_component_files(root, era="both")
    early_train, early_val = synthetic_component_files(root, era="early")
    assert len(both_train) + len(both_val) == 2 * (len(early_train) + len(early_val))
    assert all("early" in os.path.dirname(p) for p in early_train + early_val)


def test_synthetic_split_is_deterministic_across_calls(tmp_path):
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    root = _make_synthetic_tree(str(tmp_path / "synthetic"))
    first = synthetic_component_files(root, era="both")
    second = synthetic_component_files(root, era="both")
    assert first == second


def test_synthetic_rejects_unknown_era(tmp_path):
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    root = _make_synthetic_tree(str(tmp_path / "synthetic"))
    with pytest.raises(ValueError, match="era must be"):
        synthetic_component_files(root, era="middle")


def test_synthetic_finds_rois_inside_extracted_shard_subdirectories(tmp_path):
    """Each shard extracts to its own subdirectory (`synthetic_01.zip` -> `synthetic_1/ROI_*/`),
    so the directory the shards were extracted into holds shard dirs, not ROI dirs. Pointing at
    that parent is the obvious thing to do, and before this it failed with a confusing "no ROI
    dirs" error after a 10 GB download."""
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    root = str(tmp_path / "synthetic")
    os.makedirs(root)
    _make_synthetic_tree(os.path.join(root, "synthetic_1"), n_rois=6)
    _make_synthetic_tree(os.path.join(root, "synthetic_2"), n_rois=6)

    train, val = synthetic_component_files(root, era="both", val_fraction=0.2)
    # _make_synthetic_tree numbers ROIs from 0 in each shard, so the two shards collide here by
    # construction; real ROI ids are globally unique (shard 01 alone spans 1..104577). What is
    # asserted is that ROIs below the shard level are found at all, and that the split still
    # partitions cleanly.
    assert len(train) + len(val) > 0
    assert set(train).isdisjoint(val)


def test_synthetic_still_works_when_pointed_directly_at_a_shard(tmp_path):
    from spectra_sr.sen2naip_dataset import synthetic_component_files
    shard = _make_synthetic_tree(str(tmp_path / "synthetic_1"), n_rois=10)
    train, val = synthetic_component_files(shard, era="both", val_fraction=0.2)
    assert len(train) + len(val) == 20


def _make_v2_pair(root, n_rois=6):
    """Stand-in for the SEN2NAIPv2 layout: 520/130 px tiles, uint16 for BOTH lr and hr."""
    import numpy as np
    import rasterio
    for i in range(n_rois):
        d = os.path.join(root, f"ROI_{i:05d}")
        os.makedirs(d)
        for name, size in (("lr", 130), ("hr", 520)):
            with rasterio.open(os.path.join(d, f"{name}.tif"), "w", driver="GTiff",
                               height=size, width=size, count=4, dtype="uint16") as dst:
                dst.write(np.full((4, size, size), 1500, dtype=np.uint16))
    return root


def test_v2_scales_hr_by_10000_not_255(tmp_path):
    """The single most dangerous difference between the releases. v1's HR is uint8 NAIP (/255);
    v2's is uint16 already in Sentinel-2 reflectance units (/10000). Applying the v1 rule to v2
    overshoots by ~40x and nothing raises -- the images are simply wrong."""
    from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset
    root = _make_v2_pair(str(tmp_path / "v2"))
    ds = SEN2NAIPCrossSensorDataset(root, hr_patch_size=384, crops_per_file=1, seed=0,
                                    variant="v2")
    lr, hr = ds[0]
    assert abs(float(hr.mean()) - 0.15) < 1e-4, "hr should be 1500/10000, not 1500/255"
    assert abs(float(lr.mean()) - 0.15) < 1e-4
    # Both sides land on the same radiometric scale, which is the point of v2's harmonization.
    assert abs(float(hr.mean() / lr.mean()) - 1.0) < 1e-3


def test_v2_disables_radiometric_calibration_by_default(tmp_path):
    """v2's HR is already harmonized to the Sentinel-2 scale (measured per-band HR/LR ratios
    1.000/0.999/0.999/1.000). Applying v1's affine calibration would INTRODUCE the error it
    exists to remove, so the default must come from the variant, not from a fixed default."""
    from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset
    root = _make_v2_pair(str(tmp_path / "v2"))
    assert SEN2NAIPCrossSensorDataset(root, hr_patch_size=384, crops_per_file=1,
                                      variant="v2").radiometric_calibration is False
    # ...but an explicit argument still wins, so the flag stays testable.
    assert SEN2NAIPCrossSensorDataset(root, hr_patch_size=384, crops_per_file=1, variant="v2",
                                      radiometric_calibration=True).radiometric_calibration is True


def test_v2_uses_its_own_tile_geometry(tmp_path):
    from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset
    root = _make_v2_pair(str(tmp_path / "v2"))
    ds = SEN2NAIPCrossSensorDataset(root, hr_patch_size=520, crops_per_file=1, variant="v2")
    assert (ds.hr_tile_size, ds.lr_tile_size) == (520, 130)
    lr, hr = ds[0]
    assert hr.shape[-1] == 520 and lr.shape[-1] == 130
    # 520 exceeds v1's 484 tile, so the same request must be rejected under v1.
    with pytest.raises(ValueError, match="exceeds the v1 tile size"):
        SEN2NAIPCrossSensorDataset(root, hr_patch_size=520, crops_per_file=1, variant="v1")


def test_unknown_variant_is_rejected(tmp_path):
    from spectra_sr.sen2naip_dataset import SEN2NAIPCrossSensorDataset
    root = _make_v2_pair(str(tmp_path / "v2"))
    with pytest.raises(ValueError, match="variant must be one of"):
        SEN2NAIPCrossSensorDataset(root, hr_patch_size=384, variant="v3")
