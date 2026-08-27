from spectra_sr.config import Config


def test_hr_patch_size_derived_from_scale():
    cfg = Config(scale_factor=4, lr_patch_size=128)
    assert cfg.hr_patch_size == 512


def test_default_bands_are_native_10m():
    cfg = Config()
    assert cfg.bands == ("B02", "B03", "B04", "B08")
