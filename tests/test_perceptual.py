import numpy as np
import pytest
import rasterio
import torch
import torch.nn.functional as F

from spectra_sr.losses import charbonnier_loss
from spectra_sr.perceptual import VGG16_TAPS, VGGPerceptualLoss


@pytest.fixture(scope="module")
def perceptual():
    return VGGPerceptualLoss().eval()


def test_zero_for_identical_inputs(perceptual):
    x = torch.rand(1, 4, 64, 64)
    assert perceptual(x, x).item() < 1e-5


def test_prefers_sharp_but_shifted_over_blurry(naip_primary_file, perceptual):
    """The entire reason this loss exists, asserted directly rather than assumed.

    Pixelwise losses are minimized by predicting the average of all high-resolution images
    consistent with a low-resolution input, and averaging edges at slightly different positions
    produces a blur -- so a pixelwise loss genuinely scores a blurry image BETTER than a sharp
    one that is a pixel off. Measured on real NAIP imagery: charbonnier rates blurry 0.032 vs
    shifted 0.039 (prefers blur), while this loss rates them 1.584 vs 1.193 (prefers the real
    edge). That inversion is the mechanism; if it ever stops holding, this term is not doing the
    job it was added for and the whole rationale needs revisiting, so pin it.
    """
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 256, 256)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0)[:, :4] / 255.0

    blurry = F.avg_pool2d(hr, kernel_size=5, stride=1, padding=2)
    sharp_but_shifted = torch.roll(hr, shifts=(1, 1), dims=(-2, -1))

    with torch.no_grad():
        assert charbonnier_loss(blurry, hr) < charbonnier_loss(sharp_but_shifted, hr), (
            "premise check: the pixelwise loss is expected to prefer the blurry candidate"
        )
        assert perceptual(sharp_but_shifted, hr) < perceptual(blurry, hr), (
            "perceptual loss must prefer a real-but-displaced edge over a smear"
        )


def test_penalizes_blur_relative_to_truth(naip_primary_file, perceptual):
    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 256, 256)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0)[:, :4] / 255.0
    mild = F.avg_pool2d(hr, kernel_size=3, stride=1, padding=1)
    heavy = F.avg_pool2d(hr, kernel_size=9, stride=1, padding=4)
    with torch.no_grad():
        assert perceptual(mild, hr) < perceptual(heavy, hr), "more blur must cost more"


def test_vgg_is_frozen_and_stays_in_eval_mode(perceptual):
    """A trainable or train-mode measuring instrument would drift during training and silently
    change what the loss means between epochs."""
    assert not any(p.requires_grad for p in perceptual.features.parameters())
    perceptual.train()  # simulate the parent model calling .train()
    assert not perceptual.features.training, "VGG must stay in eval mode after train() is called"


def test_gradients_reach_prediction_but_not_vgg(perceptual):
    pred = torch.rand(1, 4, 64, 64, requires_grad=True)
    target = torch.rand(1, 4, 64, 64)
    perceptual(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert all(p.grad is None for p in perceptual.features.parameters())


def test_network_is_truncated_at_deepest_requested_tap():
    """Guards a real memory concern on the 4GB dev GPU: evaluating VGG layers past the deepest
    tap is pure waste, and the full 31-layer stack on 384px patches alongside the SR model is a
    genuine OOM risk."""
    shallow = VGGPerceptualLoss(layers=("relu2_2",))
    assert len(shallow.features) == VGG16_TAPS["relu2_2"]
    deeper = VGGPerceptualLoss(layers=("relu2_2", "relu3_3"))
    assert len(deeper.features) == VGG16_TAPS["relu3_3"]


def test_rejects_unknown_layer_name():
    with pytest.raises(ValueError, match="Unknown VGG tap"):
        VGGPerceptualLoss(layers=("relu9_9",))


def test_rejects_mismatched_weights_length():
    with pytest.raises(ValueError, match="one entry per layer"):
        VGGPerceptualLoss(layers=("relu2_2", "relu3_3"), weights=(1.0,))


def test_combined_loss_includes_perceptual_only_when_weighted(naip_primary_file):
    from spectra_sr.degradation import DegradationOperator
    from spectra_sr.losses import SpectraCombinedLoss

    with rasterio.open(naip_primary_file) as src:
        hr_raw = src.read(window=rasterio.windows.Window(0, 0, 64, 64)).astype(np.float32)
    hr = torch.from_numpy(hr_raw).unsqueeze(0)[:, :4] / 255.0
    op = DegradationOperator(n_bands=4, scale=4)
    with torch.no_grad():
        lr = op.forward(hr)

    off = SpectraCombinedLoss(op, w_perceptual=0.0)
    assert off.perceptual is None
    assert "perceptual" not in off(hr, hr, lr)

    on = SpectraCombinedLoss(op, w_perceptual=1.0)
    assert on.perceptual is not None
    assert "perceptual" in on(hr, hr, lr)
