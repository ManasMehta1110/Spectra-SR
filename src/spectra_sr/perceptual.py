"""Perceptual (feature-space) loss -- the established fix for blur regression, and the one
mechanism the previous five pretraining runs never touched.

Why this and not another pixelwise term: Charbonnier/SSIM/Sobel all score a prediction position
by position. Given a low-resolution input, many different sharp high-resolution images are
consistent with it, and a positionally-scored loss is minimized by predicting the AVERAGE of
those candidates -- which is blurry, because averaging edges at slightly different positions
smears them. Adding a gradient penalty (pretrain_run3/5/6, weights 0.3 and 2.5) fights that
tendency with a counter-penalty but does not remove the incentive; measured result was a flat,
inert gradient term and a model that plateaued ~0.8-9 dB below its own bicubic baseline.

A perceptual loss scores predictions by the distance between deep CNN feature activations
instead. Those features respond to the PRESENCE of texture and edge structure rather than to its
exact pixel position, so a sharp edge rendered a fraction of a pixel off is no longer punished
more harshly than no edge at all. That directly removes the incentive to smear -- which matters
doubly here, since SEN2NAIP pairs carry real sub-pixel misregistration (median QA1 0.680 px).

Band handling, stated honestly: VGG16 is pretrained on 3-channel RGB natural photographs. This
module feeds it the first three bands (R, G, B) and ignores NIR. Spectral fidelity across all
four bands is already covered by the SAM and index-preservation terms in losses.py, so NIR is
not unsupervised -- it is supervised by the terms actually designed for spectral consistency,
while this term supervises spatial texture. Whether ImageNet features transfer well to
Sentinel-2-scale reflectance imagery is a real open question, not a settled one; treat a
negative result here as informative rather than as a bug.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG16 `features` indices immediately AFTER the named ReLU, i.e. the standard tap points.
#   relu1_2 -> 4, relu2_2 -> 9, relu3_3 -> 16, relu4_3 -> 23
# relu2_2 and relu3_3 are the usual choice for super-resolution: deep enough to encode texture
# and edge structure, shallow enough to stay spatially specific rather than semantic.
VGG16_TAPS = {"relu1_2": 4, "relu2_2": 9, "relu3_3": 16, "relu4_3": 23}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGPerceptualLoss(nn.Module):
    """Frozen-VGG16 feature-space L1 distance between prediction and target.

    The VGG is truncated at the deepest requested tap (nothing past it is ever evaluated), which
    matters on the 4GB dev GPU -- running the full 31-layer stack on 384x384 patches alongside
    the SR model itself is a real memory risk, not a hypothetical one.
    """

    def __init__(self, layers: Sequence[str] = ("relu2_2", "relu3_3"),
                 weights: Optional[Sequence[float]] = None):
        super().__init__()
        unknown = [n for n in layers if n not in VGG16_TAPS]
        if unknown:
            raise ValueError(f"Unknown VGG tap(s) {unknown}; valid: {sorted(VGG16_TAPS)}")
        self.tap_indices = [VGG16_TAPS[n] for n in layers]
        self.layer_names = list(layers)
        self.layer_weights = list(weights) if weights is not None else [1.0] * len(layers)
        if len(self.layer_weights) != len(self.tap_indices):
            raise ValueError("`weights` must have one entry per layer")

        from torchvision.models import VGG16_Weights, vgg16
        full = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(full[:max(self.tap_indices)]))
        # Frozen and eval-mode permanently: this is a fixed measuring instrument, not something
        # being trained. Without requires_grad_(False) its parameters would accumulate gradients
        # every backward pass for no purpose.
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the frozen VGG in eval mode even when the parent module is set to train() --
        otherwise a plain model.train() would silently flip it back."""
        super().train(mode)
        self.features.eval()
        return self

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3]
        # Model output is unconstrained (it is bicubic + residual), so clamp before feeding a
        # network whose pretrained statistics assume displayable [0,1] imagery.
        rgb = rgb.clamp(0.0, 1.0)
        return (rgb - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] < 3:
            raise ValueError(f"perceptual loss needs >=3 bands, got {pred.shape[1]}")
        x, y = self._prepare(pred), self._prepare(target)

        total = pred.new_zeros(())
        for i, layer in enumerate(self.features):
            x = layer(x)
            # target path needs no graph -- it is a fixed reference, so keeping it out of autograd
            # saves real memory on a 4GB card.
            with torch.no_grad():
                y = layer(y)
            depth = i + 1
            if depth in self.tap_indices:
                w = self.layer_weights[self.tap_indices.index(depth)]
                total = total + w * F.l1_loss(x, y)
        return total
