"""Stage 0 -- the degradation operator A. Plan Section 5, phase 1: the first thing built, and a
hard gate everything downstream depends on.

A maps a hypothetical high-resolution scene to what Sentinel-2 would have observed: a per-band
learnable blur, anti-aliased downsampling, and (in `simulate`, not `forward`) signal-dependent
noise -- fitted against real co-located HR/Sentinel-2 pairs via `fit_to_pairs`, not assumed as
bicubic. Per the team's own prior finding (optical_guided_sr README): naive resize-only
degradation inflated bicubic's own reconstruction PSNR from 32.9 dB to 63.6 dB on identical
scenes -- getting this operator wrong silently invalidates every downstream number.

`forward` (blur + downsample only, deterministic) is what the acceptance test and Stage 3's
re-degradation cycle loss use -- it has to be a fixed, comparable mapping. `simulate` adds
signal-dependent noise on top and is for generating synthetic training pairs, where stochasticity
is the point.

IMPORTANT: the acceptance-test threshold (per-band NEDeltaRho / noise-equivalent reflectance
difference) is a real, published Sentinel-2 sensor characteristic that has NOT been sourced from
ESA's Sentinel-2 User Handbook yet -- do not treat the placeholder default below as a verified
number. Replace it before running the acceptance test against real data (plan Section 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable 1D Gaussian kernel, differentiable w.r.t. sigma (a scalar tensor).

    radius is fixed (not derived from sigma) so the kernel size stays constant during
    optimization -- letting kernel size vary with a learnable sigma would make the operation
    non-differentiable at the size-change boundaries.
    """
    x = torch.arange(-radius, radius + 1, dtype=sigma.dtype, device=sigma.device)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2 + 1e-8))
    return kernel / kernel.sum()


class DegradationOperator(nn.Module):
    """Differentiable A: HR (B, C, H, W) -> LR (B, C, H/scale, W/scale).

    Per-band separable Gaussian blur (learnable sigma per band) -> anti-aliased downsample
    (strided average pool, scale x scale). Noise is handled separately in `simulate`.
    """

    def __init__(self, n_bands: int, scale: int, kernel_radius: int = 7,
                 init_sigma: float = 1.5):
        super().__init__()
        self.n_bands = n_bands
        self.scale = scale
        self.kernel_radius = kernel_radius
        # One learnable sigma per band -- different bands can have genuinely different MTF/PSF
        # (spec Stage 0: "The PSF is anisotropic and differs across bands -- not a single
        # Gaussian" is the eventual target; per-band scalar sigma is the tractable first step,
        # not the final word).
        self.log_sigma = nn.Parameter(torch.full((n_bands,), torch.log(torch.tensor(init_sigma))))

    def _blur_kernels(self) -> torch.Tensor:
        """(n_bands, 1, k, k) separable-Gaussian-outer-product kernels, one per band."""
        sigmas = torch.exp(self.log_sigma)
        kernels = []
        for b in range(self.n_bands):
            k1d = _gaussian_kernel_1d(sigmas[b], self.kernel_radius)
            k2d = torch.outer(k1d, k1d)
            kernels.append(k2d)
        return torch.stack(kernels).unsqueeze(1)  # (n_bands, 1, k, k)

    def blur(self, hr: torch.Tensor) -> torch.Tensor:
        """Depthwise (per-band) Gaussian blur, reflect-padded so output size == input size."""
        assert hr.shape[1] == self.n_bands, f"expected {self.n_bands} bands, got {hr.shape[1]}"
        kernels = self._blur_kernels().to(hr.dtype)
        padded = F.pad(hr, [self.kernel_radius] * 4, mode="reflect")
        return F.conv2d(padded, kernels, groups=self.n_bands)

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        """Deterministic A: blur then anti-aliased downsample. Used by the acceptance test and
        Stage 3's re-degradation cycle loss -- must stay noise-free so it's a fixed, comparable
        mapping, not a stochastic one."""
        blurred = self.blur(hr)
        return F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)

    def simulate(self, hr: torch.Tensor, noise_scale: float = 0.01,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """A plus signal-dependent (approximately Poisson, via the standard Gaussian
        approximation sqrt(signal)*noise) noise -- for generating synthetic training pairs, not
        for the acceptance test or the cycle-consistency loss, both of which need forward()'s
        determinism to be comparable against a fixed real observation."""
        clean = self.forward(hr)
        signal = clean.clamp(min=0)
        noise = torch.randn(clean.shape, generator=generator, device=clean.device,
                             dtype=clean.dtype) * noise_scale * torch.sqrt(signal + 1e-8)
        return clean + noise

    def _effective_kernel(self, band: int) -> torch.Tensor:
        """The blur kernel alone is NOT the effective HR-grid degradation kernel -- forward()
        also average-pools (a box filter) as its downsampling step, and nearest-upsampling the
        LR result back to the HR grid does not undo that box-filter blur. Model the effective
        kernel as blur convolved with a `scale`x`scale` box kernel, both normalized, via a
        'full' 2D convolution -- this is what pseudo_inverse must deconvolve against, not the
        blur kernel alone. (Caught by test_pseudo_inverse_reduces_error_versus_naive_upsample:
        without this, the deconvolution under-corrects and performs worse than doing nothing.)
        """
        blur_kernel = self._blur_kernels()[band, 0]  # (k, k)
        box = torch.full((self.scale, self.scale), 1.0 / (self.scale ** 2),
                          dtype=blur_kernel.dtype, device=blur_kernel.device)
        # 'full' convolution via conv2d with the box kernel flipped (box is symmetric, so
        # flipping is a no-op here, but stated for correctness) and full padding.
        combined = F.conv2d(
            blur_kernel.view(1, 1, *blur_kernel.shape),
            box.flip(0, 1).view(1, 1, *box.shape),
            padding=(box.shape[0] - 1, box.shape[1] - 1),
        )[0, 0]
        return combined / combined.sum()

    def pseudo_inverse(self, lr: torch.Tensor, tikhonov_lambda: float = 3e-2) -> torch.Tensor:
        """Regularized approximate A_dagger: nearest-upsample back to HR grid, then a per-band
        Tikhonov-regularized Wiener deconvolution against the *effective* kernel (blur convolved
        with the downsampling box filter, see `_effective_kernel` -- deconvolving against the
        blur kernel alone ignores the box-filter contribution from average-pool downsampling and
        under-corrects). A naive transpose (upsample + correlate, no regularization) produces
        ringing at edges indistinguishable from recovered detail -- exactly the failure mode
        Stage 5's spec calls out.

        `tikhonov_lambda`'s effect on data-consistency is NOT monotonic here -- because
        `_effective_kernel` is itself an approximation (a single smooth convolution standing in
        for blur -> average-pool -> nearest-upsample, which is actually blocky, not smooth),
        very small lambda tries to exactly invert the wrong model and blows up at high
        frequencies, while very large lambda over-suppresses and barely changes the naive
        upsample. Empirically swept (see test_degradation.py): consistency error bottoms out
        around lambda=3e-2 (the default), degrading in both directions from there. Re-sweep this
        if the kernel radius, scale, or blur sigma range changes materially.
        """
        up = F.interpolate(lr, scale_factor=self.scale, mode="nearest")
        h, w = up.shape[-2:]

        out_bands = []
        for b in range(self.n_bands):
            kernel = self._effective_kernel(b)
            kh, kw = kernel.shape
            k_pad = torch.zeros(h, w, dtype=up.dtype, device=up.device)
            k_pad[:kh, :kw] = kernel
            # Center the kernel via its own peak (robust to odd- or even-sized combined
            # kernels, unlike assuming the center is exactly kh//2) so the FFT phase corresponds
            # to a zero-centered PSF -- otherwise the deconvolved image comes back shifted.
            peak = torch.argmax(kernel).item()
            peak_row, peak_col = peak // kw, peak % kw
            k_pad = torch.roll(k_pad, shifts=(-peak_row, -peak_col), dims=(0, 1))

            K = torch.fft.rfft2(k_pad)
            Y = torch.fft.rfft2(up[:, b])
            wiener = torch.conj(K) / (K.abs() ** 2 + tikhonov_lambda)
            X = Y * wiener
            x = torch.fft.irfft2(X, s=(h, w))
            out_bands.append(x)

        return torch.stack(out_bands, dim=1)

    def fit_to_pairs(self, hr_real: torch.Tensor, lr_real: torch.Tensor, n_steps: int = 300,
                      lr: float = 0.05) -> Dict[str, float]:
        """Optimize log_sigma to minimize ||forward(hr_real) - lr_real||. Returns a small history
        dict (final loss, per-band fitted sigma) rather than nothing, so a caller can log/assert
        on it instead of re-deriving the fitted state by hand.
        """
        optimizer = torch.optim.Adam([self.log_sigma], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            pred = self.forward(hr_real)
            loss = F.mse_loss(pred, lr_real)
            loss.backward()
            optimizer.step()
        return {
            "final_mse": float(loss.item()),
            "fitted_sigma": torch.exp(self.log_sigma).detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    per_band_rmse: Tuple[float, ...]
    per_band_threshold: Tuple[float, ...]


# PLACEHOLDER -- STILL NOT SOURCED. Do not treat as a real, published NEDeltaRho value, do not
# quote it in a report, and do not present a guardrail pass/fail based on it as evidence of
# anything. A single scalar reused across all four bands is additionally wrong in shape: the real
# quantity differs per band, because SNR and reference radiance both do.
#
# An attempt to source it (2026-08-28) failed: sentinel.esa.int did not resolve, the Copernicus
# SentiWiki Data Quality Report PDF returned 404, and the ResearchGate figure carrying the table
# was not machine-readable. Rather than substitute half-remembered numbers -- which would be worse
# than this placeholder, since a wrong value labelled "ESA specification" looks authoritative --
# the gap is left open and documented.
#
# TO CLOSE THIS, obtain any one of:
#   * Sentinel-2 Products Specification Document, ref S2-PDGS-CS-DI-PSD (latest is v15.0, 2024)
#   * Sentinel-2 MSI L1C Data Quality Report (monthly, OMPC.CS.DQR.001.*) -- carries measured
#     SNR@Lref per band for both S2A and S2B, which is better than the requirement value
#   * ESA Sentinel-2 MSI Technical Guide, "Mission Performance" page
#
# and extract, for B02/B03/B04/B08: the reference radiance Lref (W/m^2/sr/um) and the SNR at that
# radiance. Then convert to a noise-equivalent reflectance difference per band:
#
#     NEDeltaRho_b  =  rho_ref_b / SNR@Lref_b
#
# where rho_ref_b is the top-of-atmosphere reflectance corresponding to Lref_b. Replace this
# constant with that per-band tuple and re-run the guardrails; until then a guardrail FAIL means
# only "differs from an arbitrary 0.005", not "exceeds sensor noise".
PLACEHOLDER_NEDRHO = 0.005


def acceptance_test(operator: DegradationOperator, hr_real: torch.Tensor, lr_real: torch.Tensor,
                     per_band_threshold: Optional[Tuple[float, ...]] = None) -> AcceptanceResult:
    """Plan Section 5/8's hard gate: A(hr_real) must match lr_real within each band's sensor
    noise level. Do not proceed to Stage 1/3 if this fails -- fix the operator, don't route
    around it.
    """
    if per_band_threshold is None:
        per_band_threshold = tuple([PLACEHOLDER_NEDRHO] * operator.n_bands)

    with torch.no_grad():
        pred = operator.forward(hr_real)
        per_band_rmse = tuple(
            float(torch.sqrt(F.mse_loss(pred[:, b], lr_real[:, b])))
            for b in range(operator.n_bands)
        )
    passed = all(rmse <= thresh for rmse, thresh in zip(per_band_rmse, per_band_threshold))
    return AcceptanceResult(passed=passed, per_band_rmse=per_band_rmse,
                             per_band_threshold=per_band_threshold)
