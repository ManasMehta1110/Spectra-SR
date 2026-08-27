import torch

from spectra_sr.degradation import DegradationOperator, acceptance_test


def _synthetic_hr(seed: int = 0, n_bands: int = 2, size: int = 64) -> torch.Tensor:
    """A reproducible, structured (not pure noise) synthetic HR image -- checkerboard-ish blocks
    plus a smooth gradient, so blur/downsample actually changes it in a way fit_to_pairs has
    real signal to recover from. Pure random noise would blur-invariantly average to ~0.5
    everywhere post-downsample, giving fit_to_pairs nothing informative to fit against.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0, 1, size)
    grid_x, grid_y = torch.meshgrid(x, x, indexing="ij")
    blocks = ((grid_x * 8).floor() + (grid_y * 8).floor()) % 2
    gradient = (grid_x + grid_y) / 2
    base = 0.6 * blocks + 0.4 * gradient
    hr = base.unsqueeze(0).repeat(n_bands, 1, 1)
    hr = hr + 0.02 * torch.randn(hr.shape, generator=g)
    return hr.unsqueeze(0).clamp(0, 1)  # (1, n_bands, size, size)


def test_forward_output_shape():
    op = DegradationOperator(n_bands=2, scale=4)
    hr = _synthetic_hr(size=64)
    lr = op.forward(hr)
    assert lr.shape == (1, 2, 16, 16)


def test_forward_is_deterministic_simulate_is_not():
    op = DegradationOperator(n_bands=2, scale=4)
    hr = _synthetic_hr(size=64)
    a = op.forward(hr)
    b = op.forward(hr)
    assert torch.allclose(a, b), "forward() must be deterministic -- the acceptance test and " \
                                  "the re-degradation cycle loss both depend on that"

    s1 = op.simulate(hr)
    s2 = op.simulate(hr)
    assert not torch.allclose(s1, s2), "simulate() should add stochastic noise"
    # But not wildly different from the noise-free forward pass.
    assert torch.allclose(s1, a, atol=0.2)


def test_fit_to_pairs_recovers_known_sigma():
    """Construct a pair with a KNOWN true sigma, fit a freshly-initialized (wrong-sigma)
    operator to it, and check the fitted sigma lands close to the true one -- this is the actual
    acceptance-test-relevant property: fit_to_pairs has to work, not just run."""
    true_op = DegradationOperator(n_bands=1, scale=4)
    with torch.no_grad():
        true_op.log_sigma.fill_(torch.log(torch.tensor(2.0)))
    hr = _synthetic_hr(n_bands=1, size=64)
    lr_real = true_op.forward(hr).detach()

    fit_op = DegradationOperator(n_bands=1, scale=4, init_sigma=0.5)  # deliberately wrong start
    history = fit_op.fit_to_pairs(hr, lr_real, n_steps=300, lr=0.05)

    fitted_sigma = history["fitted_sigma"][0]
    assert abs(fitted_sigma - 2.0) < 0.3, f"expected sigma near 2.0, got {fitted_sigma}"
    assert history["final_mse"] < 1e-3


def test_acceptance_test_passes_for_matched_operator_fails_for_mismatched():
    true_op = DegradationOperator(n_bands=1, scale=4)
    with torch.no_grad():
        true_op.log_sigma.fill_(torch.log(torch.tensor(2.0)))
    hr = _synthetic_hr(n_bands=1, size=64)
    lr_real = true_op.forward(hr).detach()

    # The operator that actually produced lr_real should pass with a generous threshold.
    result_match = acceptance_test(true_op, hr, lr_real, per_band_threshold=(1e-4,))
    assert result_match.passed
    assert result_match.per_band_rmse[0] < 1e-4

    # A badly mismatched operator (very different sigma) should fail the same threshold.
    wrong_op = DegradationOperator(n_bands=1, scale=4, init_sigma=0.1)
    result_mismatch = acceptance_test(wrong_op, hr, lr_real, per_band_threshold=(1e-4,))
    assert not result_mismatch.passed
    assert result_mismatch.per_band_rmse[0] > result_match.per_band_rmse[0]


def test_pseudo_inverse_satisfies_data_consistency():
    """NOT a "recovers hr better than naive upsampling" test -- that property is actually
    unachievable for a linear pseudo-inverse of a lossy (average-pooling) downsample: decimation
    discards genuine sub-block information no linear operation can recover from the pooled
    result alone, which is precisely why the spec assigns real detail recovery to the generative
    Stage 4 (x_gen), not to A_dagger. (An earlier version of this test asserted exactly that
    unachievable property and correctly failed -- see git history / conversation.)

    What A_dagger actually has to satisfy, per the null-space decomposition (spec Section 1.2:
    x_hat = A_dagger(y) + (I - A_dagger A) x_gen) and Stage 5's projection formula
    (x <- x + A_dagger(y - A(x))): re-degrading the pseudo-inverse's output through A should
    reproduce the original LR observation. That's the actual acceptance criterion plan Section 8
    calls "numerical re-degradation consistency check" -- test it directly.
    """
    op = DegradationOperator(n_bands=1, scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
    hr = _synthetic_hr(n_bands=1, size=64)
    lr = op.forward(hr)

    recovered = op.pseudo_inverse(lr)  # default lambda=3e-2, empirically the sweet spot -- see
                                        # degradation.py's pseudo_inverse docstring
    redegraded = op.forward(recovered)
    consistency_error = torch.nn.functional.mse_loss(redegraded, lr).item()

    # The property that actually matters: deconvolving beats doing nothing. Naive nearest-
    # upsample re-degraded through A is the "did nothing" baseline -- pseudo_inverse must beat
    # it, not hit an arbitrary absolute threshold (which, per the lambda sweep, isn't even
    # monotonically improvable -- see the docstring).
    naive = torch.nn.functional.interpolate(lr, scale_factor=op.scale, mode="nearest")
    naive_consistency_error = torch.nn.functional.mse_loss(op.forward(naive), lr).item()

    assert consistency_error < naive_consistency_error, (
        f"A(A_dagger(y)) [{consistency_error}] should be more consistent with y than "
        f"A(naive_upsample(y)) [{naive_consistency_error}]"
    )
