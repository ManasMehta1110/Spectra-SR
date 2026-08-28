import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.inference import super_resolve
from spectra_sr.model import SMOKE_TEST, SpectraHATCore
from spectra_sr.uncertainty import UncertaintyHead


def _rig():
    torch.manual_seed(0)
    model = SpectraHATCore(SMOKE_TEST).eval()
    head = UncertaintyHead(n_bands=SMOKE_TEST.n_bands).eval()
    op = DegradationOperator(n_bands=SMOKE_TEST.n_bands, scale=SMOKE_TEST.scale)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))
    lr = torch.rand(1, SMOKE_TEST.n_bands, SMOKE_TEST.train_patch_size, SMOKE_TEST.train_patch_size)
    return model, head, op, lr


def test_returns_all_four_stages():
    """The PS asks for a product carrying its own uncertainty and error accounting, so the
    pipeline must deliver image + uncertainty + guardrail verdict together rather than leaving
    the caller to assemble them."""
    model, head, op, lr = _rig()
    r = super_resolve(lr, model, head, op)
    s = SMOKE_TEST.scale
    assert r.image.shape == (1, SMOKE_TEST.n_bands,
                             SMOKE_TEST.train_patch_size * s, SMOKE_TEST.train_patch_size * s)
    assert r.uncertainty_std.shape == r.image.shape
    assert r.guardrails is not None
    assert isinstance(r.consistency_error, float)


def test_uncertainty_is_positive_and_finite():
    """It is a standard deviation. A negative or non-finite value would be silently meaningless
    downstream rather than raising."""
    model, head, op, lr = _rig()
    r = super_resolve(lr, model, head, op)
    assert torch.isfinite(r.uncertainty_std).all()
    assert (r.uncertainty_std > 0).all()


def test_projection_reduces_consistency_error():
    """Stage 5's entire purpose. If enabling it does not reduce ||A(x) - y||, it is not doing its
    job and the consistency claim built on it is unfounded."""
    model, head, op, lr = _rig()
    without = super_resolve(lr, model, head, op, apply_projection=False)
    with_proj = super_resolve(lr, model, head, op, apply_projection=True)
    assert with_proj.consistency_error < without.consistency_error
    assert with_proj.projected and not without.projected


def test_stronger_projection_step_gives_stronger_consistency():
    """Guards the step parameter against being silently ignored -- a wired-but-inert knob would
    pass the test above while making the documented accuracy/consistency trade meaningless."""
    model, head, op, lr = _rig()
    half = super_resolve(lr, model, head, op, projection_step=0.5)
    full = super_resolve(lr, model, head, op, projection_step=1.0)
    assert full.consistency_error < half.consistency_error


def test_uncertainty_describes_the_delivered_pixels():
    """Uncertainty must be computed from the FINAL image, after projection. If it were computed
    from the pre-projection prediction it would describe pixels the caller never receives."""
    model, head, op, lr = _rig()
    a = super_resolve(lr, model, head, op, apply_projection=True)
    b = super_resolve(lr, model, head, op, apply_projection=False)
    assert not torch.allclose(a.uncertainty_std, b.uncertainty_std), (
        "uncertainty is identical with and without projection -- it is being computed from the "
        "unprojected prediction rather than the delivered image"
    )


def test_recalibration_factor_scales_uncertainty():
    """The measured head was over-confident by 1.25x, so shipping without applying the fitted
    factor would understate risk. Verify the knob actually reaches the delivered uncertainty."""
    model, head, op, lr = _rig()
    base = super_resolve(lr, model, head, op, uncertainty_recalibration=1.0)
    scaled = super_resolve(lr, model, head, op, uncertainty_recalibration=2.0)
    assert torch.allclose(scaled.uncertainty_std, base.uncertainty_std * 2.0)
    assert torch.allclose(scaled.image, base.image), "recalibration must not alter the image"


def test_run_checks_false_skips_guardrails():
    model, head, op, lr = _rig()
    r = super_resolve(lr, model, head, op, run_checks=False)
    assert r.guardrails is None


def test_is_deterministic():
    model, head, op, lr = _rig()
    a = super_resolve(lr, model, head, op)
    b = super_resolve(lr, model, head, op)
    assert torch.allclose(a.image, b.image)
    assert torch.allclose(a.uncertainty_std, b.uncertainty_std)


def test_loads_checkpoints_written_before_edge_features_existed():
    """Backward compatibility, verified rather than assumed. Adding edge features widened the
    uncertainty head's first conv by two channels, which makes every earlier checkpoint fail to
    load with a bare shape mismatch. The loader must infer the variant from the stored weights."""
    import tempfile, os
    from spectra_sr.inference import load_for_inference
    from spectra_sr.model import SMOKE_TEST
    from spectra_sr.uncertainty import UncertaintyHead

    old_head = UncertaintyHead(n_bands=SMOKE_TEST.n_bands, use_edge_features=False)
    model = SpectraHATCore(SMOKE_TEST)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.pt")
        torch.save({"model": model.state_dict(), "uncertainty_head": old_head.state_dict(),
                    "config": "smoke_test", "epoch": 0, "val_total_loss": 0.1}, path)
        m, h, deg, cfg = load_for_inference(path, {"smoke_test": SMOKE_TEST},
                                             torch.device("cpu"))
        assert h.use_edge_features is False
        lr = torch.rand(1, cfg.n_bands, cfg.train_patch_size, cfg.train_patch_size)
        r = super_resolve(lr, m, h, deg)   # must actually run, not merely construct
        assert torch.isfinite(r.uncertainty_std).all()
