import pandas as pd

from spectra_sr.stats import compare_all_to_reference, paired_variant_test


def _fake_ablation_df():
    rows = []
    for seed in range(5):
        rows.append({"variant": "full", "seed": seed, "test_psnr": 30.0 + seed * 0.1})
        rows.append({"variant": "no_temporal_fusion", "seed": seed, "test_psnr": 29.0 + seed * 0.1})
    return pd.DataFrame(rows)


def test_paired_variant_test_runs_and_pairs_by_seed():
    df = _fake_ablation_df()
    result = paired_variant_test(df, "full", "no_temporal_fusion")
    assert result["n_paired_seeds"] == 5
    assert result["mean_diff"] > 0


def test_compare_all_to_reference_covers_every_variant():
    df = _fake_ablation_df()
    out = compare_all_to_reference(df, reference="full")
    assert set(out["variant_b"]) == {"no_temporal_fusion"}
