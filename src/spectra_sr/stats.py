"""Statistical comparison helpers for the ablation study.

Reused verbatim from optical_guided_sr.stats (C:/Projects/optical-guided-super-resolution),
per plan Section 1 -- this is exactly the seed-matched paired t-test / Wilcoxon harness
SPECTRA-SR's own ablation grid (spec Section 5.3, plan Section 5 phase 7) needs, already
validated on a real satellite-SR ablation study. Do not fork this without a reason.

The ablation harness runs every variant across the same seed indices (seed 0..n_seeds-1 for
`full`, `no_temporal_fusion`, etc.), so seed 0's initialization/data ordering is identical across
variants. That makes a *paired* comparison (each seed against itself across two variants) valid
and substantially more powerful than an unpaired comparison at small n -- it cancels out the
seed-to-seed variance component instead of adding it to the noise floor of the comparison.

At low n_seeds these tests have essentially no power -- they'll report high p-values almost
regardless of true effect size. They exist so the methodology is in place; they become
meaningful once n_seeds>=5 on real data.
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def paired_variant_test(ablation_df: pd.DataFrame, variant_a: str, variant_b: str,
                         metric: str = "test_psnr") -> dict:
    """Paired t-test and Wilcoxon signed-rank test between two variants, matched by seed.

    Returns a dict with both tests' statistics/p-values, the mean paired difference, and a 95%
    CI on that difference (t-distribution based). Raises if the two variants don't share the
    same set of seeds -- pairing is only valid when they do.
    """
    a = ablation_df[ablation_df["variant"] == variant_a].set_index("seed")[metric]
    b = ablation_df[ablation_df["variant"] == variant_b].set_index("seed")[metric]
    common_seeds = sorted(set(a.index) & set(b.index))
    if len(common_seeds) < 2:
        raise ValueError(
            f"Need >=2 shared seeds to pair '{variant_a}' vs '{variant_b}' "
            f"(found {len(common_seeds)}). Raise exp_cfg.n_seeds."
        )
    a, b = a.loc[common_seeds].to_numpy(), b.loc[common_seeds].to_numpy()
    diffs = a - b
    n = len(diffs)

    t_stat, t_p = scipy_stats.ttest_rel(a, b)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon is undefined when all paired differences are zero.
        w_stat, w_p = np.nan, np.nan

    mean_diff = float(diffs.mean())
    sem = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
    ci_low, ci_high = mean_diff - t_crit * sem, mean_diff + t_crit * sem

    return {
        "variant_a": variant_a, "variant_b": variant_b, "metric": metric, "n_paired_seeds": n,
        "mean_diff": mean_diff, "ci95_low": ci_low, "ci95_high": ci_high,
        "paired_t_stat": float(t_stat), "paired_t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else np.nan,
    }


_RESULT_COLUMNS = [
    "variant_a", "variant_b", "metric", "n_paired_seeds", "mean_diff", "ci95_low", "ci95_high",
    "paired_t_stat", "paired_t_pvalue", "wilcoxon_stat", "wilcoxon_pvalue",
]


def compare_all_to_reference(ablation_df: pd.DataFrame, reference: str = "full",
                              metric: str = "test_psnr",
                              variants: Optional[list] = None) -> pd.DataFrame:
    """Paired-test every other variant against `reference` (default: the full model).

    Every row has the same columns (`_RESULT_COLUMNS`) whether or not that particular comparison
    succeeded -- a failed comparison (too few shared seeds) gets NaN in the numeric columns plus
    an `error` message, rather than simply omitting keys, which would otherwise let a downstream
    DataFrame silently drop columns pandas has no rows for.
    """
    if variants is None:
        variants = [v for v in ablation_df["variant"].unique() if v != reference]
    rows = []
    for v in variants:
        try:
            rows.append(paired_variant_test(ablation_df, reference, v, metric=metric))
        except ValueError as e:
            row = {col: np.nan for col in _RESULT_COLUMNS}
            row.update({"variant_a": reference, "variant_b": v, "metric": metric, "error": str(e)})
            rows.append(row)
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS + ["error"])
