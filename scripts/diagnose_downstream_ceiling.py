"""One-off diagnostic: how much headroom does the downstream NDVI-agreement check actually have?

Real pretrain-tier training (pretrain_run1/run2) has consistently shown SR improvement over
bicubic pinned near zero, both before and after adding more data. Before concluding the model or
loss function is the bottleneck, check whether the metric itself has room to show a gain at all:
substitute the TRUE HR image as a stand-in "perfect" SR prediction (its classification trivially
agrees with itself, sr_agreement=1.0) and see how much headroom that leaves over bicubic's real
agreement, on real held-out val files, under the same fixed degradation (sigma=1.0)
train_pretrain.py validates against. If even a perfect reconstruction barely beats bicubic here,
the ceiling itself is low on this synthetic setup -- not a model-quality problem.
"""
import os

import rasterio
import torch

from spectra_sr.degradation import DegradationOperator
from spectra_sr.metrics import downstream_classification_agreement

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAIP_DIR = os.path.join(REPO_ROOT, "data/raw/naip")

# Same held-out val split train_pretrain.py's _split_train_val_files produces for
# colab_realistic pretrain_run2/run3 (seed=0, val_fraction=0.2) -- reused verbatim here so this
# diagnostic checks the exact same patches/degradation those runs validated against.
VAL_FILES = [
    "az_m_3411228_ne_12_030_20230625_20240119.tif",
    "ca_m_3611637_nw_11_060_20220626.tif",
    "ca_m_3612118_sw_10_060_20220519.tif",
    "ca_m_3812129_nw_10_060_20220621.tif",
    "co_m_3910525_sw_13_030_20230926_20240104.tif",
    "fl_m_3008329_sw_17_060_20220117.tif",
    "ks_m_3909904_ne_14_030_20230818_20240209.tif",
    "nv_m_3611629_sw_11_060_20220611.tif",
    "or_m_4412247_sw_10_030_20220714.tif",
    "wa_m_4712229_ne_10_060_20231007_20240209.tif",
]

HR_PATCH = 384


def main():
    op = DegradationOperator(n_bands=4, scale=4)
    with torch.no_grad():
        op.log_sigma.fill_(torch.log(torch.tensor(1.0)))  # same fixed val sigma as train_pretrain.py

    ceiling_improvements = []
    baseline_agreements = []
    for fname in VAL_FILES:
        path = os.path.join(NAIP_DIR, fname)
        if not os.path.exists(path):
            print(f"  (missing: {fname})")
            continue
        with rasterio.open(path) as src:
            if src.width < HR_PATCH or src.height < HR_PATCH or src.count < 4:
                print(f"  (skipped, too small/few bands: {fname})")
                continue
            hr_raw = src.read(window=rasterio.windows.Window(0, 0, HR_PATCH, HR_PATCH))

        hr = torch.from_numpy(hr_raw.astype("float32")).unsqueeze(0) / 255.0
        hr = hr[:, :4]

        with torch.no_grad():
            lr = op.simulate(hr)

        # Ceiling: sr_pred = hr itself (a "perfect" model) -- measures true headroom over
        # bicubic on this exact patch/degradation, independent of any actual model's quality.
        result = downstream_classification_agreement(sr_pred=hr, lr_input=lr, hr_target=hr)
        ceiling_improvements.append(result.improvement)
        baseline_agreements.append(result.baseline_agreement)
        print(f"{fname}: baseline_agreement={result.baseline_agreement:.4f} "
              f"ceiling_improvement={result.improvement:+.4f}")

    if ceiling_improvements:
        mean_improvement = sum(ceiling_improvements) / len(ceiling_improvements)
        mean_baseline = sum(baseline_agreements) / len(baseline_agreements)
        print(f"\nMean bicubic baseline agreement: {mean_baseline:.4f}")
        print(f"Mean ceiling improvement (perfect SR vs bicubic): {mean_improvement:+.4f}")
        print("Real trained models so far (run1/run2) have shown ~-0.0007 to +0.0001 improvement "
              "-- compare that against this ceiling.")


if __name__ == "__main__":
    main()
