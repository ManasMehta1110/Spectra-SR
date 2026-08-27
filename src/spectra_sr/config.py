from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """Global, non-experiment-varying settings -- paths, band layout, and the physical constants
    tied to Sentinel-2 itself. Per-run experimental knobs (which model variant, which ablation,
    which seed) belong in a separate RunConfig once the ablation grid (plan Section 5, phase 7)
    is built, mirroring rag-poison-robustness's RunConfig / runner.run_grid split -- not here.
    """

    # Sentinel-2 L2A native-10m bands used by the Core pipeline (plan Section 1: primary bands).
    # 20m bands (B5/B6/B7/B8A/B11/B12) are Section 10 stretch scope, not Core.
    bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08")

    scale_factor: int = 4          # 10m -> 2.5m, per the PS and SPECTRA-SR spec Stage 3
    lr_patch_size: int = 128       # LR patch side length in pixels (spec Section 1, tiling)
    hr_patch_size: int = field(init=False)

    # Sentinel-2 L2A reflectance is stored as scaled uint16; not HLS's fill/scale convention --
    # keep this separate from the reused optical_guided_sr constants rather than assuming they
    # match. Confirm against the actual product metadata during Stage 1 (preprocessing) build,
    # not assumed here.
    reflectance_scale: float = 1.0 / 10000

    out_dir: str = "data/processed"
    raw_dir: str = "data/raw"

    def __post_init__(self):
        object.__setattr__(self, "hr_patch_size", self.lr_patch_size * self.scale_factor)
