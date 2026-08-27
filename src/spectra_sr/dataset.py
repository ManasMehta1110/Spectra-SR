"""PyTorch Dataset over preprocessed Sentinel-2/HR-reference pairs. Plan Section 5, phase 3.

Pretrain tier only, for now (plan Section 4.1): real HR imagery (NAIP) -> Stage 0's
DegradationOperator.simulate() synthesizes the LR side, with real signal-dependent noise, not
just the deterministic forward() used for the acceptance test/cycle loss. Fine-tune/validate
tier (real Sentinel-2 paired against real Indian HR reference) needs Stage 1's
coregister/crop_reference_to_footprint pipeline wired in once that data exists -- not built yet,
since there's no real Indian HR reference to pair against (PlanetScope application pending).

`sigma_range` implements degradation domain-randomization: the pretrain-tier degradation operator
uses a placeholder sigma (not yet fit to real Sentinel-2 data -- that only happens once real
Indian HR reference pairs exist, per fit_to_pairs()). Training against one single guessed sigma
risks the model learning habits tuned to that specific (possibly wrong) blur amount, which
fine-tuning would then have to unlearn rather than build on -- a real risk given the team's own
prior finding that a wrong degradation assumption doesn't just hurt a little, it can invalidate
the whole comparison (optical_guided_sr's 63.6->32.9 dB finding). Randomizing sigma per sample
across a plausible range hedges against that, instead of betting pretraining on one guess.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from .degradation import DegradationOperator
from .utils import logger


class NAIPPretrainDataset(torch.utils.data.Dataset):
    """Random HR crops from real NAIP GeoTIFFs, paired with a synthetic LR observation from
    Stage 0's DegradationOperator. `__len__` is `crops_per_file * len(files)` -- crops are drawn
    fresh (random offset) each `__getitem__` call, not pre-extracted, so repeated epochs over a
    small file set still see varied crops rather than memorizing a fixed patch grid.
    """

    def __init__(self, naip_dir: str, degradation_operator: DegradationOperator,
                 hr_patch_size: int = 384, crops_per_file: int = 50,
                 file_list: Optional[List[str]] = None, seed: Optional[int] = None,
                 sigma_range: Optional[Tuple[float, float]] = (0.5, 2.5)):
        # Excludes "_full_*.tif" -- acquire_naip.py's temporary full-tile download files (see
        # its download_patch docstring), which can be mid-write here if an acquisition run is
        # still in progress concurrently. Hit this for real: a background NAIP download was
        # still running while this dataset tried to read its in-progress temp file, causing a
        # genuine TIFFReadEncodedTile "got 0 bytes, expected N" GDAL error, not a synthetic one.
        self.files = file_list if file_list is not None else sorted(
            f for f in glob.glob(os.path.join(naip_dir, "*.tif"))
            if not os.path.basename(f).startswith("_full_")
        )
        if not self.files:
            raise ValueError(f"No .tif files found in {naip_dir}")
        self.operator = degradation_operator
        self.hr_patch_size = hr_patch_size
        self.crops_per_file = crops_per_file
        self.sigma_range = sigma_range
        self.rng = np.random.default_rng(seed)

        # Cache each file's (width, height, band count) up front -- avoids reopening every file
        # on every __getitem__ just to know valid crop bounds, and surfaces problems here
        # (dataset construction) rather than deep inside a training loop.
        #
        # Undersized files are SKIPPED with a warning, not a hard failure -- real bug found
        # running this for real: acquire_naip.py's bbox-intersection cropping can legitimately
        # produce a patch smaller than requested when a search result's tile only barely
        # overlaps the search bbox (hit this on 4 of 51 real files, all from one batch near a
        # state-border AOI where the actual overlap was narrow in one dimension). One bad file
        # shouldn't kill an entire multi-hour training run; log it and move on with the rest.
        self._shapes = []
        usable_files = []
        for f in self.files:
            with rasterio.open(f) as src:
                if src.width < hr_patch_size or src.height < hr_patch_size:
                    logger.warning(
                        f"Skipping {f} ({src.width}x{src.height}) -- smaller than "
                        f"hr_patch_size={hr_patch_size} in at least one dimension."
                    )
                    continue
                usable_files.append(f)
                self._shapes.append((src.width, src.height, src.count))
        self.files = usable_files
        if not self.files:
            raise ValueError(
                f"No .tif files in {naip_dir} are large enough for hr_patch_size={hr_patch_size}"
            )

    def __len__(self) -> int:
        return len(self.files) * self.crops_per_file

    def __getitem__(self, idx: int):
        file_idx = idx % len(self.files)
        path = self.files[file_idx]
        width, height, n_bands = self._shapes[file_idx]

        max_x = width - self.hr_patch_size
        max_y = height - self.hr_patch_size
        x0 = int(self.rng.integers(0, max_x + 1))
        y0 = int(self.rng.integers(0, max_y + 1))

        with rasterio.open(path) as src:
            hr_raw = src.read(window=Window(x0, y0, self.hr_patch_size, self.hr_patch_size))

        hr = torch.from_numpy(hr_raw.astype(np.float32)) / 255.0  # NAIP is uint8

        n_op_bands = self.operator.n_bands
        if n_bands != n_op_bands:
            hr = hr[:n_op_bands] if n_bands > n_op_bands else hr.repeat(
                (n_op_bands + n_bands - 1) // n_bands, 1, 1)[:n_op_bands]

        with torch.no_grad():
            if self.sigma_range is not None:
                # Randomize per sample, not fixed once at dataset construction -- each
                # __getitem__ call should see an independently-drawn degradation strength, per
                # this module's domain-randomization rationale. Mutating self.operator.log_sigma
                # in place is safe here because this operator instance is used *only* to
                # generate data (never trained/optimized itself in this context, always called
                # under no_grad) -- see train_pretrain.py's dataset_degradation instance, which
                # is kept separate from the loss-computation operator for exactly this reason.
                lo, hi = self.sigma_range
                random_sigma = self.rng.uniform(lo, hi, size=self.operator.n_bands)
                self.operator.log_sigma.copy_(
                    torch.log(torch.as_tensor(random_sigma, dtype=self.operator.log_sigma.dtype))
                )
            lr = self.operator.simulate(hr.unsqueeze(0)).squeeze(0)

        return lr, hr
