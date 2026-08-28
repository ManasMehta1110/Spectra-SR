"""Convert the SEN2NAIPv2 cross-sensor .taco archive into the plain ROI_xxxx/{lr,hr}.tif layout
that `SEN2NAIPCrossSensorDataset` already reads.

Why convert instead of reading .taco directly at training time: the v2 archive is in the legacy
TACO v1 container, which needs `tacoreader<1.0`, and that pin drags in pandas 3.x / numpy 2.5.x.
Those conflict with the training environment's pinned versions (pandas 2.3.3, numpy 2.4.4) and
with `stackstac`, `datasets` and `s3fs` already installed there. Converting once means the
training environment never imports tacoreader at all, and the existing, tested dataset class is
reused rather than a second loader being written and separately debugged.

Emits the same `metadata.json` shape the v1 loader's quality filter reads, mapping v2's own
quality field onto it: v2 reports `correlation` (2nd-percentile Pearson within 16x16 kernels
between the real Sentinel-2 and a Sentinel-2-like image degraded from the NAIP) where v1
reported QA1/QA2. Higher correlation is better, whereas v1's QA1/QA2 were both
lower-is-better distances, so `correlation` is written under its own key and NOT laundered into
a QA1/QA2 slot where the existing `--qa1-max`/`--qa2-max` filters would silently invert its
meaning.

Usage:
    PYTHONPATH=<dir containing tacoreader<1.0> \\
      python scripts/extract_sen2naipv2.py --taco data/raw/sen2naipv2/sen2naipv2-crosssensor.taco \\
                                           --out data/raw/sen2naipv2/cross-sensor
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--taco", default="data/raw/sen2naipv2/sen2naipv2-crosssensor.taco")
    p.add_argument("--out", default="data/raw/sen2naipv2/cross-sensor")
    p.add_argument("--limit", type=int, default=None,
                   help="Convert only the first N samples (for a quick structural check).")
    p.add_argument("--min-correlation", type=float, default=None,
                   help="Skip samples below this v2 quality correlation. Higher is better; the "
                        "shipped distribution is min 0.866, median 0.904.")
    args = p.parse_args()

    import tacoreader  # imported late: only this script needs it, never the training env

    ds = tacoreader.load(args.taco)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    logger.info(f"{len(ds)} samples in archive; converting {n}")
    os.makedirs(args.out, exist_ok=True)

    written, skipped = 0, 0
    shapes_seen, dtypes_seen = set(), set()
    for i in range(n):
        correlation = float(ds["correlation"][i])
        if args.min_correlation is not None and correlation < args.min_correlation:
            skipped += 1
            continue

        sample = ds.read(i)
        ids = list(sample["tortilla:id"])
        arrays, profiles = {}, {}
        for j, name in enumerate(ids):
            with rasterio.open(sample.read(j)) as src:
                arrays[name] = src.read()
                profiles[name] = src.profile
        if "lr" not in arrays or "hr" not in arrays:
            raise RuntimeError(f"sample {i} has ids {ids}, expected 'lr' and 'hr'")

        shapes_seen.add((arrays["lr"].shape, arrays["hr"].shape))
        dtypes_seen.add((str(arrays["lr"].dtype), str(arrays["hr"].dtype)))

        roi_dir = os.path.join(args.out, f"ROI_{written:05d}")
        os.makedirs(roi_dir, exist_ok=True)
        for name in ("lr", "hr"):
            profile = dict(profiles[name])
            profile.update(driver="GTiff", compress="deflate")
            with rasterio.open(os.path.join(roi_dir, f"{name}.tif"), "w", **profile) as dst:
                dst.write(arrays[name])

        with open(os.path.join(roi_dir, "metadata.json"), "w") as f:
            json.dump({
                "source": "SEN2NAIPv2/cross-sensor",
                "taco_id": str(ds["tortilla:id"][i]),
                # v2's own quality metric. Deliberately NOT written as QA1/QA2: those are
                # lower-is-better distances in v1 and this is higher-is-better, so reusing the
                # key would invert the meaning of the existing --qa1-max/--qa2-max filters.
                "correlation": correlation,
                "days_between": int(ds["days_between"][i]),
                "crs": str(ds["stac:crs"][i]),
                "centroid": str(ds["stac:centroid"][i]),
                "admin0": str(ds["rai:admin0"][i]),
                "admin1": str(ds["rai:admin1"][i]),
            }, f, indent=1)

        written += 1
        if written % 250 == 0:
            logger.info(f"  {written}/{n} written")

    logger.info(f"done: {written} ROIs written, {skipped} skipped, -> {args.out}")
    logger.info(f"shapes seen (lr, hr): {shapes_seen}")
    logger.info(f"dtypes seen (lr, hr): {dtypes_seen}")
    if len(shapes_seen) > 1:
        logger.warning("MULTIPLE tile geometries present -- the dataset class assumes one fixed "
                       "LR_TILE_SIZE/HR_TILE_SIZE, so this needs handling before training.")


if __name__ == "__main__":
    main()
