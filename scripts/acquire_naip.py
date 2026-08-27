"""Pretrain-tier data acquisition -- NAIP via Microsoft Planetary Computer's public STAC catalog.

No account/credentials needed, unlike the other three sources in plan Section 4 (Copernicus Data
Space, Bhuvan, PlanetScope all require registration). Per the plan's data-tier strategy (Section
4.1): NAIP is pretraining volume only, geography-mismatched to the actual India-focused
fine-tune/validate tiers -- do not treat anything pulled here as real evaluation data.

KNOWN ENVIRONMENT ISSUE: GDAL's own VSICURL (used by `rasterio.open(remote_url)`) cannot resolve
the NAIP blob storage host in this dev environment ("CURL error: Could not resolve host ...
Could not contact DNS servers"), even though the OS resolver (Python's socket/urllib) resolves it
fine -- confirmed via `socket.getaddrinfo`. This means GDAL-native windowed/range reads of remote
COGs don't work here; if this script is run somewhere GDAL's curl DNS works correctly, switching
`download_patch` to a `rasterio.open(href) + windowed read` (no full download) would be more
efficient. Verify that assumption before "fixing" this to remove the full download, in case the
DNS issue is specific to this machine rather than universal.

Usage:
    python scripts/acquire_naip.py --bbox -119.9 36.7 -119.8 36.8 --limit 5 --out data/raw/naip
"""
import argparse
import os
import urllib.request

import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
import rasterio.windows
from rasterio.windows import Window


def search(bbox, limit, datetime=None):
    """`limit` caps the TOTAL number of items returned (via pystac-client's max_items), not the
    STAC API page size. Passing it as `limit=` to `catalog.search()` alone does NOT cap results
    -- that only sets the per-page fetch size, and `.items()` pages through every match
    regardless; found this the hard way when a `--limit 2` run downloaded all 36 matching items
    instead of 2, before a signed URL token expired partway through (see download_patch's
    resigning logic below, added for the same reason).
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    result = catalog.search(collections=["naip"], bbox=bbox, max_items=limit, datetime=datetime)
    return list(result.items())


def download_patch(item, out_dir, bbox, patch_size: int = 1024, keep_full_tile: bool = False):
    """Download the full remote GeoTIFF via urllib (works; GDAL's VSICURL does not, see module
    docstring), then extract a patch locally via rasterio -- from the actual intersection of the
    search `bbox` with the tile, NOT a blind (0, 0, patch_size, patch_size) top-left corner.

    Real bug this fixes: a NAIP tile matching the search bbox can be much larger than the
    requested area, and its top-left corner can land entirely outside the searched bbox --
    confirmed directly: two patches pulled with bbox (-119.9, 36.7, -119.8, 36.8) landed at
    longitude -119.94 (west of the search area) and with a top edge at 36.814 (north of the 36.8
    search boundary). The STAC search matching a tile only means the tile's *overall* extent
    intersects the bbox somewhere, not that an arbitrary corner crop of it does.
    """
    os.makedirs(out_dir, exist_ok=True)
    href = planetary_computer.sign(item).assets["image"].href  # re-signed here, not once at
    # search time -- signed SAS tokens expire (~1hr), and a batch of ~400MB sequential downloads
    # can easily outlive a token signed upfront. Hit this directly: item 13 of a 36-item batch
    # got a 403 "failed to authenticate" after ~40 min without this.
    full_path = os.path.join(out_dir, f"_full_{item.id}.tif")
    patch_path = os.path.join(out_dir, f"{item.id}.tif")

    urllib.request.urlretrieve(href, full_path)

    with rasterio.open(full_path) as src:
        # bbox is WGS84 (lon/lat); reproject to the tile's own CRS (NAIP is UTM) before
        # intersecting, then clip to what the tile actually covers.
        left, bottom, right, top = rasterio.warp.transform_bounds("EPSG:4326", src.crs, *bbox)
        left = max(left, src.bounds.left)
        bottom = max(bottom, src.bounds.bottom)
        right = min(right, src.bounds.right)
        top = min(top, src.bounds.top)
        if left >= right or bottom >= top:
            raise ValueError(
                f"{item.id}: requested bbox does not actually overlap this tile's real extent "
                f"({src.bounds}) enough to crop a patch -- the STAC search matched it, but the "
                f"true geometry doesn't intersect the requested area."
            )

        window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_lengths().round_offsets()
        window = Window(window.col_off, window.row_off,
                         min(window.width, patch_size, src.width - window.col_off),
                         min(window.height, patch_size, src.height - window.row_off))

        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(height=window.height, width=window.width,
                        transform=src.window_transform(window))
        with rasterio.open(patch_path, "w", **profile) as dst:
            dst.write(data)

    if not keep_full_tile:
        os.remove(full_path)
    return patch_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--out", type=str, default="data/raw/naip")
    p.add_argument("--patch-size", type=int, default=1024)
    p.add_argument("--keep-full-tile", action="store_true")
    args = p.parse_args()

    items = search(args.bbox, args.limit)
    print(f"Found {len(items)} NAIP items")
    for item in items:
        path = download_patch(item, args.out, tuple(args.bbox), args.patch_size, args.keep_full_tile)
        print(f"  saved {path}")
