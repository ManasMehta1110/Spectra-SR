"""Sentinel-2 L2A acquisition via Copernicus Data Space Ecosystem (CDSE) -- the confirmed LR
source (plan Section 4). Requires CDSE OAuth2 client credentials in .env:

    CDSE_CLIENT_ID=...
    CDSE_CLIENT_SECRET=...

(Generate these as an OAuth client under CDSE dashboard -> User settings, not your account
password -- scoped and revocable, same reasoning as optical_guided_sr's Earthdata .env pattern.)

Two real bugs found and fixed while building this against live CDSE data, kept documented here
so they don't get silently reintroduced:

1. `SentinelHubRequest`'s evalscript API returns each band as a normalized [0,1] float by
   default. Casting that straight to a UINT16 output truncates every value to 0 or 1 -- the
   scale-by-10000 multiplication in `EVALSCRIPT` below is required, matching the standard
   Sentinel-2 L2A DN convention already assumed by config.py's `reflectance_scale`.
2. `catalog.search(..., limit=N)` (pystac-client / acquire_naip.py's version of this mistake)
   does NOT cap total results -- verified this one doesn't recur here by using
   `results[:limit]` explicitly on the materialized list rather than trusting a search kwarg to
   do it.

Usage:
    python scripts/acquire_sentinel2.py --bbox 75.50 30.50 75.55 30.55 \\
        --start 2026-06-01 --end 2026-08-01 --max-cloud 15 --limit 3 --out data/raw/sentinel2
"""
import argparse
import os

import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox, CRS, DataCollection, MimeType, SentinelHubCatalog, SentinelHubRequest, SHConfig,
    bbox_to_dimensions,
)

BANDS = ("B02", "B03", "B04", "B08")  # native 10m bands, matches config.Config.bands

EVALSCRIPT = f"""
//VERSION=3
function setup() {{
  return {{ input: [{", ".join(f'"{b}"' for b in BANDS)}],
            output: {{ bands: {len(BANDS)}, sampleType: "UINT16" }} }};
}}
function evaluatePixel(sample) {{
  return [{", ".join(f"sample.{b} * 10000" for b in BANDS)}];
}}
"""


def _config() -> SHConfig:
    load_dotenv()
    config = SHConfig()
    config.sh_client_id = os.environ["CDSE_CLIENT_ID"]
    config.sh_client_secret = os.environ["CDSE_CLIENT_SECRET"]
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config


def search(bbox_coords, start, end, max_cloud, limit, config):
    bbox = BBox(bbox_coords, crs=CRS.WGS84)
    catalog = SentinelHubCatalog(config=config)
    results = catalog.search(
        "sentinel-2-l2a", bbox=bbox, time=(start, end),
        filter=f"eo:cloud_cover < {max_cloud}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"],
                "exclude": []},
    )
    items = list(results)  # materialize before slicing -- see module docstring, bug 2
    return bbox, items[:limit]


def download_patch(bbox, date_str, out_path, config, resolution: int = 10):
    """Real reflectance DNs (uint16, scaled x10000) for `bbox` on `date_str`, written as a
    proper georeferenced GeoTIFF -- not the raw sentinelhub array, so it's directly usable by
    spectra_sr.preprocessing and rasterio-based tooling downstream."""
    size = bbox_to_dimensions(bbox, resolution=resolution)
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                "cdse", service_url=config.sh_base_url),
            time_interval=(date_str, date_str),
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, config=config,
    )
    arr = request.get_data()[0]  # (H, W, n_bands), uint16
    arr = np.moveaxis(arr, -1, 0)  # -> (n_bands, H, W) for rasterio

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    transform = from_bounds(*bbox, width=size[0], height=size[1])
    profile = {
        "driver": "GTiff", "dtype": "uint16", "count": len(BANDS),
        "height": size[1], "width": size[0], "crs": "EPSG:4326", "transform": transform,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = BANDS
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--max-cloud", type=float, default=15)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--out", type=str, default="data/raw/sentinel2")
    args = p.parse_args()

    config = _config()
    bbox, items = search(tuple(args.bbox), args.start, args.end, args.max_cloud, args.limit, config)
    print(f"Found {len(items)} Sentinel-2 L2A scenes (after limiting to {args.limit})")

    for item in items:
        date_str = item["properties"]["datetime"][:10]
        out_path = os.path.join(args.out, f"{item['id']}.tif")
        path = download_patch(bbox, date_str, out_path, config)
        print(f"  saved {path} (cloud cover {item['properties']['eo:cloud_cover']}%)")
