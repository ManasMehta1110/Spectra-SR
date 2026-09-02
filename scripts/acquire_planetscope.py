"""PlanetScope acquisition via Planet's Data API -- the HR reference source for the India
fine-tune tier, once paired against real Sentinel-2 for the same AOI/date (see
acquire_sentinel2.py, and preprocessing.coregister for the pairing step neither script does
yet). Requires a Planet API key in .env:

    PL_API_KEY=pl_...

Account must have Education & Research (or another) subscription attached -- confirmed working
for this project's account (2-year E&R Basic, 3,000 sq km/month, verified via
`/auth/v1/experimental/public/my/subscriptions`), not merely a valid-looking key. Run `--check`
before spending quota on a real pull.

Real behavior confirmed against the live API while building this, kept documented so it isn't
silently reintroduced:

1. **A 30-day embargo on PlanetScope applies to E&R accounts.** An item search returns real
   results regardless of age, but `item['_permissions']` is an empty list for anything inside
   the embargo window -- confirmed empirically: a scene 4 weeks old had zero permissions, one
   3 months old had a full download permission list. This script defaults `--end` to 35 days
   before today for exactly this reason; requesting recent imagery will silently return items
   you cannot download (empty permissions), not an error.
2. **Assets are not immediately downloadable even when permitted.** Every asset starts
   `status: "inactive"` and needs an explicit POST to its `_links.activate` link, then polling
   the same assets endpoint until `status` flips to `"active"` and a `location` URL appears.
   Real observed activation latency: seconds to a couple of minutes, not instant.
3. **The downloaded file is already a proper georeferenced GeoTIFF** (unlike
   acquire_sentinel2.py's raw sentinelhub array, which has to be wrapped in rasterio manually) --
   stream it straight to disk, no reconstruction needed.
4. **`ortho_analytic_4b_sr`** is the asset requested: orthorectified, 4-band (RGB+NIR),
   *surface reflectance* -- the Planet product closest in convention to Sentinel-2 L2A's own
   surface reflectance, though the two sensors' band centers are not identical (PlanetScope's
   are narrower/shifted vs Sentinel-2's) -- a real cross-sensor spectral difference to measure,
   not assume away, exactly like the NAIP/Sentinel-2 pairing protocol already does for v1/v2.

Usage:
    python scripts/acquire_planetscope.py --check
    python scripts/acquire_planetscope.py --bbox 77.55 12.90 77.60 12.95 \\
        --start 2026-01-01 --end 2026-07-01 --max-cloud 0.1 --limit 2 --out data/raw/planetscope
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import requests

EMBARGO_DAYS = 30
ASSET_TYPE = "ortho_analytic_4b_sr"


def _api_key() -> str:
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise RuntimeError("Set PL_API_KEY in .env before using this script.")
    return key


def check(key: str) -> None:
    """Round-trips a real auth + subscription check before spending quota -- same reasoning as
    hf_checkpoint.py's `check`: confirms the account is actually provisioned, not just that the
    key string looks valid."""
    who = requests.get("https://api.planet.com/data/v1/", auth=(key, ""), timeout=15)
    who.raise_for_status()
    print("Data API reachable, key authenticates.")

    subs = requests.get("https://api.planet.com/auth/v1/experimental/public/my/subscriptions",
                        auth=(key, ""), timeout=15).json()
    if not subs:
        print("WARNING: no active subscription on this account -- search may work but "
              "downloads will show empty _permissions on every item.")
        return
    for sub in subs:
        plan = sub["plan"]
        print(f"plan: {plan['name']} (state={plan['state']}) "
              f"quota: {sub['quota_used']:.2f}/{sub['quota_sqkm']} sq km this period "
              f"active_to: {sub['active_to'][:10]}")


def search(bbox, start, end, max_cloud, limit, key):
    payload = {
        "item_types": ["PSScene"],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": {
                    "type": "Polygon",
                    "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                                     [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
                }},
                {"type": "DateRangeFilter", "field_name": "acquired",
                 "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
                {"type": "RangeFilter", "field_name": "cloud_cover",
                 "config": {"lte": max_cloud}},
            ],
        },
    }
    r = requests.post("https://api.planet.com/data/v1/quick-search",
                      auth=(key, ""), json=payload, timeout=30)
    r.raise_for_status()
    items = r.json()["features"][:limit]  # materialize-then-slice; see acquire_sentinel2.py's
                                            # own documented bug for why this order matters
    return items


def order_clipped(item_id, bbox, out_path, key, poll_seconds: float = 5.0,
                  timeout_seconds: float = 600.0):
    """Orders API clip workflow -- delivers just `bbox`, not the full scene.

    Found by direct measurement, not assumption: a plain Data-API asset download (see
    `activate_and_download`) delivers the ENTIRE scene regardless of the search AOI -- one test
    pull for a ~25 sq km search area returned a ~1,026 sq km / 552MB file (the scene's full
    footprint). Quota accounting for that path is also unclear: `quota_used` read 0.0 immediately
    after the download completed, which could be reporting lag or could mean whole-scene
    Data-API downloads aren't what the E&R quota tracks at all -- either way, not something to
    build a multi-AOI batch pull on top of without checking further. The clip tool is the
    mechanism the E&R welcome email itself names ("clip your data with preferred clipping"), and
    is the only way to get a file actually sized to the AOI rather than the whole scene.
    """
    order_payload = {
        "name": f"spectra-sr-{item_id}",
        "products": [{"item_ids": [item_id], "item_type": "PSScene",
                     "product_bundle": "analytic_sr_udm2"}],
        "tools": [{"clip": {"aoi": {
            "type": "Polygon",
            "coordinates": [[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                             [bbox[2], bbox[3]], [bbox[0], bbox[3]], [bbox[0], bbox[1]]]],
        }}}],
    }
    r = requests.post("https://api.planet.com/compute/ops/orders/v2/",
                      auth=(key, ""), json=order_payload, timeout=30)
    r.raise_for_status()
    order = r.json()
    order_url = order["_links"]["_self"]

    deadline = time.time() + timeout_seconds
    while order["state"] not in ("success", "failed"):
        if time.time() > deadline:
            raise TimeoutError(f"order for {item_id} did not complete within {timeout_seconds}s")
        time.sleep(poll_seconds)
        order = requests.get(order_url, auth=(key, ""), timeout=15).json()

    if order["state"] == "failed":
        raise RuntimeError(f"order for {item_id} failed: {order.get('error_hints', order)}")

    # The bundle contains multiple .tif files -- confirmed by inspection: a UDM2 usable-data-mask
    # (`..._3B_udm2_clip.tif`) sorts BEFORE the actual reflectance image
    # (`..._3B_AnalyticMS_SR_clip.tif`) in the results list, so a naive "first .tif" match grabs
    # the mask, not the image -- band value ranges (0/1 flags, one 0-100 "confidence" band) are
    # the giveaway if this regresses. Matched explicitly by name instead of position.
    results = order["_links"]["results"]

    def _find(substr):
        return next((res for res in results if substr in res["name"] and res["name"].endswith(".tif")),
                    None)

    image_result = _find("AnalyticMS_SR_clip")
    if image_result is None:
        raise RuntimeError(f"no AnalyticMS_SR .tif in order results for {item_id}: "
                          f"{[res['name'] for res in results]}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    resp = requests.get(image_result["location"], stream=True, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    # UDM2 usable-data mask, same clip -- real per-pixel cloud/shadow/haze/confidence flags.
    # Downloaded alongside the image so pairing quality can be checked against real cloud data
    # rather than the item-level whole-scene cloud_cover percentage, which measures the ENTIRE
    # scene (often ~100km across) and says nothing about whether this specific small AOI is
    # actually clear.
    udm2_result = _find("udm2_clip")
    if udm2_result is not None:
        udm2_path = out_path.replace(".tif", "_udm2.tif")
        resp = requests.get(udm2_result["location"], stream=True, timeout=60)
        resp.raise_for_status()
        with open(udm2_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    return out_path


def activate_and_download(item, out_path, key, poll_seconds: float = 5.0, timeout_seconds: float = 300.0):
    assets = requests.get(item["_links"]["assets"], auth=(key, ""), timeout=15).json()
    if ASSET_TYPE not in assets:
        raise RuntimeError(f"{item['id']}: {ASSET_TYPE} not in available assets "
                           f"({list(assets.keys())}) -- likely still inside the embargo window.")
    asset = assets[ASSET_TYPE]
    if "download" not in asset.get("_permissions", []):
        raise RuntimeError(f"{item['id']}: no download permission on {ASSET_TYPE} -- inside "
                           f"the 30-day embargo, or the subscription doesn't cover this item type.")

    if asset["status"] != "active":
        requests.post(asset["_links"]["activate"], auth=(key, ""), timeout=15)
        deadline = time.time() + timeout_seconds
        while asset["status"] != "active":
            if time.time() > deadline:
                raise TimeoutError(f"{item['id']}: asset did not activate within "
                                   f"{timeout_seconds}s -- try again later.")
            time.sleep(poll_seconds)
            assets = requests.get(item["_links"]["assets"], auth=(key, ""), timeout=15).json()
            asset = assets[ASSET_TYPE]

    resp = requests.get(asset["location"], auth=(key, ""), stream=True, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="Verify auth + subscription, spend no quota, exit.")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"))
    p.add_argument("--start", type=str)
    p.add_argument("--end", type=str, default=None,
                   help=f"Defaults to {EMBARGO_DAYS} days before today, to avoid the embargo "
                        f"window by construction rather than discovering it via empty "
                        f"permissions after a wasted search.")
    p.add_argument("--max-cloud", type=float, default=0.1,
                   help="Fraction 0-1, not percent -- Planet's own convention.")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--out", type=str, default="data/raw/planetscope")
    p.add_argument("--no-clip", action="store_true",
                   help="Use the plain Data-API download instead of an Orders-API clip. "
                        "Delivers the FULL scene (measured: ~1,000 sq km / ~550MB for one item, "
                        "regardless of --bbox size) -- default is the clip path, which actually "
                        "sizes the file to --bbox.")
    args = p.parse_args()

    api_key = _api_key()
    if args.check:
        check(api_key)
        raise SystemExit(0)

    if not (args.bbox and args.start):
        raise SystemExit("--bbox and --start are required unless --check is passed.")
    end = args.end or (datetime.now(timezone.utc) - timedelta(days=EMBARGO_DAYS)).strftime("%Y-%m-%d")

    items = search(tuple(args.bbox), args.start, end, args.max_cloud, args.limit, api_key)
    print(f"Found {len(items)} PSScene items (after limiting to {args.limit}), end date {end}")

    for item in items:
        out_path = os.path.join(args.out, f"{item['id']}.tif")
        try:
            if args.no_clip:
                path = activate_and_download(item, out_path, api_key)
            else:
                path = order_clipped(item["id"], tuple(args.bbox), out_path, api_key)
            print(f"  saved {path} (acquired {item['properties']['acquired'][:10]}, "
                  f"cloud_cover {item['properties']['cloud_cover']:.3f})")
        except (RuntimeError, TimeoutError) as exc:
            print(f"  skipped {item['id']}: {exc}")
