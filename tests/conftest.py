"""Shared test fixtures. `naip_primary_file` exists specifically to fix a real, recurring test
fragility: every test module that used `glob.glob(...)[0]` to pick "a" real NAIP file was
implicitly depending on directory listing order, which silently changes whenever a file is
added/removed (e.g. by re-running acquire_naip.py) -- confirmed directly when adding more real
NAIP patches made an unrelated, previously-passing overfit test fail, because [0] now resolved
to a different real image with a (legitimately, slightly) different convergence rate for the
same fixed step budget. Pinning to one specific, known-good filename fixes the flakiness at its
actual source instead of re-tuning thresholds every time the directory changes.
"""
import glob
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAIP_DIR = os.path.join(REPO_ROOT, "data/raw/naip")
_PREFERRED_FILE = "ca_m_3611918_nw_11_060_20220620.tif"  # confirmed within-bbox, used
                                                          # successfully throughout this session


def _naip_files():
    return sorted(
        f for f in glob.glob(os.path.join(NAIP_DIR, "*.tif"))
        if not os.path.basename(f).startswith("_full_")
    )


@pytest.fixture
def naip_primary_file():
    files = _naip_files()
    if not files:
        pytest.skip("no real NAIP file in data/raw/naip/")
    preferred = os.path.join(NAIP_DIR, _PREFERRED_FILE)
    return preferred if preferred in files else files[0]
