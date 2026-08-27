import logging
import os
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spectra_sr")
# Explicit level on this logger itself, not just via basicConfig on the root -- a logger with no
# level of its own (NOTSET) defers to its parent's *current* effective level, which pytest's
# logging plugin (root defaults to WARNING) can silently override after import, swallowing every
# INFO message without erroring. Same fix as optical_guided_sr.utils.
logger.setLevel(logging.INFO)


def setup_file_logging(log_path: str) -> None:
    """Mirror every `logger` call to a text file, not just the console -- a multi-hour training
    run leaves nothing on disk to inspect if the session dies otherwise."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.info(f"Logging to file: {log_path}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Make a run reproducible across numpy / random / torch.

    `deterministic=True` also pins cuDNN to deterministic algorithms -- required for bit-for-bit
    repro on GPU (at some throughput cost); without it, two runs with the same seed can still
    diverge because cuDNN picks non-deterministic conv algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
