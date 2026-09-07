"""Deterministic-seed helpers for reproducible runs."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set seeds for python, numpy, torch (cpu + cuda), and CUDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
