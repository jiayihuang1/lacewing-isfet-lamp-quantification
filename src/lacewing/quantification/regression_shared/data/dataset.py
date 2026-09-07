"""Regression dataset / split / loaders.

Reads a regression cache (X, y_ttp_min, chip_id, well_id, pixel_id,
split) and builds train/val/test torch DataLoaders.

Split convention (matches lacewing.quantification.regression_shared.data.labels):
  - split == 0 -> train  (5 dose-response chips, wells 0-3)
  - split == 1 -> test   (SD chip, wells 0-4)

A train/val carve-out is done WITHIN the train split using
StratifiedGroupKFold on (chip_id), keeping every chip intact in either
train or val.  Stratifying by chip guarantees val sees at least one
held-out chip (TTP label), so val MAE is a meaningful early-stop signal
rather than a noisy in-distribution number.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from lacewing.quantification.regression_shared.core import paths


DEFAULT_VAL_FRAC = 0.20         # one chip out of five = 20%


@dataclass
class SplitArrays:
    X_tr: np.ndarray; y_tr: np.ndarray
    X_va: np.ndarray; y_va: np.ndarray
    X_te: np.ndarray; y_te: np.ndarray
    chip_te: np.ndarray
    well_te: np.ndarray
    pixel_te: np.ndarray
    # Train+val identifiers (for per-chip val MAE) and the target's
    # train-set mean/std so callers can de-normalise predictions.
    chip_tr: np.ndarray
    chip_va: np.ndarray
    well_tr: np.ndarray
    well_va: np.ndarray
    pixel_tr: np.ndarray   # cache pixel id per training example
    pixel_va: np.ndarray   # cache pixel id per validation example
    y_mean: float
    y_std:  float


class PixelDataset(Dataset):
    """(1, T) per-pixel signal + scalar regression target."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        import torch
        x = torch.from_numpy(self.X[idx][None, :])      # (1, T)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def _load_cache(cache_stem: str) -> dict:
    path = paths.regression_cache_path(cache_stem)
    if not path.exists():
        raise FileNotFoundError(f"regression cache not found: {path}")
    with np.load(path) as data:
        return {
            "X":         data["X"].astype(np.float32, copy=False),
            "y_ttp_min": data["y_ttp_min"].astype(np.float32, copy=False),
            "chip_id":   np.asarray(data["chip_id"]),
            "well_id":   np.asarray(data["well_id"]),
            "pixel_id":  np.asarray(data["pixel_id"]),
            "split":     np.asarray(data["split"]),
        }


def make_split(cache_stem: str, seed: int = 0,
               val_n_chips: int = 1, normalise_y: bool = True
               ) -> SplitArrays:
    """Return train/val/test arrays with `val_n_chips` chips held out as val.

    The validation set is a single dose-response chip (default; you
    can bump val_n_chips to hold out more).  Selection is
    deterministic-from-seed via a single shuffle of the unique train
    chip names.

    If ``normalise_y`` is True, the returned y_tr / y_va arrays are
    standardised to mean 0, std 1 using TRAIN-only statistics; y_te
    keeps the raw qLAMP TTP in minutes (so test MAE is reported in
    minutes directly).  The mean/std needed to de-normalise predictions
    is exposed on the SplitArrays.
    """
    data = _load_cache(cache_stem)
    X = data["X"]
    y = data["y_ttp_min"]
    chip = data["chip_id"]
    well = data["well_id"]
    pix  = data["pixel_id"]
    split = data["split"]

    # Drop NaN-labelled pixels (NTC / PTC) before splitting. The KP
    # regression cache retains them so the RQ1 classifier training has
    # negative examples; the regression harness requires valid TTP labels
    # on every training pixel. On the SARS-CoV-2 cache this is a no-op
    # because build_regression_cache.py already filters at write time.
    keep = ~np.isnan(y)
    if keep.sum() < len(y):
        n_dropped = int((~keep).sum())
        print(f"  make_split: dropped {n_dropped} NaN-labelled pixels "
              f"({100*n_dropped/len(y):.1f}%) from cache '{cache_stem}'")
        X, y, chip, well, pix, split = X[keep], y[keep], chip[keep], well[keep], pix[keep], split[keep]

    idx_train_all = np.flatnonzero(split == 0)
    idx_test      = np.flatnonzero(split == 1)

    # Carve out val: pick `val_n_chips` chips from the train chip set.
    train_chips = sorted(np.unique(chip[idx_train_all]).tolist())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_chips))
    val_chip_set = set(train_chips[i] for i in perm[:val_n_chips])

    val_mask = np.isin(chip, sorted(val_chip_set))
    idx_va = np.intersect1d(idx_train_all, np.flatnonzero(val_mask))
    idx_tr = np.setdiff1d(idx_train_all, idx_va)

    y_tr_min = y[idx_tr]                       # raw minutes
    y_va_min = y[idx_va]
    y_te_min = y[idx_test]

    if normalise_y:
        y_mean = float(y_tr_min.mean())
        y_std  = float(y_tr_min.std() + 1e-8)
        y_tr_arr = (y_tr_min - y_mean) / y_std
        y_va_arr = (y_va_min - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_tr_arr = y_tr_min.copy()
        y_va_arr = y_va_min.copy()

    # Test labels stay in minutes — we de-normalise predictions before
    # comparing.
    y_te_arr = y_te_min.copy()

    return SplitArrays(
        X_tr=X[idx_tr], y_tr=y_tr_arr,
        X_va=X[idx_va], y_va=y_va_arr,
        X_te=X[idx_test], y_te=y_te_arr,
        chip_te=chip[idx_test], well_te=well[idx_test], pixel_te=pix[idx_test],
        chip_tr=chip[idx_tr], chip_va=chip[idx_va],
        well_tr=well[idx_tr], well_va=well[idx_va],
        pixel_tr=pix[idx_tr], pixel_va=pix[idx_va],
        y_mean=y_mean, y_std=y_std,
    )


def make_loaders(arr: SplitArrays, batch_size: int = 16,
                 num_workers: int = 0
                 ) -> tuple[DataLoader, DataLoader, DataLoader]:
    tr = DataLoader(PixelDataset(arr.X_tr, arr.y_tr),
                    batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, drop_last=False)
    va = DataLoader(PixelDataset(arr.X_va, arr.y_va),
                    batch_size=256, shuffle=False, num_workers=num_workers)
    te = DataLoader(PixelDataset(arr.X_te, arr.y_te),
                    batch_size=256, shuffle=False, num_workers=num_workers)
    return tr, va, te
