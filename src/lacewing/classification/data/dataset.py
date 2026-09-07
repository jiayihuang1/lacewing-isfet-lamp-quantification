"""Torch Dataset wrapper + train/val/test split logic.

The cached .npz built by build_dataset.py contains a flat (N_pixels, 450)
matrix. This module:
  1. Loads the cache once and applies the chosen feature transform
     (raw trace or spectrogram).
  2. Builds split indices for either:
       - 'random'  : stratified pixel-level 70/15/15
       - 'chip'    : leave-one-Final-chip-out cross-validation;
                     the held-out chip's pixels go to test, the train
                     pool is split 85/15 train/val (stratified by class).
  3. Returns torch DataLoaders ready for training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from . import build_dataset
from ..core import paths
from ..features import spectrogram, time_domain


# k for the new `chipkfold` split. 5 folds on a 31-chip cache gives
# ~25 chips train+val / ~6 chips test per fold — the supervisor's
# Week-4 "50:50 / more chips for validation" ask, generalised to
# k-fold CV so we can average across folds (see Meeting_Notes Week 4).
KFOLD_K = 5
# k_val controls how the train pool is sub-split into train/val while
# keeping chips intact. k_val=7 -> ~15% of chips become val.
KFOLD_VAL_K = 7


@dataclass
class SplitArrays:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    # Test-set pixel metadata for well-level majority voting at evaluation
    chip_te: np.ndarray
    well_te: np.ndarray
    pixel_te: np.ndarray
    # Raw 450-sample test traces (before feature transform). Needed for
    # failure-analysis viewers that must show the original signal even
    # when the model itself was fed spectrograms (cnn2d_spec).
    X_raw_te: np.ndarray | None = None


class PixelDataset(Dataset):
    """Generic torch Dataset for either raw-trace or spectrogram features."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        # Add channel dim where needed
        if x.ndim == 1:                 # raw trace -> (1, T)
            x = x.unsqueeze(0)
        elif x.ndim == 2:               # spectrogram -> (1, F, T_win)
            x = x.unsqueeze(0)
        return x, torch.tensor(self.y[idx], dtype=torch.float32)


def _load_cache(cache_path=None):
    cache_path = cache_path or paths.cache_path("dataset_per_pixel")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache {cache_path} not found. Run "
            "`python -m lacewing.classification.data.build_dataset` first."
        )
    return np.load(cache_path, allow_pickle=False)


def _apply_features(X: np.ndarray, features: str) -> np.ndarray:
    if features == "raw":
        return time_domain.transform(X)
    if features == "spectrogram":
        return spectrogram.transform(X)
    raise ValueError(f"Unknown features: {features}")


def make_split(features: str, split: str, fold: int | str = 0,
               seed: int = 0, cache_path=None) -> SplitArrays:
    """Build SplitArrays for one (features, split, fold, seed) config.

    Parameters
    ----------
    features : 'raw' or 'spectrogram'
    split    : 'random' or 'chip'
    fold     : ignored if split == 'random'; for 'chip', the held-out
               chip key (must match a key in paths.FINAL_CHIPS, or any
               chip-name string present in the cache's `chip_id` array).
    seed     : RNG seed for the split.
    cache_path : optional override for the dataset cache path.
    """
    cache = _load_cache(cache_path)
    X_raw = cache["X"]
    y = cache["y"]
    chip = cache["chip_id"]
    well = cache["well_id"]
    pix = cache["pixel_id"]

    X = _apply_features(X_raw, features)

    if split == "random":
        idx_all = np.arange(len(y))
        idx_trval, idx_te = train_test_split(
            idx_all, test_size=0.15, stratify=y, random_state=seed,
        )
        # 0.15/0.85 = 0.1765 of trval -> val; remaining ~70% of total -> train
        idx_tr, idx_va = train_test_split(
            idx_trval, test_size=0.15 / 0.85, stratify=y[idx_trval],
            random_state=seed,
        )
    elif split == "chip":
        # Accept either a Final-chip key (e.g. '1e5') or a literal chip
        # folder name (so chip-fold works with any all-data cache too).
        fold_str = str(fold)
        if fold_str in paths.FINAL_CHIPS:
            held_chip_name = paths.FINAL_CHIPS[fold_str].name
        else:
            held_chip_name = fold_str
        is_held = (chip == held_chip_name)
        if not is_held.any():
            raise ValueError(
                f"chip-fold='{fold}' not found in cache; expected one of "
                f"FINAL_CHIPS keys or a chip folder name present in chip_id."
            )
        idx_te = np.where(is_held)[0]
        idx_pool = np.where(~is_held)[0]
        idx_tr, idx_va = train_test_split(
            idx_pool, test_size=0.15, stratify=y[idx_pool],
            random_state=seed,
        )
    elif split == "sd_test":
        # exp6: train+val from the 5 Final chips, test = the chips named in
        # `fold` (comma-separated chip folder names — defaults to the two
        # supervisor-selected SD chips).
        #
        # Stratified by class and grouped by chip so each Final chip
        # contributes both train and val pixels (option (ii) from the
        # Week-5 design discussion).
        if fold in (0, "0", None, ""):
            fold_str = (
                "D20240719_E03_C44_F4500KHz_U_COV_SD,"
                "D20240814_E00_C00_F4500KHz_U_PnG_Bead"
            )
        else:
            fold_str = str(fold)
        test_chips = [c.strip() for c in fold_str.split(",") if c.strip()]
        final_chip_names = {p.name for p in paths.FINAL_CHIPS.values()}
        is_test = np.isin(chip, test_chips)
        is_final = np.isin(chip, list(final_chip_names))
        if not is_test.any():
            raise ValueError(
                f"sd_test: none of the requested test chips {test_chips} "
                f"were found in the cache `chip_id` column. Run "
                "build_dataset.py with --scope all so the SD chips are "
                "included.")
        if not is_final.any():
            raise ValueError(
                "sd_test: no 5-Final-chip pixels found in the cache. "
                "Run build_dataset.py with --scope all (the all-data cache "
                "contains both Final and SD chips).")
        idx_te = np.where(is_test)[0]
        idx_pool = np.where(is_final)[0]
        # Chip-stratified 85/15 inside the Final pool: each Final chip
        # contributes ~85% of its pixels to train and ~15% to val, with
        # class balance preserved.
        pool_y = y[idx_pool]
        pool_chip = chip[idx_pool]
        # Use StratifiedGroupKFold with k = 7 to get ~14% chips out per
        # split — but here we DON'T want a leave-chip-out: we want each
        # chip in both train AND val. So fall back to per-chip pixel-level
        # stratified split.
        rng = np.random.default_rng(seed)
        tr_inner: list[int] = []
        va_inner: list[int] = []
        for c in np.unique(pool_chip):
            sub = np.where(pool_chip == c)[0]
            sub_y = pool_y[sub]
            sub_tr, sub_va = train_test_split(
                sub, test_size=0.15, stratify=sub_y,
                random_state=int(rng.integers(0, 2**31 - 1)))
            tr_inner.extend(sub_tr.tolist())
            va_inner.extend(sub_va.tolist())
        idx_tr = idx_pool[np.array(tr_inner)]
        idx_va = idx_pool[np.array(va_inner)]
    elif split == "chipkfold":
        # k-fold chip-level CV. Each fold holds out a *group* of chips
        # (not a single chip) so the test set is larger (~6 chips out of
        # 31) and the across-fold variance reflects model generalisation,
        # not "which one chip got unlucky."
        #
        # Stratification respects the positive/negative pixel ratio per
        # fold; grouping ensures no chip leaks between train/test.
        try:
            k = int(KFOLD_K)
        except Exception:
            k = 5
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        # StratifiedGroupKFold yields k (train_idx, test_idx) splits.
        # We pick the one indexed by `fold` (0..k-1).
        fold_i = int(fold)
        if fold_i < 0 or fold_i >= k:
            raise ValueError(
                f"chipkfold fold={fold} out of range for k={k}; expected 0..{k-1}.")
        all_splits = list(sgkf.split(np.zeros(len(y)), y, groups=chip))
        idx_pool_train, idx_te = all_splits[fold_i]
        # Within the train pool, carve out an 85/15 train/val split that
        # also keeps chips intact between train and val. Re-use the same
        # SGKF idea but with k = round(1 / 0.15) ~= 7 splits and grab one
        # group of ~15% of chips as val.
        try:
            k_val = int(KFOLD_VAL_K)
        except Exception:
            k_val = 7
        sgkf_val = StratifiedGroupKFold(n_splits=k_val, shuffle=True,
                                        random_state=seed + 1)
        pool_y = y[idx_pool_train]
        pool_chip = chip[idx_pool_train]
        train_inner, val_inner = next(iter(
            sgkf_val.split(np.zeros(len(pool_y)), pool_y, groups=pool_chip)))
        idx_tr = idx_pool_train[train_inner]
        idx_va = idx_pool_train[val_inner]
    else:
        raise ValueError(f"Unknown split: {split}")

    return SplitArrays(
        X_tr=X[idx_tr], y_tr=y[idx_tr],
        X_va=X[idx_va], y_va=y[idx_va],
        X_te=X[idx_te], y_te=y[idx_te],
        chip_te=chip[idx_te], well_te=well[idx_te], pixel_te=pix[idx_te],
        X_raw_te=X_raw[idx_te].astype(np.float32),
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
