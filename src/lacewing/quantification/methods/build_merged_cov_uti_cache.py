"""Merge COV and UTI caches for the W18 mixed-regime experiments.

Produces two output caches:
    Analysis/quantification/methods/cache/merged_cov_uti_cls_cache.npz  (classification)
    Analysis/quantification/methods/cache/merged_cov_uti_reg_cache.npz  (regression)

Both carry `origin` in {"cov", "uti"} per pixel so downstream training scripts
can do leave-one-UTI-chip-out CV while keeping ALL COV pixels in the training
pool.

Regression cache: UTI pixels ALL get split=0 (train). Actual test set is
selected at training time by holding out one UTI chip's rows. This mirrors
how the existing COV reg cache uses `split=1` for test -- we don't reuse
that slot for UTI because LOOCV loops through multiple holdouts.

Schema notes:
- chip_id namespaces are disjoint: COV chips start `D2024...`; UTI chips
  start `uti_...`. We do NOT prefix -- the source caches already produce
  non-colliding strings.
- chip_id is stored as <U64 (not <U40) because at least one COV chip name
  (`D20240704_E00_C00_F4500KHz_U_png6w_dilut_2`, 42 chars) exceeds 40
  characters; using <U40 would silently truncate it and could collide
  with another chip name.
- pixel_id: COV keeps its original int; UTI gets 0..n-1 within each well.
- well_id: both int32.
- UTI classifier/segmentation caches only contain the chips currently
  present in their source .npz files -- this code does not hardcode chip
  counts or names, so it keeps working as more UTI chips (e.g. KP_03) are
  added upstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lacewing.classification import paths as cls_paths
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

PROJECT_ROOT = DATA_ROOT  # was: parents[3]
COV_CLS_CACHE_STEM = "dataset_all_filt_abcd_ntcRaw_madk1p5_spatA3"
COV_REG_CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"
UTI_CLS_PATH = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output" / "uti_classifier_cache.npz"
UTI_REG_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "uti_seg_manual_labels_v1" / "cache.npz"
DEFAULT_OUT_DIR = LACEWING_PKG_DIR / "quantification" / "methods" / "cache"

CHIP_ID_DTYPE = "<U64"


def _load_cov_cls() -> dict:
    """Load COV classification cache and coerce to the merged schema."""
    p = cls_paths.cache_path(COV_CLS_CACHE_STEM)
    if not p.exists():
        # Fall back to the ntcRaw base if the spatA3 version isn't on disk (local dev)
        p = cls_paths.cache_path("dataset_all_filt_abcd_ntcRaw")
    d = np.load(p, allow_pickle=False)
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(np.int8),
        "chip_id": d["chip_id"].astype(CHIP_ID_DTYPE),
        "well_id": d["well_id"].astype(np.int32),
        "pixel_id": d["pixel_id"].astype(np.int32),
    }


def _load_uti_cls() -> dict:
    if not UTI_CLS_PATH.exists():
        raise SystemExit(f"UTI classifier cache missing: {UTI_CLS_PATH}. Run build_uti_classifier_cache first.")
    d = np.load(UTI_CLS_PATH, allow_pickle=False)
    # UTI cache has no pixel_id; synthesise 0..n-1 within each (chip, well)
    N = len(d["X"])
    pixel_id = np.zeros(N, dtype=np.int32)
    # sequential per chip x well ordering
    key = np.array([f"{c}::{w}" for c, w in zip(d["chip_tag"], d["well_id"])])
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        pixel_id[idx] = np.arange(len(idx))
    return {
        "X": d["X"].astype(np.float32),
        "y": d["y"].astype(np.int8),
        "chip_id": d["chip_tag"].astype(CHIP_ID_DTYPE),
        "well_id": d["well_id"].astype(np.int32),
        "pixel_id": pixel_id,
    }


def build_merged_cls_cache(out_path: Path | None = None) -> Path:
    """Build the merged COV+UTI classification cache."""
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / "merged_cov_uti_cls_cache.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cov = _load_cov_cls()
    uti = _load_uti_cls()
    n_cov, n_uti = len(cov["X"]), len(uti["X"])
    origin = np.concatenate([
        np.full(n_cov, "cov", dtype="<U8"),
        np.full(n_uti, "uti", dtype="<U8"),
    ])
    np.savez_compressed(
        out_path,
        X=np.concatenate([cov["X"], uti["X"]], axis=0),
        y=np.concatenate([cov["y"], uti["y"]], axis=0),
        chip_id=np.concatenate([cov["chip_id"], uti["chip_id"]], axis=0),
        well_id=np.concatenate([cov["well_id"], uti["well_id"]], axis=0),
        pixel_id=np.concatenate([cov["pixel_id"], uti["pixel_id"]], axis=0),
        origin=origin,
    )
    return out_path


def build_merged_reg_cache(out_path: Path | None = None) -> Path:
    """Build the merged COV+UTI regression cache."""
    if out_path is None:
        out_path = DEFAULT_OUT_DIR / "merged_cov_uti_reg_cache.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # COV: use the same _load_cache the training pipeline uses to preserve `split`.
    from lacewing.quantification.regression_shared.data.dataset import _load_cache
    cov = _load_cache(COV_REG_CACHE_STEM)

    if not UTI_REG_PATH.exists():
        raise SystemExit(f"UTI seg cache missing: {UTI_REG_PATH}. Run build_plate_segmentation_cache first.")
    u = np.load(UTI_REG_PATH, allow_pickle=False)

    # UTI: reconstruct pixel_id sequentially within each (chip, well)
    N_uti = len(u["X"])
    uti_pixel_id = np.zeros(N_uti, dtype=np.int32)
    key = np.array([f"{c}::{w}" for c, w in zip(u["chip_tag"], u["well_id"])])
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        uti_pixel_id[idx] = np.arange(len(idx))

    # UTI y = user_ttp_min (per-pixel). All UTI pixels are amp+ by construction.
    uti_y = u["user_ttp_min"].astype(np.float32)

    n_cov, n_uti = len(cov["X"]), N_uti
    origin = np.concatenate([
        np.full(n_cov, "cov", dtype="<U8"),
        np.full(n_uti, "uti", dtype="<U8"),
    ])

    np.savez_compressed(
        out_path,
        X=np.concatenate([cov["X"].astype(np.float32), u["X"].astype(np.float32)], axis=0),
        y_ttp_min=np.concatenate([cov["y_ttp_min"].astype(np.float32), uti_y], axis=0),
        chip_id=np.concatenate([cov["chip_id"].astype(CHIP_ID_DTYPE), u["chip_tag"].astype(CHIP_ID_DTYPE)], axis=0),
        well_id=np.concatenate([cov["well_id"].astype(np.int32), u["well_id"].astype(np.int32)], axis=0),
        pixel_id=np.concatenate([cov["pixel_id"].astype(np.int32), uti_pixel_id], axis=0),
        split=np.concatenate([cov["split"].astype(np.int8), np.zeros(N_uti, dtype=np.int8)], axis=0),
        origin=origin,
    )
    return out_path


def main() -> None:
    print("Building merged CLS cache...")
    p = build_merged_cls_cache()
    print(f"[ok] wrote {p}")

    print("Building merged REG cache...")
    p = build_merged_reg_cache()
    print(f"[ok] wrote {p}")


if __name__ == "__main__":
    main()
