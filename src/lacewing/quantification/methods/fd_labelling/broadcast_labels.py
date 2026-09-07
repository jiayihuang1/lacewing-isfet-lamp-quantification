"""Broadcast manual boundary labels to per-pixel per-timestep label arrays.

Reads manual_labels.json from Task 3 + the raw regression cache. Emits
manual_labels.npz containing per-pixel int8 label arrays of shape (N, 450)
plus a well-level train/holdout split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lacewing.quantification.regression_shared.data.dataset import _load_cache
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


N_SAMPLES = 450
CLS_BASELINE = 0
CLS_DRIFT = 1
CLS_RISING = 2
CLS_POST_AMP = 3

IN_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.json"
OUT_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.npz"


def boundaries_to_labels(b1: int, b2: int, b3: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Convert (b1, b2, b3) to a per-timestep int8 4-class label array."""
    labels = np.empty(n_samples, dtype=np.int8)
    labels[:b1] = CLS_BASELINE
    labels[b1:b2] = CLS_DRIFT
    labels[b2:b3] = CLS_RISING
    labels[b3:] = CLS_POST_AMP
    return labels


def pick_holdout_wells(all_wells: list[str], seed: int, n_holdout: int = 6) -> list[str]:
    """Deterministic random selection of n_holdout wells for seg-eval holdout."""
    wells_sorted = sorted(all_wells)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(wells_sorted))
    picked = [wells_sorted[i] for i in sorted(perm[:n_holdout].tolist())]
    return picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--n-holdout", type=int, default=6)
    args = p.parse_args()

    if not IN_PATH.exists():
        raise SystemExit(f"missing {IN_PATH} — run build_labels.py first")
    data = json.loads(IN_PATH.read_text())
    wells = data["wells"]
    if not wells:
        raise SystemExit("no wells labelled")

    ok_wells = [
        wk for wk, entry in wells.items()
        if any(v.get("status") == "ok" for v in entry.get("labels", {}).values())
    ]
    if not ok_wells:
        raise SystemExit("no wells have any 'ok' cluster labels")

    holdout = set(pick_holdout_wells(ok_wells, seed=args.holdout_seed,
                                     n_holdout=args.n_holdout))
    print(f"[broadcast] {len(ok_wells)} wells with ok labels; "
          f"holdout ({len(holdout)}): {sorted(holdout)}")

    pixel_ids: list[int] = []
    label_arrays: list[np.ndarray] = []
    well_keys: list[str] = []
    for wk, entry in wells.items():
        for cluster_str, meta in entry.get("labels", {}).items():
            if meta.get("status") != "ok":
                continue
            b1, b2, b3 = meta["boundaries"]
            lbl = boundaries_to_labels(b1, b2, b3)
            pixels = entry["pixel_indices_in_cache"].get(cluster_str, [])
            for p_idx in pixels:
                pixel_ids.append(int(p_idx))
                label_arrays.append(lbl)
                well_keys.append(wk)

    if not pixel_ids:
        raise SystemExit("no labelled pixels after filtering")

    pixel_ids_arr = np.array(pixel_ids, dtype=np.int32)
    labels_arr = np.stack(label_arrays, axis=0)  # (N, 450)
    well_keys_arr = np.array(well_keys, dtype="U64")
    holdout_mask = np.array([wk in holdout for wk in well_keys], dtype=bool)
    train_mask = ~holdout_mask

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        pixel_ids=pixel_ids_arr,
        labels=labels_arr,
        well_keys=well_keys_arr,
        train_pixel_mask=train_mask,
        holdout_pixel_mask=holdout_mask,
    )
    print(f"[broadcast] wrote {OUT_PATH}: {len(pixel_ids)} pixels "
          f"({train_mask.sum()} train / {holdout_mask.sum()} holdout)")


if __name__ == "__main__":
    main()
