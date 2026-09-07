# Analysis/quantification/methods/fd_labelling/build_labels.py
"""Orchestrator: cluster pixels per well and drive the labelling UI.

Iterates wells in the regression cache, clusters each well's pixels into
k groups, picks the closest-to-centroid pixel as representative, and drives
the matplotlib labelling UI to collect (b1, b2, b3) boundaries per cluster.
Saves progress after every well.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing.quantification.methods.fd_labelling.cluster_pixels import cluster_well
from lacewing.quantification.methods.fd_labelling.labelling_ui import (
    label_cluster,
    quit_requested,
    _reset_quit,
)
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"
N_SAMPLES = 450
SAMPLES_PER_MIN = 15
OUT_PATH = LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.json"


def _load_cache_full():
    """Load the raw cache (not the split); returns dict with X, y_ttp_min, chip_id, well_id, pixel_id."""
    from lacewing.quantification.regression_shared.data.dataset import _load_cache
    return _load_cache(CACHE_STEM)


def _group_pixels_by_well(cache: dict) -> dict[tuple[str, int], np.ndarray]:
    """Return {(chip_id, well_id): pixel_indices_into_cache} for amp-positive pixels only.

    Amp-positive means y_ttp_min is not NaN.
    """
    y = cache["y_ttp_min"]
    valid = ~np.isnan(y)
    chip = cache["chip_id"]
    well = cache["well_id"]
    out: dict[tuple[str, int], np.ndarray] = {}
    for i in np.flatnonzero(valid):
        key = (str(chip[i]), int(well[i]))
        out.setdefault(key, []).append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def _pick_representative_pixel(
    traces: np.ndarray, cluster_ids: np.ndarray, cluster_id: int
) -> int:
    """Return the index (within `traces`) of the pixel closest to the cluster's centroid.

    Feature = per-trace normalised shape (matches cluster_pixels.cluster_well internals).
    Returns a local index into `traces` (not a global cache index).
    """
    from lacewing.quantification.methods.fd_labelling.cluster_pixels import _normalise_trace
    mask = cluster_ids == cluster_id
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        raise ValueError(f"cluster {cluster_id} has no pixels")
    feats = _normalise_trace(traces[idxs])
    centroid = feats.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(feats - centroid, axis=1)
    best = int(idxs[int(dists.argmin())])
    return best


def _load_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "cache_stem": CACHE_STEM,
            "n_samples": N_SAMPLES,
            "samples_per_min": SAMPLES_PER_MIN,
            "k_per_well": None,
            "wells": {},
        }
    return json.loads(path.read_text())


def _save(existing: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wells", default="all",
                        help="Comma-separated 'chip_id::well_id' keys, or 'all'.")
    parser.add_argument("--k", type=int, default=4,
                        help="Clusters per well.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip wells already fully labelled.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo even if already labelled.")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise SystemExit("choose --resume OR --overwrite, not both")

    print(f"[build_labels] loading cache {CACHE_STEM}")
    cache = _load_cache_full()
    groups = _group_pixels_by_well(cache)
    print(f"[build_labels] {len(groups)} amp-positive wells in cache")

    existing = _load_existing(OUT_PATH)
    existing["k_per_well"] = args.k
    if args.overwrite:
        existing["wells"] = {}

    # Filter wells to consider.
    if args.wells == "all":
        wells_to_do = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    else:
        pairs = [w.split("::") for w in args.wells.split(",")]
        wells_to_do = [(chip, int(well)) for chip, well in pairs]

    _reset_quit()

    for chip_id, well_id in wells_to_do:
        key = f"{chip_id}::{well_id}"
        if args.resume and key in existing["wells"]:
            # Check if all k clusters are labelled (ok or invalid, not skipped/missing).
            entry = existing["wells"][key]
            if len(entry.get("labels", {})) == args.k:
                print(f"[skip] {key}: already labelled")
                continue

        if (chip_id, well_id) not in groups:
            print(f"[warn] {key}: no amp-positive pixels")
            continue
        pixel_idxs = groups[(chip_id, well_id)]
        traces = cache["X"][pixel_idxs]  # (n_pixels, T)

        if len(pixel_idxs) < args.k:
            print(f"[skip] {key}: only {len(pixel_idxs)} pixels < k={args.k}")
            continue

        cluster_ids = cluster_well(traces, k=args.k, random_state=0)

        entry: dict[str, Any] = existing["wells"].setdefault(key, {})
        entry.setdefault("cluster_ids", list(range(args.k)))
        pixel_map = entry.setdefault("pixel_indices_in_cache", {})
        labels = entry.setdefault("labels", {})

        for c in range(args.k):
            if str(c) in labels and not args.overwrite:
                continue
            in_cluster = np.flatnonzero(cluster_ids == c)
            if len(in_cluster) == 0:
                print(f"[warn] {key} cluster {c}: no pixels — skipping")
                continue
            # rep_pixel_local is an index into traces (local, not global cache index)
            rep_pixel_local = _pick_representative_pixel(traces, cluster_ids, c)
            rep_trace = traces[rep_pixel_local]
            rep_ttp = float(np.median(cache["y_ttp_min"][pixel_idxs[in_cluster]]))

            pixel_map[str(c)] = pixel_idxs[in_cluster].tolist()

            print(f"[label] {key} cluster {c}: {len(in_cluster)} pixels, rep_ttp={rep_ttp:.2f} min")
            result = label_cluster(
                trace=rep_trace,
                ttp_true_min=rep_ttp,
                well_id=key,
                cluster_id=c,
                samples_per_min=SAMPLES_PER_MIN,
            )
            if result is None:
                if quit_requested():
                    print("[build_labels] quit requested — saving progress and exiting")
                    _save(existing, OUT_PATH)
                    return
                print(f"[skip] {key} cluster {c}: deferred")
                continue
            labels[str(c)] = result

        _save(existing, OUT_PATH)
        print(f"[saved] {key} — {len(labels)}/{args.k} clusters labelled")

    _save(existing, OUT_PATH)
    print(f"[done] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
