"""Stage 1: fit a 5PL sigmoid to every per-pixel trace in the classification
cache and write a fits artefact.

Input:  Analysis/classification/data/cache/<dataset>.npz
        (defaults to dataset_per_pixel.npz, as built by build_dataset.py)
Output: Analysis/features/cache/fits_<dataset>.npz

The fits artefact is row-aligned to the input cache (same N, same row order)
so downstream stats can join by index.

Columns:
    chip_id, well_id, pixel_id, y           - copied through from input
    params_5pl (N, 5)                       - Fm, Fb, Sc, Cs, As
    r_squared (N,)                          - fit R^2 on the raw trace
    fit_ok (N,) bool                        - all params finite and fit converged
    amp_Fm, amp_xs_xe, amp_fmax_fmin,
        amp_endpoints (N,)                  - four amplitude definitions, signed
    xs, xe, xms, xp1, xp2 (N,)              - critical-point t-coordinates
    F_max, F_min, F0 (N,)                   - fitted curve extrema and raw start

Run:
    python -m lacewing.features.fit_dataset
    python -m lacewing.features.fit_dataset --dataset dataset_all --jobs 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

THIS_FILE = Path(__file__).resolve()
VIEWER_DIR = THIS_FILE.parent
FEATURES_DIR = VIEWER_DIR.parent
ANALYSIS_DIR = FEATURES_DIR.parent
PROJECT_ROOT = ANALYSIS_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from lacewing.features.sigmoid_fitting import (  # noqa: E402
    fit_5p,
    sigmoid_5p,
    extract_kinetic_parameters,
)

CLASSIFICATION_CACHE = ANALYSIS_DIR / "classification" / "data" / "cache"
OUT_DIR = VIEWER_DIR

NAN5 = np.full(5, np.nan, dtype=np.float64)


def _fit_one(y_row: np.ndarray, t: np.ndarray):
    """Fit 5PL to one trace and return a flat tuple of features.

    Returns
    -------
    (params_5pl[5], r_squared, fit_ok, amp_Fm, amp_xs_xe, amp_fmax_fmin,
     amp_endpoints, xs, xe, xms, xp1, xp2, F_max, F_min, F0)
    """
    y = y_row.astype(np.float64)
    F0 = float(y[0])

    try:
        params, _ = fit_5p(t, y)
        params = np.asarray(params, dtype=np.float64)
        if not np.all(np.isfinite(params)):
            raise ValueError("non-finite params")

        F_fit = sigmoid_5p(t, *params)
        if not np.all(np.isfinite(F_fit)):
            raise ValueError("non-finite fitted curve")

        feats = extract_kinetic_parameters(t, y, tuple(params), threshold=0.1)

        Fm = float(params[0])
        amp_Fm = Fm
        amp_xs_xe = float(feats["y_xs"]) - float(feats["y_xe"])
        amp_fmax_fmin = float(F_fit.max()) - float(F_fit.min())
        amp_endpoints = float(y[-1]) - F0

        ssr = float(np.sum((y - F_fit) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ssr / (sst + 1e-10)

        return (
            params, r2, True,
            amp_Fm, amp_xs_xe, amp_fmax_fmin, amp_endpoints,
            float(feats["xs"]), float(feats["xe"]), float(feats["xms"]),
            float(feats["xp1"]), float(feats["xp2"]),
            float(F_fit.max()), float(F_fit.min()), F0,
        )
    except Exception:
        nan = np.nan
        return (
            NAN5.copy(), nan, False,
            nan, nan, nan, nan,
            nan, nan, nan, nan, nan,
            nan, nan, F0,
        )


def fit_dataset(dataset: str, jobs: int) -> Path:
    in_path = CLASSIFICATION_CACHE / f"{dataset}.npz"
    if not in_path.exists():
        raise FileNotFoundError(f"Input cache not found: {in_path}")

    print(f"Loading {in_path} ...")
    data = np.load(in_path)
    X = data["X"]
    N, W = X.shape
    print(f"  N={N} pixels, W={W} samples")

    t = np.arange(W, dtype=np.float64)

    print(f"Fitting 5PL with joblib (n_jobs={jobs}) ...")
    start = time.time()
    # joblib batches per-task call overhead is significant for ~40 ms jobs;
    # use larger batches to reduce IPC.
    results = Parallel(n_jobs=jobs, batch_size=64, verbose=5)(
        delayed(_fit_one)(X[i], t) for i in range(N)
    )
    elapsed = time.time() - start
    print(f"  done in {elapsed/60:.1f} min ({elapsed/N*1000:.1f} ms/pixel)")

    params_5pl = np.stack([r[0] for r in results]).astype(np.float64)
    r_squared = np.array([r[1] for r in results], dtype=np.float64)
    fit_ok = np.array([r[2] for r in results], dtype=bool)
    amp_Fm = np.array([r[3] for r in results], dtype=np.float64)
    amp_xs_xe = np.array([r[4] for r in results], dtype=np.float64)
    amp_fmax_fmin = np.array([r[5] for r in results], dtype=np.float64)
    amp_endpoints = np.array([r[6] for r in results], dtype=np.float64)
    xs = np.array([r[7] for r in results], dtype=np.float64)
    xe = np.array([r[8] for r in results], dtype=np.float64)
    xms = np.array([r[9] for r in results], dtype=np.float64)
    xp1 = np.array([r[10] for r in results], dtype=np.float64)
    xp2 = np.array([r[11] for r in results], dtype=np.float64)
    F_max = np.array([r[12] for r in results], dtype=np.float64)
    F_min = np.array([r[13] for r in results], dtype=np.float64)
    F0 = np.array([r[14] for r in results], dtype=np.float64)

    print(f"  fit_ok: {int(fit_ok.sum())}/{N} ({100*fit_ok.mean():.1f}%)")
    if fit_ok.any():
        r2_ok = r_squared[fit_ok]
        print(f"  R^2 on ok fits: median {np.median(r2_ok):.3f}, "
              f"p10 {np.quantile(r2_ok, 0.1):.3f}, p90 {np.quantile(r2_ok, 0.9):.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"fits_{dataset}.npz"
    np.savez_compressed(
        out_path,
        chip_id=data["chip_id"],
        well_id=data["well_id"],
        pixel_id=data["pixel_id"],
        y=data["y"],
        params_5pl=params_5pl,
        r_squared=r_squared,
        fit_ok=fit_ok,
        amp_Fm=amp_Fm,
        amp_xs_xe=amp_xs_xe,
        amp_fmax_fmin=amp_fmax_fmin,
        amp_endpoints=amp_endpoints,
        xs=xs, xe=xe, xms=xms, xp1=xp1, xp2=xp2,
        F_max=F_max, F_min=F_min, F0=F0,
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({sz_mb:.1f} MB)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset_per_pixel",
                   help="Name of the cache under classification/data/cache/ "
                        "without the .npz extension (default: dataset_per_pixel)")
    p.add_argument("--jobs", type=int, default=-1,
                   help="joblib n_jobs (-1 = all cores)")
    args = p.parse_args()
    fit_dataset(args.dataset, args.jobs)
