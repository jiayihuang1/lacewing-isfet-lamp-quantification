"""Evaluate every exp6/exp7/exp8 checkpoint on the Multi test caches.

For each saved ``checkpoints/best.pt`` under
``Analysis/classification/results/{exp6,exp7,exp8}_*/runs/``:

  1. Read the run's ``config.yaml`` to get ``model``, ``cache``, ``seed``.
  2. Resolve the matching Multi cache by name (see ``MULTI_CACHE_FOR``).
  3. Load the cache and the model state.
  4. Run inference.
  5. Compute pixel + well-level metrics (no chip-level: out of scope).

Outputs one row per (experiment, run) into a single CSV.

Run::

    python -m lacewing.quantification.multi_data_shared.run_eval
    python -m lacewing.quantification.multi_data_shared.run_eval --device cpu
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from lacewing.classification import models as model_registry
from lacewing.classification.core import evaluate as eval_mod
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[2]
RESULTS_ROOT = LACEWING_PKG_DIR / "classification" / "results"
MULTI_CACHE_DIR = Path(__file__).resolve().parent / "cache"
OUTPUT_CSV = Path(__file__).resolve().parent / "output" / "eval_summary.csv"


# --------------------------------------------------------------------
# Mapping: training cache stem -> matching Multi cache stem
# --------------------------------------------------------------------
#
# Training caches are named like:
#   dataset_all                              (raw, exp6)
#   dataset_all_filt_abcd_ntcRaw             (pct, exp7)
#   dataset_all_filt_abcd_ntcRaw_madk1p5     (MAD k=1.5, exp8)
#
# Multi caches are named like:
#   multi_raw
#   multi_filt_abcd_ntcRaw
#   multi_filt_abcd_ntcRaw_madk1p5
#
# So the mapping is just "replace 'dataset_all' with 'multi'", then
# strip 'filt_' -> 'filt_' (no-op).  Implementation below.

def _multi_cache_stem_for(training_cache: str) -> str | None:
    """Translate a training cache stem to its Multi sibling."""
    stem = training_cache.strip()
    if stem == "dataset_all":
        return "multi_raw"
    if stem.startswith("dataset_all_filt_"):
        return "multi_filt_" + stem[len("dataset_all_filt_"):]
    if stem.startswith("dataset_final_filt_"):
        return "multi_filt_" + stem[len("dataset_final_filt_"):]
    return None


# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------

@dataclass
class RunRecord:
    experiment: str
    model: str
    seed: int
    training_cache: str
    multi_cache_stem: str
    run_dir: Path


def discover_runs(experiments: Iterable[str]) -> list[RunRecord]:
    """Walk RESULTS_ROOT/<exp>/runs/*/ and collect runnable RunRecords.

    Skips: runs without a checkpoints/best.pt, runs whose training
    cache doesn't have a matching Multi cache on disk.
    """
    records: list[RunRecord] = []
    for exp in experiments:
        exp_dir = RESULTS_ROOT / exp
        if not exp_dir.exists():
            continue
        for run_dir in sorted((exp_dir / "runs").iterdir()
                               if (exp_dir / "runs").exists() else []):
            if not run_dir.is_dir():
                continue
            ckpt = run_dir / "checkpoints" / "best.pt"
            cfg_path = run_dir / "config.yaml"
            if not ckpt.exists() or not cfg_path.exists():
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            training_cache = str(cfg.get("cache", "")).strip()
            multi_stem = _multi_cache_stem_for(training_cache)
            if multi_stem is None:
                continue
            multi_path = MULTI_CACHE_DIR / f"{multi_stem}.npz"
            if not multi_path.exists():
                continue
            records.append(RunRecord(
                experiment=exp,
                model=str(cfg.get("model", "")),
                seed=int(cfg.get("seed", -1)),
                training_cache=training_cache,
                multi_cache_stem=multi_stem,
                run_dir=run_dir,
            ))
    return records


def list_experiments_for(prefix: str) -> list[str]:
    return sorted(p.name for p in RESULTS_ROOT.iterdir()
                  if p.is_dir() and p.name.startswith(prefix))


# --------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------

# Per-process cache: don't reload the Multi cache for every run, just once.
_LOADED_CACHE: dict[str, dict] = {}


def _load_multi_cache(stem: str) -> dict:
    if stem in _LOADED_CACHE:
        return _LOADED_CACHE[stem]
    npz = np.load(MULTI_CACHE_DIR / f"{stem}.npz")
    payload = {
        "X":        npz["X"].astype(np.float32),
        "y":        npz["y"].astype(np.uint8),
        "chip_id":  npz["chip_id"],
        "well_id":  npz["well_id"],
        "pixel_id": npz["pixel_id"],
    }
    _LOADED_CACHE[stem] = payload
    return payload


def _build_model(model_name: str, input_len: int) -> torch.nn.Module:
    # Mirrors core.train: the model factory takes the input length
    # (raw features = (1, T) tuple).
    if model_registry.features_for(model_name) == "raw":
        return model_registry.build(model_name, (1, input_len))
    if model_registry.features_for(model_name) == "spectrogram":
        # We don't have spectrogram features for Multi yet; signal upstream.
        raise NotImplementedError(
            "spectrogram-features models (cnn2d_spec) need a spectrogram "
            "cache for Multi; not implemented in this eval"
        )
    raise ValueError(f"unknown feature kind for model {model_name}")


def _inference(model: torch.nn.Module, X: np.ndarray, device: str,
                batch_size: int = 4096) -> np.ndarray:
    """Forward pass over X, returning sigmoid(logit) per pixel."""
    model.eval().to(device)
    scores = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device).unsqueeze(1)
            logits = model(xb)
            scores[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return scores


def _eval_one_run(rec: RunRecord, device: str) -> dict | None:
    """Returns a result dict, or None if inference fails."""
    cache = _load_multi_cache(rec.multi_cache_stem)
    X, y = cache["X"], cache["y"]
    if X.shape[0] == 0:
        return None

    try:
        model = _build_model(rec.model, X.shape[1])
    except NotImplementedError as e:
        return {"skip_reason": str(e)}
    ckpt = torch.load(rec.run_dir / "checkpoints" / "best.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    is_autoencoder = (rec.model == "autoencoder")
    if is_autoencoder:
        # Match train.py: classify via reconstruction MSE.  We don't have
        # the calibration split here so threshold at the median train-set
        # error -> fallback: just take an arbitrary 0.5 sigmoid as score.
        # This is rough but autoencoder is one of 7 models; flag it.
        return {"skip_reason": "autoencoder eval needs calibration set "
                                "(not implemented for Multi)"}

    scores = _inference(model, X, device)
    pix = eval_mod.pixel_metrics(y, scores)
    keys, well_truth, well_pred = eval_mod.well_majority_vote(
        y, scores, cache["chip_id"], cache["well_id"]
    )
    well_acc = float((well_truth == well_pred).mean()) if len(well_truth) else float("nan")
    return {
        "pixel_acc":  pix.accuracy,
        "pixel_auc":  pix.auroc,
        "pixel_f1":   pix.f1,
        "pixel_prec": pix.precision,
        "pixel_rec":  pix.recall,
        "well_acc":   well_acc,
        "n_pixels":   int(X.shape[0]),
        "n_wells":    int(len(well_truth)),
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu",
                        help="Inference device (default cpu).")
    parser.add_argument("--experiments", default="exp6,exp7,exp8",
                        help="Comma-separated experiment prefixes "
                             "(default: exp6,exp7,exp8).")
    parser.add_argument("--out", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    experiments: list[str] = []
    for prefix in args.experiments.split(","):
        experiments.extend(list_experiments_for(prefix.strip()))
    experiments = sorted(set(experiments))
    print(f"Found {len(experiments)} experiment dirs matching prefixes "
          f"{args.experiments}: {experiments}", flush=True)

    records = discover_runs(experiments)
    print(f"Discovered {len(records)} runs with checkpoints + matching Multi caches",
          flush=True)

    # Open the CSV up front, write rows as we go, so a crash mid-run
    # doesn't lose the rows we've already computed.  Field set is fixed
    # by the schema below (matches what _eval_one_run returns).
    fieldnames = [
        "experiment", "model", "seed", "training_cache", "multi_cache",
        "pixel_acc", "pixel_auc", "pixel_f1", "pixel_prec", "pixel_rec",
        "well_acc", "n_pixels", "n_wells",
    ]
    fh = args.out.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    fh.flush()

    n_rows = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        try:
            result = _eval_one_run(rec, args.device)
        except Exception as e:
            print(f"[{i}/{len(records)}] FAIL {rec.experiment} {rec.model} "
                  f"seed={rec.seed}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue
        if result is None or "skip_reason" in (result or {}):
            reason = (result or {}).get("skip_reason", "no-data")
            print(f"[{i}/{len(records)}] SKIP {rec.experiment} {rec.model} "
                  f"seed={rec.seed}: {reason}", flush=True)
            n_skip += 1
            continue
        row = {
            "experiment":     rec.experiment,
            "model":          rec.model,
            "seed":           rec.seed,
            "training_cache": rec.training_cache,
            "multi_cache":    rec.multi_cache_stem,
            **result,
        }
        writer.writerow(row)
        fh.flush()
        n_rows += 1
        if i % 10 == 0 or i == len(records):
            rate = i / max(time.time() - t0, 1e-6)
            eta_min = (len(records) - i) / max(rate, 1e-6) / 60
            print(f"[{i}/{len(records)}] {rec.experiment} {rec.model} "
                  f"seed={rec.seed}: pixel={result['pixel_acc']:.3f} "
                  f"well={result['well_acc']:.3f}  | "
                  f"rate={rate:.2f}/s  eta={eta_min:.1f} min  "
                  f"(rows={n_rows} skip={n_skip} fail={n_fail})",
                  flush=True)

    fh.close()
    elapsed = time.time() - t0
    print(f"\nEvaluated {n_rows} runs in {elapsed/60:.1f} min "
          f"(skip={n_skip}, fail={n_fail})", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
