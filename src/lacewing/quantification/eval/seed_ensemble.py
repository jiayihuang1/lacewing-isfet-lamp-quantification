"""Seed ensemble: average per-model outputs across seeds, then extract.

Reads saved predictions from N seed dirs of the SAME (framing, config).
Averages either extra__density (F-A) or extra__window_probs (F-B) across
seeds, then applies the chosen extraction rule to derive TTP.  Writes a
scoreboard row to seed_ensemble_scoreboard.csv (no NPZ artefacts kept
on disk — the ensembled predictions are recomputable from the source
seed dirs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.scoreboard import compute_all_metrics
from lacewing.quantification.eval.fa_extraction_sweep import (
    extract_ttp_hard_argmax,
)
from lacewing.quantification.methods._windowing import aggregate_to_ttp_first_positive


SAMPLES_PER_MIN = 15

HERE = Path(__file__).resolve().parent
METHODS_RESULTS_ROOT = HERE.parent / "methods" / "results"
OUT_CSV = HERE / "seed_ensemble_scoreboard.csv"


def average_arrays_across_seeds(arrays: list[np.ndarray]) -> np.ndarray:
    """Mean over the seed axis. Raises if shapes disagree."""
    ref_shape = arrays[0].shape
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape != ref_shape:
            raise ValueError(
                f"shape mismatch: seed 0 has {ref_shape}, seed {i} has {a.shape}"
            )
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _score_and_append(method_id: str, ttp_pred_min: np.ndarray,
                      labels_path: Path, cfg: dict) -> None:
    from lacewing.quantification.eval.schema import PredictionArrays, read_labels
    from lacewing.quantification.eval.scoreboard import append_scoreboard_row, compute_all_metrics
    preds = PredictionArrays(ttp_pred_min=ttp_pred_min.astype(np.float32), extras={})
    labels = read_labels(labels_path)
    metrics = compute_all_metrics(preds, labels, split="test")
    append_scoreboard_row(OUT_CSV, method_id=method_id, config_json=cfg, metrics=metrics)


def ensemble_fa(dirs: list[Path], extraction: str = "hard_argmax") -> None:
    densities = [np.load(d / "predictions.npz", allow_pickle=False)["extra__density"]
                 for d in dirs]
    avg = average_arrays_across_seeds(densities)
    if extraction != "hard_argmax":
        raise ValueError(f"only hard_argmax supported for F-A ensemble; got {extraction}")
    ttp_pred_min = extract_ttp_hard_argmax(avg)
    method_id = f"ensemble_fa_{dirs[0].parent.name}_{extraction}"
    cfg = {"framing": "fa", "extraction": extraction,
           "member_dirs": [str(d.relative_to(METHODS_RESULTS_ROOT)) for d in dirs]}
    _score_and_append(method_id, ttp_pred_min, dirs[0] / "labels.npz", cfg)
    print(f"[seed_ensemble] {method_id}: {len(dirs)} members averaged")


def ensemble_fb(dirs: list[Path], thr: float = 0.7, K: int = 3) -> None:
    all_probs = [np.load(d / "predictions.npz", allow_pickle=False)["extra__window_probs"]
                 for d in dirs]
    probs_meta = np.load(dirs[0] / "predictions.npz", allow_pickle=False)
    window = int(probs_meta["extra__window"][0])
    stride = int(probs_meta["extra__stride"][0])
    avg = average_arrays_across_seeds(all_probs)
    ttp_pred_min = aggregate_to_ttp_first_positive(
        avg, threshold=thr, k_consecutive=K,
        window=window, stride=stride, samples_per_min=SAMPLES_PER_MIN,
    )
    method_id = f"ensemble_fb_{dirs[0].parent.name}_K{K}_thr{thr:g}"
    cfg = {"framing": "fb", "K": K, "thr": thr, "window": window, "stride": stride,
           "member_dirs": [str(d.relative_to(METHODS_RESULTS_ROOT)) for d in dirs]}
    _score_and_append(method_id, ttp_pred_min, dirs[0] / "labels.npz", cfg)
    print(f"[seed_ensemble] {method_id}: {len(dirs)} members averaged")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirs", nargs="+", required=True,
                   help="Same-config seed dirs, relative to methods/results/")
    p.add_argument("--framing", choices=["fa", "fb"], required=True)
    p.add_argument("--extraction", default="hard_argmax",
                   help="For F-A: hard_argmax. For F-B: (K, thr) via --k / --thr.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--thr", type=float, default=0.7)
    args = p.parse_args()

    dirs = [METHODS_RESULTS_ROOT / d for d in args.dirs]
    for d in dirs:
        if not (d / "predictions.npz").exists():
            raise FileNotFoundError(f"missing predictions.npz in {d}")

    if args.framing == "fa":
        ensemble_fa(dirs, extraction=args.extraction)
    else:
        ensemble_fb(dirs, thr=args.thr, K=args.k)


if __name__ == "__main__":
    main()
