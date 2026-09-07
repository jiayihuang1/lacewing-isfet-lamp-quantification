"""F-A extraction-rule sweep — post-hoc on saved density outputs.

Applies 4 extraction rules to `extra__density` from F-A prediction dirs.
Writes a scoreboard CSV per (dir × seed × rule).  No re-training.

Run::
    python -m lacewing.quantification.eval.fa_extraction_sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.schema import PredictionArrays, read_labels, validate_alignment
from lacewing.quantification.eval.scoreboard import append_scoreboard_row, compute_all_metrics


SAMPLES_PER_MIN: int = 15
N_SAMPLES: int = 450

HERE = Path(__file__).resolve().parent
METHODS_RESULTS_ROOT = HERE.parent / "methods" / "results"
DEFAULT_SCOREBOARD = HERE / "fa_extraction_scoreboard.csv"


# ---------------------------------------------------------------------------
# Extraction rules — vectorised over pixels
# ---------------------------------------------------------------------------

def extract_ttp_hard_argmax(density: np.ndarray) -> np.ndarray:
    """density: (n_pixels, T) → ttp_min: (n_pixels,)

    Returns the time (in minutes) of the single peak sample.
    """
    idx = density.argmax(axis=1)
    return idx.astype(np.float32) / float(SAMPLES_PER_MIN)


def extract_ttp_soft_argmax_top5(density: np.ndarray) -> np.ndarray:
    """Weighted mean of top-5 density values → sample index → time in min.

    Matches the default F-A extraction behaviour used during training.
    """
    T = density.shape[1]
    top_idx = np.argpartition(density, kth=T - 5, axis=1)[:, -5:]   # (n, 5) unsorted
    top_vals = np.take_along_axis(density, top_idx, axis=1)          # (n, 5)
    w = top_vals / (top_vals.sum(axis=1, keepdims=True) + 1e-12)
    weighted_idx = (w * top_idx.astype(np.float32)).sum(axis=1)
    return weighted_idx.astype(np.float32) / float(SAMPLES_PER_MIN)


def extract_ttp_expected_value(density: np.ndarray) -> np.ndarray:
    """First moment of density: sum(t * density) / sum(density) → time in min."""
    t = np.arange(density.shape[1], dtype=np.float32)[None, :]      # (1, T)
    w = density / (density.sum(axis=1, keepdims=True) + 1e-12)
    idx = (w * t).sum(axis=1)
    return idx.astype(np.float32) / float(SAMPLES_PER_MIN)


def extract_ttp_first_moment_threshold(density: np.ndarray) -> np.ndarray:
    """Smallest t such that cumulative normalised density up to t > 0.5 (median)."""
    cum = np.cumsum(density, axis=1)
    total = cum[:, -1:] + 1e-12
    cdf = cum / total
    # First index where cdf > 0.5.
    idx = (cdf > 0.5).argmax(axis=1)
    return idx.astype(np.float32) / float(SAMPLES_PER_MIN)


RULES: dict[str, object] = {
    "hard_argmax":          extract_ttp_hard_argmax,
    "soft_argmax_top5":     extract_ttp_soft_argmax_top5,
    "expected_value":       extract_ttp_expected_value,
    "first_moment_thr0p5":  extract_ttp_first_moment_threshold,
}


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def _iter_fa_dirs() -> list[Path]:
    """Enumerate F-A prediction dirs that contain `extra__density` in predictions.npz."""
    dirs: list[Path] = []
    if not METHODS_RESULTS_ROOT.exists():
        return dirs
    for p in sorted(METHODS_RESULTS_ROOT.rglob("predictions.npz")):
        try:
            with np.load(p, allow_pickle=False) as npz:
                if "extra__density" in npz.files:
                    dirs.append(p.parent)
        except Exception:
            continue
    return dirs


def _score_one(dir_: Path, rule_name: str, scoreboard_csv: Path) -> None:
    """Score one (dir × rule) and append a row to the scoreboard CSV."""
    preds_path = dir_ / "predictions.npz"
    labels_path = dir_ / "labels.npz"
    if not preds_path.exists() or not labels_path.exists():
        print(f"  [skip] {dir_}: missing predictions.npz or labels.npz")
        return

    preds_raw = np.load(preds_path, allow_pickle=False)
    if "extra__density" not in preds_raw.files:
        return

    density = preds_raw["extra__density"]
    rule_fn = RULES[rule_name]
    ttp_pred_min = rule_fn(density).astype(np.float32)

    preds = PredictionArrays(ttp_pred_min=ttp_pred_min, extras={})
    labels = read_labels(labels_path)
    validate_alignment(preds, labels)

    metrics = compute_all_metrics(preds, labels, split="test")

    # Load config.json for provenance if it exists.
    config_path = dir_ / "config.json"
    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}

    method_id = f"{dir_.parent.name}_{dir_.name}_{rule_name}"
    cfg = {
        **run_config,
        "extraction_rule": rule_name,
        "dir": str(dir_.relative_to(METHODS_RESULTS_ROOT)),
    }
    append_scoreboard_row(scoreboard_csv, method_id=method_id, config_json=cfg, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc F-A extraction-rule sweep over saved density outputs."
    )
    parser.add_argument("--scoreboard", type=Path, default=DEFAULT_SCOREBOARD,
                        help="Output CSV path (default: fa_extraction_scoreboard.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the scoreboard CSV before starting (fresh run)")
    parser.add_argument("--only_rule", type=str, default=None,
                        help="Apply only this rule name (for debugging)")
    args = parser.parse_args()

    if args.overwrite and args.scoreboard.exists():
        args.scoreboard.unlink()
        print(f"[reset] removed {args.scoreboard}")

    dirs = _iter_fa_dirs()
    print(f"[fa_extraction_sweep] found {len(dirs)} F-A dirs with density")
    if not dirs:
        print("Nothing to do.")
        return

    rules = list(RULES.keys())
    if args.only_rule:
        if args.only_rule not in RULES:
            raise SystemExit(f"Unknown rule '{args.only_rule}'. Available: {list(RULES)}")
        rules = [args.only_rule]

    total_rows = 0
    for d in dirs:
        for rule_name in rules:
            _score_one(d, rule_name, args.scoreboard)
            total_rows += 1
        print(f"  {d.parent.name}/{d.name}: {len(rules)} rules → scoreboard")

    print(f"\nTotal: {total_rows} scoreboard rows written to {args.scoreboard}")


if __name__ == "__main__":
    main()
