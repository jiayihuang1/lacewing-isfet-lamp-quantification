"""F-B extraction-rule ablation: apply many TTP-extraction rules to each F-B checkpoint.

For every F-B run dir with `predictions.npz` containing the `window_probs` extra:
  * Load window_probs, window, stride from the extras
  * Apply every rule in `extraction_rules.ALL_RULES`
  * Compute the full metric portfolio via the P7 harness
  * Append one row per (checkpoint × rule) to a SEPARATE scoreboard
    at Analysis/quantification/eval/fb_extraction_scoreboard.csv

The main scoreboard (`Analysis/quantification/eval/scoreboard.csv`) is left
untouched.  Only the winning combination will later be promoted to the main
scoreboard as an updated "best F-B" row.

Run::
    python -m lacewing.quantification.eval.fb_extraction_sweep

Cheap — pure Python + NumPy, no GPU, no training.  60 checkpoints × 16 rules
takes < 5 min locally.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.extraction_rules import ALL_RULES
from lacewing.quantification.eval.schema import (
    PredictionArrays,
    read_labels,
    read_predictions,
    validate_alignment,
)
from lacewing.quantification.eval.scoreboard import append_scoreboard_row, compute_all_metrics
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
METHODS_RESULTS = LACEWING_PKG_DIR / "quantification" / "methods" / "results"
DEFAULT_SCOREBOARD = LACEWING_PKG_DIR / "quantification" / "eval" / "fb_extraction_scoreboard.csv"


def _find_fb_run_dirs() -> list[tuple[str, str, Path]]:
    """Return list of (family_dir_name, seed_key, run_dir_path) for F-B runs.

    Family dir names look like `p1_fb_slide_cls_w{W}_stride{S}_spatA3`;
    seed dirs look like `seed{N}`.
    """
    out: list[tuple[str, str, Path]] = []
    if not METHODS_RESULTS.exists():
        return out
    for fam_dir in sorted(METHODS_RESULTS.iterdir()):
        if not fam_dir.is_dir():
            continue
        if not re.match(r"^p1_fb_slide_cls_w\d+_stride\d+_spatA3$", fam_dir.name):
            continue
        for seed_dir in sorted(fam_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            if not re.match(r"^seed\d+$", seed_dir.name):
                continue
            out.append((fam_dir.name, seed_dir.name, seed_dir))
    return out


def _method_id_for(family_dir: str, seed_key: str, rule_id: str) -> str:
    """Compact method_id: <family_dir_without_prefix>_<seed>_<rule>.

    e.g. `p1_fb_slide_cls_w60_stride5_spatA3_seed0_first_above_K3_thr0p5`
    """
    return f"{family_dir}_{seed_key}_{rule_id}"


def _read_window_probs(preds: PredictionArrays) -> tuple[np.ndarray, int, int] | None:
    """Extract per-window probs + window + stride from predictions.npz extras.

    Returns None if the extras don't include `window_probs`.
    """
    if "window_probs" not in preds.extras:
        return None
    probs = preds.extras["window_probs"].astype(np.float32)
    window = int(preds.extras.get("window", np.array([-1]))[0])
    stride = int(preds.extras.get("stride", np.array([-1]))[0])
    if window <= 0 or stride <= 0:
        return None
    return probs, window, stride


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoreboard", type=Path, default=DEFAULT_SCOREBOARD)
    parser.add_argument("--only_rule", type=str, default=None,
                        help="If set, apply only this rule id (for debugging one combination)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the scoreboard CSV before starting (fresh run)")
    args = parser.parse_args()

    if args.overwrite and args.scoreboard.exists():
        args.scoreboard.unlink()
        print(f"[reset] removed {args.scoreboard}")

    runs = _find_fb_run_dirs()
    if not runs:
        print(f"No F-B run dirs found under {METHODS_RESULTS}.  Nothing to do.")
        return
    print(f"Found {len(runs)} F-B run dirs")

    rules = ALL_RULES
    if args.only_rule:
        rules = [r for r in ALL_RULES if r.id == args.only_rule]
        if not rules:
            raise SystemExit(f"No rule with id={args.only_rule}.  See extraction_rules.ALL_RULES.")

    total_rows = 0
    total_skipped = 0

    for family_dir, seed_key, run_dir in runs:
        preds_path = run_dir / "predictions.npz"
        labels_path = run_dir / "labels.npz"
        if not preds_path.exists() or not labels_path.exists():
            print(f"[skip] {family_dir}/{seed_key}: missing predictions.npz or labels.npz")
            total_skipped += 1
            continue

        preds = read_predictions(preds_path)
        labels = read_labels(labels_path)

        window_pack = _read_window_probs(preds)
        if window_pack is None:
            print(f"[skip] {family_dir}/{seed_key}: no window_probs in extras (retrain with updated sliding_window_cls.py)")
            total_skipped += 1
            continue

        window_probs, window, stride = window_pack

        # Load the run's config for provenance in scoreboard row.
        config_path = run_dir / "config.json"
        run_config = json.loads(config_path.read_text()) if config_path.exists() else {}

        for rule in rules:
            ttp_pred = rule.fn(window_probs, window, stride)
            # Build a PredictionArrays with the rule's TTP and score through harness.
            preds_rule = PredictionArrays(
                ttp_pred_min=ttp_pred.astype(np.float32),
                extras={},
            )
            validate_alignment(preds_rule, labels)
            metrics = compute_all_metrics(preds_rule, labels, split="test")
            method_id = _method_id_for(family_dir, seed_key, rule.id)
            row_config = {
                **run_config,
                "extraction_rule_id": rule.id,
                "extraction_rule_label": rule.label,
                "extraction_rule_family": rule.family,
            }
            append_scoreboard_row(
                args.scoreboard,
                method_id=method_id,
                config_json=row_config,
                metrics=metrics,
            )
            total_rows += 1
        print(f"[ok] {family_dir}/{seed_key}: {len(rules)} rules → scoreboard")

    print(f"\nTotal: {total_rows} scoreboard rows, {total_skipped} skipped.")
    print(f"Scoreboard: {args.scoreboard}")


if __name__ == "__main__":
    main()
