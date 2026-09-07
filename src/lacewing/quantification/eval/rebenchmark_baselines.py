"""Rebenchmark rule-based + F4 baselines through the P7 harness.

Reads each method's canonical predictions.npz + labels.npz, computes the
full metric portfolio, appends a row to Analysis/quantification/eval/scoreboard.csv.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

from lacewing.quantification.eval.schema import read_predictions, read_labels, validate_alignment
from lacewing.quantification.eval.scoreboard import compute_all_metrics, append_scoreboard_row
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


BASELINES: list[tuple[str, str]] = [
    ("rule_ttp",   "Analysis/quantification/results/ttp"),
    ("rule_sdm",   "Analysis/quantification/results/sdm_cy0/sdm"),
    ("rule_cy0",   "Analysis/quantification/results/sdm_cy0/cy0"),
    # F4 wMSE (best-fixed, all seeds available on disk):
    ("f4_wmse_cnn1d_spatA3_seed0", "Analysis/quantification/methods/results/f6_pdf_cnn1d_spatA3_wmse/seed0_sigma5_wmse0.1"),
    ("f4_wmse_cnn1d_spatA3_seed1", "Analysis/quantification/methods/results/f6_pdf_cnn1d_spatA3_wmse/seed1_sigma5_wmse0.1"),
    ("f4_wmse_cnn1d_spatA3_seed2", "Analysis/quantification/methods/results/f6_pdf_cnn1d_spatA3_wmse/seed2_sigma5_wmse0.1"),
    ("f4_wmse_cnn1d_spatA3_seed3", "Analysis/quantification/methods/results/f6_pdf_cnn1d_spatA3_wmse/seed3_sigma5_wmse0.1"),
    ("f4_wmse_cnn1d_spatA3_seed4", "Analysis/quantification/methods/results/f6_pdf_cnn1d_spatA3_wmse/seed4_sigma5_wmse0.1"),
]

# P1 framing sweep — F-B / F-C / F-D / F-E all rebenchmarked.
#
# F-C (sliding-window regression) was dropped from the OLD-metric scoreboard because
# its median Spearman r = -0.3 across 5 seeds (anti-correlated calibration
# curve) showed the framing is broken.  Under the new metric portfolio
# (2026-07-09) F-C is re-included for audit-trail completeness — it will
# fail the ρ ≤ -0.9 filter but its numbers are honestly reported alongside
# the working framings.
#
# method_id uses the full run-dir name + _seedN so downstream consumers
# (viewer, notebook) can reverse-map the method_id back to the dir cleanly.
P1_DIRS = [
    "p1_fb_slide_cls_w60_stride5_spatA3",
    "p1_fc_slide_reg_w60_stride5_spatA3",
    "p1_fd_unet_seg_ow30_spatA3",
    "p1_fe_bigru_full_spatA3",
]
for dir_name in P1_DIRS:
    for seed in range(5):
        BASELINES.append((
            f"{dir_name}_seed{seed}",
            f"Analysis/quantification/methods/results/{dir_name}/seed{seed}",
        ))

# P2 backbone sweep on F-E's framing (7 backbones × 5 seeds = 35 rows).
# Same aggregation convention as P1 — method_id = full run-dir name + _seedN.
P2_BACKBONES = [
    "bigru",              # F-E reference
    "gru",                # unidirectional counterpart
    "ann",                # diagnostic no-context MLP
    "cnn_gru_par",        # hybrid CNN + BiGRU
    "transformer_patch",  # PatchTST-style attention
    "unet",               # 1D U-Net (paper 19)
    "tcn",                # TCN (paper 20)
]
for backbone in P2_BACKBONES:
    dir_name = f"p2_{backbone}_full_spatA3"
    for seed in range(5):
        BASELINES.append((
            f"{dir_name}_seed{seed}",
            f"Analysis/quantification/methods/results/{dir_name}/seed{seed}",
        ))

# F-B confound-check: F-B framing on selected backbones at W=60 stride=5.
# Only run once the P2 winner is known — see docs/superpowers/plans/
# 2026-07-06-decision-log.md.  Kept in the rebenchmark list so that when
# any confound-check dir appears on disk, its rows are automatically added.
CONFOUND_BACKBONES = [
    "ann", "gru", "bigru", "cnn_gru_par", "transformer_patch", "unet", "tcn",
]
for backbone in CONFOUND_BACKBONES:
    dir_name = f"fb_confound_{backbone}_w60_stride5_spatA3"
    for seed in range(5):
        BASELINES.append((
            f"{dir_name}_seed{seed}",
            f"Analysis/quantification/methods/results/{dir_name}/seed{seed}",
        ))
    dir_name = f"fd_confound_{backbone}_spatA3"
    for seed in range(5):
        BASELINES.append((
            f"{dir_name}_seed{seed}",
            f"Analysis/quantification/methods/results/{dir_name}/seed{seed}",
        ))

# W15-W16 auto-discovery — P3/P4/F-A sigma-schedule/F-D backbone-swap output
# dirs are produced by HPC array jobs with data-dependent names (alpha values,
# masked/contrastive x backbone x frozen/fullft x fa/fb combos, schedule tags).
# Rather than hardcode every combo, glob the results/ tree for seed dirs that
# have both predictions.npz + labels.npz and append them automatically.
RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results"


def _discover_seed_dirs(pattern: str) -> list[tuple[str, str]]:
    """Glob RESULTS_ROOT for `pattern`, then walk each match's seed* subdirs.

    Returns (method_id, run_dir) tuples for seed dirs that have both
    predictions.npz and labels.npz on disk. method_id = f"{dir_name}_{seed_dir_name}".
    """
    found: list[tuple[str, str]] = []
    for run_dir in sorted(RESULTS_ROOT.glob(pattern)):
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.glob("seed*")):
            if (seed_dir / "predictions.npz").exists() and (seed_dir / "labels.npz").exists():
                method_id = f"{run_dir.name}_{seed_dir.name}"
                found.append((method_id, str(seed_dir)))
    return found


# P3: existing 4-arch sweep + W16 refinements ({a1s, a2p, a5, a3n, a4w}).
# Trailing "*" catches _ptmasked/_ptcontrastive (combo) + _ptcls (a2p) tags.
BASELINES.extend(_discover_seed_dirs("p3_a*_alpha*_unet_w120_stride10*"))

# P4: masked/contrastive SSL pretraining x {unet, gru, cnn_gru_par} x
# {frozen, fullft} x {fa, fb} framing.
BASELINES.extend(_discover_seed_dirs("p4_masked_*"))
BASELINES.extend(_discover_seed_dirs("p4_contrastive_*"))

# F-A sigma-schedule curriculum runs (Task 12) — dir name has seed{N}_sched{cosine|adaptive}.
BASELINES.extend(_discover_seed_dirs("p2_fa_curriculum_bb_unet_wmse"))

# F-B sliding-window classifier at window=120 stride=10 (RQ2 winner) —
# includes the backbone sweep (bb_unet, bb_gru, bb_bigru, bb_tcn, bb_ann,
# bb_transformer_patch) plus soft-target and dropout variants.
BASELINES.extend(_discover_seed_dirs("p1_fb_slide_cls_w120_stride10_spatA3*"))

# F-D backbone swap (GRU / CNN-GRU-par in place of the manual-4 U-Net backbone).
BASELINES.extend(_discover_seed_dirs("p1_fd_manual4_spatA3_bb_gru"))
BASELINES.extend(_discover_seed_dirs("p1_fd_manual4_spatA3_bb_cnn_gru_par"))

SCOREBOARD_CSV = LACEWING_PKG_DIR / "quantification" / "eval" / "scoreboard.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoreboard", type=Path, default=SCOREBOARD_CSV)
    args = parser.parse_args()

    for method_id, run_dir in BASELINES:
        run_dir = Path(run_dir)
        preds_path = run_dir / "predictions.npz"
        labels_path = run_dir / "labels.npz"
        if not preds_path.exists() or not labels_path.exists():
            print(f"[skip] {method_id}: missing {preds_path} or {labels_path}")
            continue
        preds = read_predictions(preds_path)
        labels = read_labels(labels_path)
        validate_alignment(preds, labels)
        metrics = compute_all_metrics(preds, labels, split="test")
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        append_scoreboard_row(args.scoreboard, method_id=method_id, config_json=config, metrics=metrics)
        flag = "✓" if metrics["passes_spearman_filter"] else "✗"
        print(f"[ok]   {method_id}: "
              f"MAE={metrics['per_well_mae_min']:.2f}, "
              f"|Δslope|={metrics['slope_proximity_min_per_decade']:.2f} "
              f"(raw {metrics['slope_min_per_decade']:.2f}), "
              f"r²={metrics['per_well_r2']:.3f}, "
              f"CoV={metrics['within_well_cov']:.3f}, "
              f"ρ={metrics['spearman_r_well_means']:.2f} {flag}")

    print(f"\nScoreboard written to {args.scoreboard}")


if __name__ == "__main__":
    main()
