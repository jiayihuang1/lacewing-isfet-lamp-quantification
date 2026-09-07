"""F-D deeper failure-analysis diagnostics (post-hoc, no HPC / no training).

Four diagnostics run entirely off EXISTING F-D checkpoints and cached
test-time predictions (test_time_predictions.npz + labels.npz per seed):

  D1 — Test-time boundary error vs internal-val boundary error.
       Quantifies the generalisation gap: is the model's "rising" boundary
       error much worse on held-out SD test wells than on the internal
       (manual-label) validation split reported in fd_boundary_scoreboard.csv?

  D2 — Per-well breakdown heatmap (well x seed), signed + absolute error
       in minutes, saved as a PNG plus the underlying CSV.

  D4 — Calibration scatter: predicted vs true TTP for all 5014 test pixels
       per seed, coloured by well, with per-well medians overlaid.

  D10 — Time-flip sanity check: does the model's rising-boundary
       prediction depend on the input trace, or has it memorised a fixed
       sample position (positional-encoding pathology)?

All four operate on last.pt (the checkpoint used for cached test-time
inference) unless a different ckpt is explicitly requested.

Usage
-----
    python -m lacewing.quantification.eval.fd_deeper_diagnostics
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from lacewing.quantification.methods.unet_segmentation import (
    CLS_AMP as _CLS_RISING,   # class 2 in manual schema = "rising"
    SAMPLES_PER_MIN,
    N_SAMPLES,
    build_model,
)
from lacewing.quantification.regression_shared.data import dataset as reg_ds

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE.parent / "methods" / "results"
FD_DIR = RESULTS_ROOT / "p1_fd_manual4_spatA3"
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

BOUNDARY_SCOREBOARD = HERE / "fd_boundary_scoreboard.csv"

OUT_CSV_PER_WELL = HERE / "fd_per_well_test_errors.csv"
FIGS_DIR = Path("Meetings/Week_15_22072026/figs")
OUT_FIG_HEATMAP = FIGS_DIR / "fig_fd_per_well_heatmap.png"
OUT_FIG_CALIBRATION = FIGS_DIR / "fig_fd_calibration_scatter.png"

SEEDS = [0, 1, 2]
COLLAPSE_POINT_MIN = 21.0  # observed collapse point (~seed1/seed2 predict ~21 min everywhere)


# ---------------------------------------------------------------------------
# Shared loading helpers
# ---------------------------------------------------------------------------

def _seed_dir(seed: int) -> Path:
    return FD_DIR / f"seed{seed}"


def _load_test_time_argmax(seed: int) -> np.ndarray:
    """(5014, 450) int8 argmax per timestep, from cached test_time_predictions.npz."""
    d = np.load(_seed_dir(seed) / "test_time_predictions.npz")
    return d["argmax"]


def _load_labels(seed: int) -> dict:
    d = np.load(_seed_dir(seed) / "labels.npz", allow_pickle=False)
    return {
        "ttp_true_min": d["ttp_true_min"],
        "chip_id": d["chip_id"],
        "well_id": d["well_id"],
        "log10_concentration": d["log10_concentration"],
        "split": d["split"],
    }


def _first_rising_index(argmax_2d: np.ndarray, rising_cls: int = _CLS_RISING) -> np.ndarray:
    """(N, T) argmax -> (N,) first index where class==rising_cls, else -1."""
    mask = argmax_2d == rising_cls
    has_rising = mask.any(axis=1)
    first = np.where(has_rising, mask.argmax(axis=1), -1)
    return first.astype(np.int64)


# ---------------------------------------------------------------------------
# D1 — Test-time boundary error
# ---------------------------------------------------------------------------

def compute_test_boundary_errors(seed: int, ckpt: str = "last.pt") -> pd.DataFrame:
    """Per-test-pixel signed boundary ("rising" onset) error, in samples.

    Uses the cached test_time_predictions.npz argmax (produced by last.pt
    at test-time, per the F-D training script) rather than re-running the
    model, since the cache already reflects `ckpt`="last.pt" inference.

    Columns: pixel_idx, well_id, chip_id, log10_conc, first_rising_pred,
    first_rising_true, dr_error_samples, dr_error_signed_samples.
    """
    if ckpt != "last.pt":
        raise ValueError(
            f"only ckpt='last.pt' is supported (test_time_predictions.npz was "
            f"generated from last.pt); got {ckpt!r}"
        )

    argmax = _load_test_time_argmax(seed)          # (5014, 450)
    labels = _load_labels(seed)

    first_rising_pred = _first_rising_index(argmax, rising_cls=_CLS_RISING)
    pseudo_true_rising = np.round(labels["ttp_true_min"] * SAMPLES_PER_MIN).astype(np.int64)

    dr_error_signed = first_rising_pred - pseudo_true_rising
    dr_error_abs = np.abs(dr_error_signed)

    n = len(first_rising_pred)
    df = pd.DataFrame({
        "pixel_idx": np.arange(n),
        "well_id": labels["well_id"],
        "chip_id": labels["chip_id"],
        "log10_conc": labels["log10_concentration"],
        "first_rising_pred": first_rising_pred,
        "first_rising_true": pseudo_true_rising,
        "dr_error_samples": dr_error_abs,
        "dr_error_signed_samples": dr_error_signed,
    })
    return df


def _aggregate_test_boundary_errors(df: pd.DataFrame) -> dict:
    signed = df["dr_error_signed_samples"].to_numpy(dtype=np.float64)
    return {
        "median": float(np.median(signed)),
        "mean": float(np.mean(signed)),
        "p10": float(np.percentile(signed, 10)),
        "p90": float(np.percentile(signed, 90)),
        "std": float(np.std(signed)),
    }


def _per_well_dr_error(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, sub in df.groupby("well_id"):
        signed = sub["dr_error_signed_samples"].to_numpy(dtype=np.float64)
        rows.append({
            "well_id": int(well),
            "median_dr_error": float(np.median(signed)),
            "std_dr_error": float(np.std(signed)),
            "n_pixels": len(sub),
        })
    return pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)


def _read_internal_val_scoreboard() -> pd.DataFrame:
    if not BOUNDARY_SCOREBOARD.exists():
        return pd.DataFrame()
    return pd.read_csv(BOUNDARY_SCOREBOARD)


def run_d1(seeds: list[int]) -> dict:
    """Run D1 for all seeds; print gap vs internal-val, return raw results."""
    scoreboard = _read_internal_val_scoreboard()
    results = {}
    print("\n=== D1: Test-time boundary error vs internal-val ===")
    for seed in seeds:
        df = compute_test_boundary_errors(seed, ckpt="last.pt")
        agg = _aggregate_test_boundary_errors(df)
        per_well = _per_well_dr_error(df)
        results[seed] = {"per_pixel": df, "global": agg, "per_well": per_well}

        internal_val_median = None
        if not scoreboard.empty:
            method_id = f"p1_fd_manual4_spatA3_seed{seed}_last"
            row = scoreboard[scoreboard["method_id"] == method_id]
            if len(row):
                internal_val_median = float(row.iloc[0]["median_dr_error"])

        gap_str = (
            f"gap={agg['median'] - internal_val_median:+.1f} samples"
            if internal_val_median is not None else "gap=n/a (no scoreboard row)"
        )
        print(
            f"  seed {seed}: internal_val_median_dr_error={internal_val_median} "
            f"test_median_dr_error_signed={agg['median']:.1f} "
            f"(mean={agg['mean']:.1f}, p10={agg['p10']:.1f}, p90={agg['p90']:.1f}, std={agg['std']:.1f})  "
            f"{gap_str}"
        )
        print(f"    per-well median dr_error (signed samples): "
              + ", ".join(f"w{int(r.well_id)}={r.median_dr_error:.1f}(std={r.std_dr_error:.1f})"
                           for r in per_well.itertuples()))
    return results


# ---------------------------------------------------------------------------
# D2 — Per-well breakdown
# ---------------------------------------------------------------------------

def compute_per_well_errors(seed: int, ckpt: str = "last.pt") -> pd.DataFrame:
    """One row per SD well: n_pixels, true TTP, median predicted TTP, signed/abs error (min).

    Columns: seed, well, log10_conc, n_pixels, true_ttp_min,
    median_pred_ttp_min, signed_error_min, mae_min.
    """
    if ckpt != "last.pt":
        raise ValueError("only ckpt='last.pt' is supported for test-time diagnostics")

    argmax = _load_test_time_argmax(seed)
    labels = _load_labels(seed)

    first_rising_pred = _first_rising_index(argmax, rising_cls=_CLS_RISING)
    # Fallback for "no rising found": use N_SAMPLES - 1, matching _predict_ttp's convention.
    pred_ttp_min = np.where(
        first_rising_pred >= 0, first_rising_pred, N_SAMPLES - 1
    ).astype(np.float64) / SAMPLES_PER_MIN

    rows = []
    wells = np.unique(labels["well_id"])
    for well in sorted(wells.tolist()):
        mask = labels["well_id"] == well
        true_ttp = labels["ttp_true_min"][mask]
        pred_ttp = pred_ttp_min[mask]
        log10_conc = float(np.unique(labels["log10_concentration"][mask])[0])
        true_ttp_min = float(np.unique(true_ttp)[0]) if len(np.unique(true_ttp)) == 1 else float(np.median(true_ttp))
        median_pred = float(np.median(pred_ttp))
        signed_err = median_pred - true_ttp_min
        mae = float(np.median(np.abs(pred_ttp - true_ttp)))
        rows.append({
            "seed": seed,
            "well": int(well),
            "log10_conc": log10_conc,
            "n_pixels": int(mask.sum()),
            "true_ttp_min": true_ttp_min,
            "median_pred_ttp_min": median_pred,
            "signed_error_min": signed_err,
            "mae_min": mae,
        })
    return pd.DataFrame(rows)


def run_d2(seeds: list[int]) -> pd.DataFrame:
    print("\n=== D2: Per-well breakdown ===")
    all_rows = []
    for seed in seeds:
        df = compute_per_well_errors(seed, ckpt="last.pt")
        all_rows.append(df)
        for r in df.itertuples():
            print(
                f"  seed {seed} well {r.well} (log10={r.log10_conc:.0f}): "
                f"n={r.n_pixels} true={r.true_ttp_min:.2f}min "
                f"pred_median={r.median_pred_ttp_min:.2f}min "
                f"signed_err={r.signed_error_min:+.2f}min mae={r.mae_min:.2f}min"
            )
    combined = pd.concat(all_rows, ignore_index=True)
    OUT_CSV_PER_WELL.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV_PER_WELL, index=False)
    print(f"  [ok] wrote {OUT_CSV_PER_WELL}")
    return combined


def make_per_well_heatmap(seeds: list[int]) -> None:
    """(well x seed) heatmap of signed error (min), cell text = median MAE."""
    if plt is None:  # pragma: no cover
        print("[warn] matplotlib unavailable — skipping heatmap")
        return

    dfs = [compute_per_well_errors(seed, ckpt="last.pt") for seed in seeds]
    combined = pd.concat(dfs, ignore_index=True)

    wells = sorted(combined["well"].unique().tolist())
    n_wells = len(wells)
    n_seeds = len(seeds)

    signed_grid = np.full((n_wells, n_seeds), np.nan)
    mae_grid = np.full((n_wells, n_seeds), np.nan)
    for i, well in enumerate(wells):
        for j, seed in enumerate(seeds):
            row = combined[(combined["well"] == well) & (combined["seed"] == seed)]
            if len(row):
                signed_grid[i, j] = row.iloc[0]["signed_error_min"]
                mae_grid[i, j] = row.iloc[0]["mae_min"]

    vmax = np.nanmax(np.abs(signed_grid)) if np.any(~np.isnan(signed_grid)) else 1.0
    fig, ax = plt.subplots(figsize=(2.2 * n_seeds + 2, 1.3 * n_wells + 2))
    im = ax.imshow(signed_grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    well_conc = {int(r.well): r.log10_conc for r in combined.itertuples()}
    ax.set_yticks(range(n_wells))
    ax.set_yticklabels([f"well {w} (1e{well_conc.get(w, float('nan')):.0f})" for w in wells])
    ax.set_title(
        "F-D per-well test-time error (last.pt)\n"
        "color = signed error [blue=early, red=late], text = median MAE (min)"
    )

    for i in range(n_wells):
        for j in range(n_seeds):
            if not np.isnan(mae_grid[i, j]):
                ax.text(
                    j, i, f"{mae_grid[i, j]:.1f}",
                    ha="center", va="center",
                    color="black", fontsize=10,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("signed error (min): predicted - true")

    fig.tight_layout()
    OUT_FIG_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_HEATMAP, dpi=150)
    plt.close(fig)
    print(f"  [ok] wrote {OUT_FIG_HEATMAP}")


# ---------------------------------------------------------------------------
# D4 — Calibration scatter
# ---------------------------------------------------------------------------

def make_calibration_scatter(seeds: list[int]) -> None:
    """3-panel (one per seed) scatter of pred vs true TTP for all test pixels."""
    if plt is None:  # pragma: no cover
        print("[warn] matplotlib unavailable — skipping calibration scatter")
        return

    fig, axes = plt.subplots(1, len(seeds), figsize=(5.5 * len(seeds), 5), sharex=True, sharey=True)
    if len(seeds) == 1:
        axes = [axes]

    well_colors = plt.get_cmap("tab10")

    for ax, seed in zip(axes, seeds):
        argmax = _load_test_time_argmax(seed)
        labels = _load_labels(seed)
        first_rising_pred = _first_rising_index(argmax, rising_cls=_CLS_RISING)
        pred_ttp_min = np.where(
            first_rising_pred >= 0, first_rising_pred, N_SAMPLES - 1
        ).astype(np.float64) / SAMPLES_PER_MIN
        true_ttp_min = labels["ttp_true_min"].astype(np.float64)
        well_id = labels["well_id"]

        wells = sorted(np.unique(well_id).tolist())
        for k, well in enumerate(wells):
            mask = well_id == well
            ax.scatter(
                true_ttp_min[mask], pred_ttp_min[mask],
                s=8, alpha=0.35, color=well_colors(k % 10),
                label=f"well {well}", linewidths=0,
            )

        # y = x diagonal
        lims = [
            min(true_ttp_min.min(), pred_ttp_min.min()) - 1,
            max(true_ttp_min.max(), pred_ttp_min.max()) + 1,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x (perfect)")

        # collapse-point reference line
        ax.axhline(COLLAPSE_POINT_MIN, color="grey", linestyle=":", linewidth=1,
                   label=f"~{COLLAPSE_POINT_MIN:.0f} min collapse point")

        # per-well medians with error bars (std across pixels in well)
        for k, well in enumerate(wells):
            mask = well_id == well
            med_true = np.median(true_ttp_min[mask])
            med_pred = np.median(pred_ttp_min[mask])
            std_pred = np.std(pred_ttp_min[mask])
            ax.errorbar(
                med_true, med_pred, yerr=std_pred,
                fmt="D", markersize=10, color=well_colors(k % 10),
                markeredgecolor="black", markeredgewidth=1.2,
                capsize=4, zorder=5,
            )

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("qLAMP true TTP (min)")
        ax.set_title(f"seed {seed}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("predicted TTP (min)")
    axes[-1].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("F-D calibration: predicted vs true TTP, all 5014 test pixels (last.pt)")
    fig.tight_layout()

    OUT_FIG_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_CALIBRATION, dpi=150)
    plt.close(fig)
    print(f"  [ok] wrote {OUT_FIG_CALIBRATION}")


# ---------------------------------------------------------------------------
# D10 — Time-flip sanity check
# ---------------------------------------------------------------------------

def _predict_first_rising_for_trace(model: torch.nn.Module, x_1d: np.ndarray, device: str = "cpu") -> int:
    """Run model on a single (T,) trace, return first index with argmax==rising, else -1."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(x_1d).astype(np.float32))[None, None, :].to(device)
        logits = model(x)  # (1, 4, T)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    idxs = np.where(pred == _CLS_RISING)[0]
    return int(idxs[0]) if len(idxs) else -1


def time_flip_test(seed: int, n_samples: int = 20) -> dict:
    """Time-flip sanity check for one F-D seed's last.pt model.

    For a random sample of `n_samples` test pixels, predicts the first
    "rising" index on the trace both normally and time-reversed.

    If the model is genuinely input-dependent, flipped predictions should
    be roughly `N_SAMPLES - normal` (a temporal mirror). If the model has
    memorised a fixed answer position (positional pathology), flipped
    predictions will stay near the same value as normal.

    Returns
    -------
    dict with keys: seed, n_samples, normal (list), flipped (list),
    normal_range, flipped_range,
    correlation_normal_vs_flipped_complement, median_abs_diff.
    """
    device = "cpu"
    ckpt_path = _seed_dir(seed) / "checkpoints" / "last.pt"
    model = build_model().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    arr = reg_ds.make_split(CACHE_STEM, seed=seed, val_n_chips=1, normalise_y=False)
    X_te = arr.X_te

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X_te), size=min(n_samples, len(X_te)), replace=False)

    normal_preds = []
    flipped_preds = []
    for i in idxs:
        x = X_te[i]
        normal_preds.append(_predict_first_rising_for_trace(model, x, device))
        flipped_preds.append(_predict_first_rising_for_trace(model, x[::-1], device))

    normal_arr = np.array(normal_preds, dtype=np.float64)
    flipped_arr = np.array(flipped_preds, dtype=np.float64)
    flipped_complement = (N_SAMPLES - 1) - flipped_arr

    # Correlation between normal and (N_SAMPLES-1 - flipped).
    if np.std(normal_arr) < 1e-9 or np.std(flipped_complement) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(normal_arr, flipped_complement)[0, 1])

    median_abs_diff = float(np.median(np.abs(normal_arr - flipped_arr)))

    return {
        "seed": seed,
        "n_samples": len(idxs),
        "pixel_indices": idxs.tolist(),
        "normal": normal_preds,
        "flipped": flipped_preds,
        "normal_range": [int(normal_arr.min()), int(normal_arr.max())],
        "flipped_range": [int(flipped_arr.min()), int(flipped_arr.max())],
        "correlation_normal_vs_flipped_complement": corr,
        "median_abs_diff": median_abs_diff,
    }


def run_d10(seeds: list[int], n_samples: int = 20) -> dict:
    print("\n=== D10: Time-flip sanity check ===")
    print("Time-flip test:")
    results = {}
    for seed in seeds:
        res = time_flip_test(seed, n_samples=n_samples)
        results[seed] = res
        print(
            f"  seed {seed}: normal_range={res['normal_range']}, "
            f"flipped_range={res['flipped_range']}, "
            f"corr(normal, 450-flipped)={res['correlation_normal_vs_flipped_complement']:.3f}, "
            f"median|Δ|={res['median_abs_diff']:.1f} samples"
        )
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    d1_results = run_d1(SEEDS)
    d2_df = run_d2(SEEDS)
    make_per_well_heatmap(SEEDS)
    make_calibration_scatter(SEEDS)
    d10_results = run_d10(SEEDS, n_samples=20)

    print("\n=== Summary ===")
    print(f"  D1: test-time boundary errors computed for seeds {SEEDS}")
    print(f"  D2: per-well CSV written to {OUT_CSV_PER_WELL}, heatmap to {OUT_FIG_HEATMAP}")
    print(f"  D4: calibration scatter written to {OUT_FIG_CALIBRATION}")
    print(f"  D10: time-flip results computed for seeds {SEEDS}")


if __name__ == "__main__":
    main()
