"""Validate ALL systematic-TTP variants against the manual UTI labels.

For every UTI well with (manual chip TTP, plate TTP), run each variant and
compare its output to the human's pick.

Outputs:
  Analysis/quantification/output/systematic_vs_manual_uti.csv
      one row per (well, variant) with manual, plate, systematic, delta_min
  Analysis/quantification/output/fig_systematic_vs_manual_uti.png
      per-variant scatter panels + per-variant MAE bar summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lacewing.quantification.systematic_ttp import compute_all_variants, VARIANTS


PROJECT_ROOT = DATA_ROOT  # was: parents[2]
UTI_OUT     = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output"
MANUAL_JSON = UTI_OUT / "manual_chip_ttps.json"
CHIP_STAGES = [
    UTI_OUT / "uti_260630_EC_SD_stages.json",
    UTI_OUT / "uti_260710_EC_SD_stages.json",
    UTI_OUT / "uti_260728_EC_SD_stages.json",
]
OUT_DIR = LACEWING_PKG_DIR / "quantification" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "systematic_vs_manual_uti.csv"
FIG_PATH = OUT_DIR / "fig_systematic_vs_manual_uti.png"

CHIP_LABELS = {
    "uti_260630_EC_SD": "06-30 SD",
    "uti_260710_EC_SD": "07-10 SD",
    "uti_260728_EC_SD": "07-28 SD",
}
CHIP_COLOURS = {
    "uti_260630_EC_SD": "#1976D2",
    "uti_260710_EC_SD": "#43A047",
    "uti_260728_EC_SD": "#E53935",
}

VARIANT_LABELS = {
    "closest_peak_d1":       "V1 · closest d1 peak",
    "zero_cross_d2":         "V4 · d2 zero-cross (+→−)",
    "cy0_after_plate":       "V6 · cy0 after plate",
    "thr_deriv_after_plate": "V7 · thr_deriv after plate",
}


def _load_manual_labels() -> dict:
    return json.loads(MANUAL_JSON.read_text())["labels"]


def _run_one(chip_tag: str, well: dict, manual: dict) -> list[dict]:
    key = f"{chip_tag}__well{well['well']}"
    manual_lbl = manual.get(key)
    if manual_lbl is None or manual_lbl.get("is_no_amp") or manual_lbl.get("chip_ttp_min") is None:
        return [{"chip_tag": chip_tag, "well": well["well"], "label": well["label"],
                 "log10_conc": well.get("log10_conc"),
                 "manual_ttp_min": None, "plate_ttp_min": None,
                 "variant": v, "systematic_ttp_min": None, "delta_min": None,
                 "reason": "no manual label / no-amp"} for v in VARIANTS]
    plate_ttp = manual_lbl.get("plate_ttp_mean_min")
    if plate_ttp is None:
        return [{"chip_tag": chip_tag, "well": well["well"], "label": well["label"],
                 "log10_conc": well.get("log10_conc"),
                 "manual_ttp_min": manual_lbl.get("chip_ttp_min"),
                 "plate_ttp_min": None,
                 "variant": v, "systematic_ttp_min": None, "delta_min": None,
                 "reason": "no plate TTP"} for v in VARIANTS]

    t = np.asarray(well["time_min"], dtype=np.float64)
    sig = np.asarray(well["smoothed_signal"], dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(sig)
    t = t[mask]; sig = sig[mask]

    results = compute_all_variants(sig, t, plate_ttp_min=float(plate_ttp))
    manual_val = float(manual_lbl["chip_ttp_min"])
    out = []
    for v in VARIANTS:
        r = results[v]
        delta = (r.ttp_min - manual_val) if r.ttp_min is not None else None
        out.append({"chip_tag": chip_tag, "well": well["well"], "label": well["label"],
                    "log10_conc": well.get("log10_conc"),
                    "manual_ttp_min": manual_val, "plate_ttp_min": float(plate_ttp),
                    "variant": v, "systematic_ttp_min": r.ttp_min, "delta_min": delta,
                    "reason": r.reason})
    return out


def _write_csv(rows: list[dict]) -> None:
    import csv
    with open(CSV_PATH, "w", newline="") as f:
        cols = ["chip_tag", "well", "label", "log10_conc",
                "manual_ttp_min", "plate_ttp_min", "variant",
                "systematic_ttp_min", "delta_min", "reason"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {CSV_PATH}  ({len(rows)} rows)")


def _plot(rows: list[dict]) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.28)

    # Row 1: per-variant scatter panels
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[0, i])
        valid = [r for r in rows if r["variant"] == v and r["systematic_ttp_min"] is not None]
        lo, hi = 0.0, 25.0
        ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=1.0)
        ax.fill_between([lo, hi], [lo - 2, hi - 2], [lo + 2, hi + 2],
                        color="#4CAF50", alpha=0.10)
        for chip in CHIP_LABELS:
            sub = [r for r in valid if r["chip_tag"] == chip]
            if not sub: continue
            ax.scatter([r["manual_ttp_min"] for r in sub],
                       [r["systematic_ttp_min"] for r in sub],
                       s=80, alpha=0.75, color=CHIP_COLOURS[chip],
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=CHIP_LABELS[chip] if i == 0 else None)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if i == 0:
            ax.set_ylabel("systematic TTP (min)")
        ax.set_xlabel("manual TTP (min)")
        mae = float(np.mean([abs(r["delta_min"]) for r in valid])) if valid else float("nan")
        within2 = 100.0 * sum(1 for r in valid if abs(r["delta_min"]) <= 2.0) / len(valid) if valid else 0
        ax.set_title(f"{VARIANT_LABELS[v]}\nMAE={mae:.2f} · {within2:.0f}% within ±2 min · n={len(valid)}",
                     fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.spines["bottom"].set_visible(True); ax.spines["left"].set_visible(True)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, frameon=True)

    # Row 2 spans all 4 columns: per-variant MAE bars + per-chip breakdown
    axB = fig.add_subplot(gs[1, :])
    xs = np.arange(len(VARIANTS))
    chip_offset = {chip: (i - 1) * 0.22 for i, chip in enumerate(CHIP_LABELS)}
    width = 0.20
    for chip in CHIP_LABELS:
        maes = []
        for v in VARIANTS:
            sub = [r for r in rows if r["variant"] == v and r["chip_tag"] == chip
                                       and r["systematic_ttp_min"] is not None]
            maes.append(float(np.mean([abs(r["delta_min"]) for r in sub])) if sub else 0.0)
        offset = chip_offset[chip]
        bars = axB.bar(xs + offset, maes, width, color=CHIP_COLOURS[chip],
                        edgecolor="black", linewidth=0.5, alpha=0.85, label=CHIP_LABELS[chip])
        for b, v in zip(bars, maes):
            if v > 0:
                axB.text(b.get_x() + b.get_width() / 2, v + 0.08,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Overall MAE line
    overall = []
    for v in VARIANTS:
        sub = [r for r in rows if r["variant"] == v and r["systematic_ttp_min"] is not None]
        overall.append(float(np.mean([abs(r["delta_min"]) for r in sub])) if sub else float("nan"))
    axB.plot(xs, overall, marker="D", color="#111", linewidth=1.2, markersize=8,
             label="pooled MAE", zorder=5)
    for x, v in zip(xs, overall):
        axB.text(x, v + 0.15, f"pool {v:.2f}", ha="center", va="bottom",
                 fontsize=9, color="#111", fontweight="bold")

    axB.axhline(2.0, linestyle="--", color="#43A047", linewidth=1.2, label="±2 min tolerance")
    axB.set_xticks(xs)
    axB.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=9)
    axB.set_ylabel("MAE (min) vs manual TTP")
    axB.set_title("MAE per variant, split by held-out chip", fontsize=11)
    axB.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    axB.legend(loc="upper left", fontsize=8, frameon=True, ncol=4)
    for sp in axB.spines.values(): sp.set_visible(False)
    axB.spines["bottom"].set_visible(True); axB.spines["left"].set_visible(True)

    fig.suptitle("Systematic-TTP rule validation on UTI — 4 variants, plate TTP as anchor",
                 fontsize=13, y=0.995, fontweight="bold")
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {FIG_PATH}")


def main() -> None:
    manual = _load_manual_labels()
    all_rows: list[dict] = []
    for path in CHIP_STAGES:
        stages = json.loads(path.read_text())
        chip_tag = stages["chip_tag"]
        print(f"\n=== {chip_tag}  ({len(stages['wells'])} wells) ===")
        for w in stages["wells"]:
            rows = _run_one(chip_tag, w, manual)
            all_rows.extend(rows)
            # Print one row per well summarising all variants
            v_map = {r["variant"]: r for r in rows}
            v1 = v_map["closest_peak_d1"]
            if v1["systematic_ttp_min"] is None:
                print(f"  well {v1['well']:2d} ({v1['label']:>10s})  SKIP: {v1['reason']}")
                continue
            print(f"  well {v1['well']:2d} ({v1['label']:>10s})  manual={v1['manual_ttp_min']:6.2f}  "
                  f"plate={v1['plate_ttp_min']:6.2f}  ", end="")
            for v in VARIANTS:
                r = v_map[v]
                if r["systematic_ttp_min"] is None:
                    print(f"{v[:8]:>8s}=  none  ", end="")
                else:
                    print(f"{v[:8]:>8s}={r['systematic_ttp_min']:5.2f} (Δ{r['delta_min']:+5.2f})  ", end="")
            print()
    _write_csv(all_rows)
    _plot(all_rows)

    # Summary table
    print("\n=== SUMMARY: MAE per variant ===")
    for v in VARIANTS:
        sub = [r for r in all_rows if r["variant"] == v and r["systematic_ttp_min"] is not None]
        if not sub:
            print(f"  {VARIANT_LABELS[v]}: no valid rows")
            continue
        deltas = [r["delta_min"] for r in sub]
        abs_d = [abs(d) for d in deltas]
        print(f"  {VARIANT_LABELS[v]:35s} n={len(sub):2d}  MAE={np.mean(abs_d):.2f}  "
              f"median|Δ|={np.median(abs_d):.2f}  bias={np.mean(deltas):+.2f}  "
              f"within ±2: {100*sum(1 for d in abs_d if d <= 2)/len(sub):.0f}%")


if __name__ == "__main__":
    main()
