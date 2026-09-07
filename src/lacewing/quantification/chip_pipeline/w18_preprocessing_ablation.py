"""W18 Task 9 — preprocessing ablation: MAD-k x spatA3 order sweep on UTI chips.

Answers: does the deployed MAD-ABCD (k=1.5) + spatA3 (order=3) preprocessing
combo actually beat simpler variants on UTI, when scored with the deployed
classical `threshold_derivative` extractor against manual TTP labels?

For each (mad_k, spat_order) in the 3x4=12-variant grid, this script:
  1. Monkey-patches lacewing.quantification.chip_pipeline.process_chip's
     module-level MAD_K / SPAT_ORDER globals (same pattern
     compute_chip_ttps.py / process_all_uti_chips.py use for
     WELL_LABELS / WELL_LOG10 / NTC_WELLS / TRIM_SEARCH_MAX_MIN).
  2. Re-runs pipe.process_chip() from scratch (save_per_pixel=False) on all
     4 UTI regime chips (uti_260630_EC_SD, uti_260710_EC_SD,
     uti_260728_EC_SD, uti_260808_KP), applying each chip's own
     WELL_LABELS / WELL_LOG10 / NTC_WELLS / TRIM_SEARCH_MAX_MIN from
     process_all_uti_chips.CHIPS (KP_03 keeps its trim_search_max_min=8.0
     override; SD chips use the pipeline default).
  3. For each amp-positive well with a manual chip_ttp_min label, runs
     extract_ttp / extract_cy0 / extract_sdm on w.mean_after_spat (the
     post-spatA3 well-mean the classical extractors normally consume; at
     spat_order=0 spatA_smooth_chip is a no-op, so mean_after_spat ==
     mean_after_qc — no special-casing needed).
  4. Scores |extracted_ttp - manual_ttp| per well; aggregates MAE and
     %-within-2-min per (mad_k, spat_order, method) across all 4 chips.

Output:
    Analysis/quantification/chip_pipeline/output/w18_preprocessing_ablation.csv
    columns: mad_k, spatA3_order, method, n_wells, mae_min, pass_pct_2min

Run:
    python -m lacewing.quantification.chip_pipeline.w18_preprocessing_ablation
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline import process_all_uti_chips as multi
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp
from lacewing.quantification.methods.cy0 import extract_cy0
from lacewing.quantification.methods.sdm import extract_sdm


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_CSV = OUT_DIR / "w18_preprocessing_ablation.csv"
MANUAL_LABELS_JSON = OUT_DIR / "manual_chip_ttps.json"

# The 4 UTI-regime chips (KP_03, KP_04 excluded per spec Sec4.3).
UTI_CHIP_TAGS = (
    "uti_260630_EC_SD",
    "uti_260710_EC_SD",
    "uti_260728_EC_SD",
    "uti_260808_KP",
)

# Ablation grid: MAD-k x spatA3 order = 3 x 4 = 12 variants.
MAD_K_GRID = (1.5, 2.0, 3.0)
SPAT_ORDER_GRID = (0, 1, 3, 5)

PASS_THRESHOLD_MIN = 2.0

METHODS = ("threshold_derivative", "cy0", "sdm")


def load_manual_labels() -> dict[str, dict]:
    """Return {(chip_tag, well_id): manual_record} for amp+ wells with a label."""
    data = json.loads(MANUAL_LABELS_JSON.read_text())
    out: dict[tuple[str, int], dict] = {}
    for rec in data["labels"].values():
        if rec.get("is_no_amp"):
            continue
        if rec.get("chip_ttp_min") is None:
            continue
        out[(rec["chip_tag"], rec["well_id"])] = rec
    return out


def run_variant(mad_k: float, spat_order: int, manual: dict[tuple[str, int], dict],
                 chip_cfgs: list) -> dict[str, list[float]]:
    """Run process_chip on all 4 UTI chips under this (mad_k, spat_order)
    setting and return {method: [abs_err_min, ...]} across all amp+ wells
    that have a manual label.
    """
    errors: dict[str, list[float]] = {m: [] for m in METHODS}

    old_mad_k = pipe.MAD_K
    old_spat_order = pipe.SPAT_ORDER
    old_labels = pipe.WELL_LABELS
    old_log10 = pipe.WELL_LOG10
    old_ntc = pipe.NTC_WELLS
    old_trim = pipe.TRIM_SEARCH_MAX_MIN
    try:
        pipe.MAD_K = mad_k
        pipe.SPAT_ORDER = spat_order
        for cfg in chip_cfgs:
            pipe.WELL_LABELS = cfg.well_labels
            pipe.WELL_LOG10 = cfg.well_log10
            pipe.NTC_WELLS = tuple(cfg.ntc_wells)
            pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min

            if not cfg.chip_dir.exists():
                print(f"    [SKIP] {cfg.chip_tag}: chip_dir not found")
                continue

            result = pipe.process_chip(cfg.chip_dir, save_per_pixel=False)

            for w in result.wells:
                key = (cfg.chip_tag, w.well)
                manual_rec = manual.get(key)
                if manual_rec is None:
                    continue
                manual_ttp = manual_rec["chip_ttp_min"]

                sig = w.mean_after_spat
                if sig is None or np.all(np.isnan(sig)):
                    continue
                time_min = w.time_min
                search_start = w.search_start_min

                ttp_thr, _peak = extract_ttp(time_min, sig, search_start)
                ttp_cy0 = extract_cy0(time_min, sig, search_start)
                ttp_sdm = extract_sdm(time_min, sig, search_start)

                for method, val in (
                    ("threshold_derivative", ttp_thr),
                    ("cy0", ttp_cy0),
                    ("sdm", ttp_sdm),
                ):
                    if val is None or not np.isfinite(val):
                        continue
                    errors[method].append(abs(float(val) - float(manual_ttp)))
    finally:
        pipe.MAD_K = old_mad_k
        pipe.SPAT_ORDER = old_spat_order
        pipe.WELL_LABELS = old_labels
        pipe.WELL_LOG10 = old_log10
        pipe.NTC_WELLS = old_ntc
        pipe.TRIM_SEARCH_MAX_MIN = old_trim

    return errors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manual = load_manual_labels()
    print(f"[ok] loaded {len(manual)} amp+ wells with manual chip_ttp_min "
          f"across {len(set(k[0] for k in manual))} chips")

    chip_cfgs = [c for c in multi.CHIPS if c.chip_tag in UTI_CHIP_TAGS]
    assert len(chip_cfgs) == 4, f"expected 4 UTI chips, found {len(chip_cfgs)}"

    rows: list[dict] = []
    t_start = time.time()
    n_variants = len(MAD_K_GRID) * len(SPAT_ORDER_GRID)
    variant_i = 0

    for mad_k in MAD_K_GRID:
        for spat_order in SPAT_ORDER_GRID:
            variant_i += 1
            t0 = time.time()
            print(f"\n{'=' * 70}\n[{variant_i}/{n_variants}] "
                  f"mad_k={mad_k}  spatA3_order={spat_order}\n{'=' * 70}")
            errors = run_variant(mad_k, spat_order, manual, chip_cfgs)
            dt = time.time() - t0
            print(f"  done in {dt:.1f}s")

            for method in METHODS:
                errs = errors[method]
                n = len(errs)
                if n == 0:
                    mae = float("nan")
                    pass_pct = float("nan")
                else:
                    mae = float(np.mean(errs))
                    pass_pct = 100.0 * sum(1 for e in errs if e <= PASS_THRESHOLD_MIN) / n
                rows.append({
                    "mad_k": mad_k,
                    "spatA3_order": spat_order,
                    "method": method,
                    "n_wells": n,
                    "mae_min": mae,
                    "pass_pct_2min": pass_pct,
                })
                print(f"    {method:<22} n={n:>3}  MAE={mae:.3f} min  "
                      f"pass@2min={pass_pct:.1f}%")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mad_k", "spatA3_order", "method", "n_wells",
                           "mae_min", "pass_pct_2min"])
        writer.writeheader()
        writer.writerows(rows)

    total_dt = time.time() - t_start
    print(f"\n[ok] wrote {OUT_CSV}  ({len(rows)} rows)  total {total_dt / 60:.1f} min")

    # ---- Summary: winner = min MAE for threshold_derivative (deployed baseline) ----
    print(f"\n{'=' * 70}\nSUMMARY (method=threshold_derivative, deployed baseline)\n{'=' * 70}")
    thr_rows = [r for r in rows if r["method"] == "threshold_derivative"
                and not np.isnan(r["mae_min"])]
    thr_rows.sort(key=lambda r: r["mae_min"])
    print(f"{'mad_k':>6} {'spat_order':>10} {'n':>4} {'mae_min':>9} {'pass@2min':>10}")
    for r in thr_rows:
        print(f"{r['mad_k']:>6} {r['spatA3_order']:>10} {r['n_wells']:>4} "
              f"{r['mae_min']:>9.3f} {r['pass_pct_2min']:>9.1f}%")

    if thr_rows:
        winner = thr_rows[0]
        deployed = next((r for r in rows if r["method"] == "threshold_derivative"
                          and r["mad_k"] == 1.5 and r["spatA3_order"] == 3), None)
        print(f"\nWINNER: mad_k={winner['mad_k']} spatA3_order={winner['spatA3_order']} "
              f"MAE={winner['mae_min']:.3f} min  pass@2min={winner['pass_pct_2min']:.1f}% "
              f"(n={winner['n_wells']})")
        if deployed is not None:
            print(f"DEPLOYED (mad_k=1.5, spatA3_order=3): "
                  f"MAE={deployed['mae_min']:.3f} min  "
                  f"pass@2min={deployed['pass_pct_2min']:.1f}% (n={deployed['n_wells']})")
            if (winner["mad_k"], winner["spatA3_order"]) == (1.5, 3):
                print("  -> deployed combo IS the winner.")
            else:
                delta = deployed["mae_min"] - winner["mae_min"]
                print(f"  -> deployed combo is NOT the winner "
                      f"(winner beats it by {delta:.3f} min MAE).")
    else:
        print("No valid threshold_derivative rows to summarise.")


if __name__ == "__main__":
    main()
