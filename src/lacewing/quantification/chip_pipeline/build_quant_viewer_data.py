"""Build the JSON data file for the UTI quant-method viewer.

Runs the same UTI processing pipeline as build_uti_classifier_cache.py
(BS + MAD-ABCD QC k=1.5 + spatA3), then for each of the 3 amp-positive SD
chips writes per-well:
    time_min          — chip time axis, minutes
    final_mean        — well-mean signal (post-spatA3), the same input the classical extractors run on
    pixel_traces      — list of per-pixel spatA3 signals (kept pixels only, for background overlay)
    manual_ttp        — user's manually-clicked TTP (from manual_chip_ttps.json)
    plate_ttp         — LC96 plate qPCR TTP in minutes (from manual_chip_ttps.json), or null
    is_no_amp         — bool

Output:
    Analysis/visualisation/uti_method_steps_data.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline.process_all_uti_chips import CHIPS
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
OUT_PATH = LACEWING_PKG_DIR / "visualisation" / "uti_method_steps_data.json"
MANUAL_LABELS = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output" / "manual_chip_ttps.json"

# 3 amp-positive SD chips only (skip NC test chip — nothing amplifies, methods trivially fail).
INCLUDE_CHIPS = {"uti_260630_EC_SD", "uti_260710_EC_SD", "uti_260728_EC_SD", "uti_260808_KP"}
# Cap pixel-trace count per well for JSON size sanity (viewer performance).
# Wells have 500–1600 kept pixels; 300 sampled pixels still make a dense
# translucent cloud but keeps the JSON under ~70 MB.
MAX_PIXELS_PER_WELL = 300
# Round pixel-trace values to 5 decimals — signals are in millivolts,
# any precision past 1e-5 V is below the sensor noise floor and just
# inflates the JSON. Well-mean stays full precision.
PIXEL_ROUND_DECIMALS = 5


def _load_manual_labels() -> dict:
    return json.loads(MANUAL_LABELS.read_text())["labels"]


def main() -> None:
    labels = _load_manual_labels()

    chips_out: dict[str, dict] = {}
    for cfg in CHIPS:
        if cfg.chip_tag not in INCLUDE_CHIPS:
            continue
        print(f"\n{'=' * 70}\n{cfg.chip_tag}\n{'=' * 70}")
        if not cfg.chip_dir.exists():
            print(f"  [skip] missing chip dir"); continue

        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10 = cfg.well_log10
        pipe.NTC_WELLS = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min

        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)

        wells_out = []
        for w in result.wells:
            if w.per_pixel_spat is None or len(w.per_pixel_spat) == 0:
                continue

            X = w.per_pixel_spat.astype(np.float32, copy=False)
            n_kept = X.shape[0]
            # Random-sample pixel traces to cap JSON size.
            if n_kept > MAX_PIXELS_PER_WELL:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(n_kept, size=MAX_PIXELS_PER_WELL, replace=False)
                pixel_traces = X[idx]
                sampled = True
            else:
                pixel_traces = X
                sampled = False

            key = f"{cfg.chip_tag}__well{int(w.well)}"
            lbl = labels.get(key, {})

            wells_out.append({
                "well": int(w.well),
                "label": w.label,
                "log10_conc": (None if not np.isfinite(w.log10_conc) else float(w.log10_conc)),
                "n_kept": int(n_kept),
                "n_pixels_shown": int(pixel_traces.shape[0]),
                "sampled": sampled,
                "time_min": [float(x) for x in w.time_min.tolist()],
                "final_mean": [float(x) for x in w.mean_after_spat.tolist()],
                "pixel_traces": [[round(float(v), PIXEL_ROUND_DECIMALS) for v in row.tolist()] for row in pixel_traces],
                "manual_ttp": lbl.get("chip_ttp_min"),
                "plate_ttp": lbl.get("plate_ttp_mean_min"),
                "is_no_amp": bool(lbl.get("is_no_amp", False)) if lbl else None,
            })
            print(f"  well {int(w.well):2d} {w.label:12s}  T={len(w.time_min):4d}  "
                  f"kept={n_kept:5d}  shown={pixel_traces.shape[0]:4d}  "
                  f"manual_ttp={lbl.get('chip_ttp_min')}  plate_ttp={lbl.get('plate_ttp_mean_min')}")

        chips_out[cfg.chip_tag] = {
            "folder": cfg.chip_dir.name,
            "wells": wells_out,
        }

    doc = {"chips": chips_out}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"\n[ok] wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
