"""Run the multi-titan pipeline on all 4 UTI trial chips (A3).

Reuses lacewing.quantification.chip_pipeline.process_chip.process_chip() +
result_to_payload() by monkey-patching the module-level chip config globals
for each chip.

Outputs per-chip stages JSON to Analysis/quantification/chip_pipeline/output/
    <chip_tag>_stages.json
    <chip_tag>_ttp_summary.json

Skips the KP chip (supervisor flag).

Usage
-----
    python -m lacewing.quantification.chip_pipeline.process_all_uti_chips
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import the existing pipeline module (this triggers _titan_setup which pins the multi-titan clone).
from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
DATA_ROOT = PROJECT_ROOT / "Data" / "UTI Trial Data"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Per-chip config table
# ---------------------------------------------------------------------------
# Each entry defines everything needed to process one chip:
#   chip_tag         — output-file prefix
#   chip_dir         — where the raw .bin files live
#   well_labels      — chip_well_idx (0-9) → human-readable label
#   well_log10       — chip_well_idx → log10 concentration (NaN for NTC/water)
#   ntc_wells        — set of chip_well_idx that are NTC/water
#   lc96_well_map    — chip_well_idx → list of LC96 plate positions (3 replicates)
#   lc96_unreliable  — set of chip_well_idx flagged unreliable by supervisor
#   lc96_export      — path to LC96 xlsx or txt export (for Cq lookup)
#   plate_features   — path to plate_features.json (from A1)
# ---------------------------------------------------------------------------

@dataclass
class ChipConfig:
    chip_tag: str
    chip_dir: Path
    well_labels: dict[int, str]
    well_log10: dict[int, float]
    ntc_wells: tuple[int, ...]
    lc96_well_map: dict[int, list[str]]
    lc96_unreliable: set[int]
    lc96_export: Path | None
    plate_features: Path
    # Optional per-chip override for the artefact-trim search window.
    # Default None → pipeline uses trim_chip_artefact's built-in 5.0 min.
    # Bump this to give the trim algorithm more room when a chip's firmware
    # settle transient is longer than 5 min (e.g. KP_03 needs ~6 min).
    trim_search_max_min: float | None = None


CHIPS: list[ChipConfig] = [
    # 26-06-24 NC test — 10 wells, each with a different primer set + water.
    # Well 8 (PTC) is the only well loaded with real E. coli DNA (1e4 c/rxn).
    # Well 9 is NTC. Others are primer-only water tests.
    ChipConfig(
        chip_tag="uti_2606024_NC_test",
        chip_dir=DATA_ROOT / "26-06-24 Negative Control Test" / "D20260624_E00_C00_F4500KHz_U_UTI_Trial_01_neg_test",
        well_labels={
            0: "E.coli primer + water",
            1: "K.pneumo primer + water",
            2: "P.aerug primer + water",
            3: "E.faceilis primer + water",
            4: "S.sapro primer + water",
            5: "P.mirabilis primer + water",
            6: "CTX-M-15 primer + water",
            7: "KPC primer + water",
            8: "PTC (E.coli 1e4)",
            9: "NTC",
        },
        well_log10={
            0: float("nan"), 1: float("nan"), 2: float("nan"), 3: float("nan"),
            4: float("nan"), 5: float("nan"), 6: float("nan"), 7: float("nan"),
            8: 4.0, 9: float("nan"),
        },
        ntc_wells=(0, 1, 2, 3, 4, 5, 6, 7, 9),  # all except PTC treated as "not amplifying"
        lc96_well_map={
            0: ["B2", "C2", "D2"],
            1: ["B3", "C3", "D3"],
            2: ["B4", "C4", "D4"],
            3: ["B5", "C5", "D5"],
            4: ["B6", "C6", "D6"],
            5: ["B7", "C7", "D7"],
            6: ["B8", "C8", "D8"],
            7: ["B9", "C9", "D9"],
            8: ["F2", "F3", "F4"],
            9: ["F7", "F8", "F9"],
        },
        lc96_unreliable=set(),
        lc96_export=None,
        plate_features=DATA_ROOT / "26-06-24 Negative Control Test" / "plate_features.json",
    ),
    # 26-06-30 EC — 10-well E. coli SD, chip conc E4/E3/E2/E1/NC (4 log decades)
    ChipConfig(
        chip_tag="uti_260630_EC_SD",
        chip_dir=DATA_ROOT / "26-06-30 E. Coli Serial Dilution Synthetic" / "D20260630_E00_C00_F4500KHz_U_UTI_Trial_02_02_e_coli",
        well_labels={
            0: "E4 (1e4)", 1: "E4 (1e4)",
            2: "E3 (1e3)", 3: "E3 (1e3)",
            4: "E2 (1e2)", 5: "E2 (1e2)",
            6: "E1 (1e1)", 7: "E1 (1e1)",
            8: "NC", 9: "NC",
        },
        well_log10={0: 4.0, 1: 4.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 2.0,
                    6: 1.0, 7: 1.0, 8: float("nan"), 9: float("nan")},
        ntc_wells=(8, 9),
        # From E.C LC Results.xlsx (26-06-30):
        #   E4: D5, B8, C8            E3: B7, C7, D6            E2: B6, C6, D7
        #   E1: B5, C5, D8            NTC: F4, F5, F6
        # Chip wells 0/1 (both E4) share the same triplicate; 2/3 (E3) share; etc.
        lc96_well_map={
            0: ["D5", "B8", "C8"], 1: ["D5", "B8", "C8"],
            2: ["B7", "C7", "D6"], 3: ["B7", "C7", "D6"],
            4: ["B6", "C6", "D7"], 5: ["B6", "C6", "D7"],
            6: ["B5", "C5", "D8"], 7: ["B5", "C5", "D8"],
            8: ["F4", "F5", "F6"], 9: ["F4", "F5", "F6"],
        },
        lc96_unreliable=set(),
        lc96_export=None,
        plate_features=DATA_ROOT / "26-06-30 E. Coli Serial Dilution Synthetic" / "plate_features.json",
    ),
    # 26-07-10 EC — 10-well E. coli SD, chip conc E5/E4/E3/E2/NC (4 log decades)
    ChipConfig(
        chip_tag="uti_260710_EC_SD",
        chip_dir=DATA_ROOT / "26-07-10 E. Coli Serial Dilution Synthetic" / "D20260710_E00_C00_F4500KHz_U_UTI_Trial_03_01",
        well_labels={
            0: "E5 (1e5)", 1: "E5 (1e5)",
            2: "E4 (1e4)", 3: "E4 (1e4)",
            4: "E3 (1e3)", 5: "E3 (1e3)",
            6: "E2 (1e2)", 7: "E2 (1e2)",
            8: "NTC", 9: "NTC",
        },
        well_log10={0: 5.0, 1: 5.0, 2: 4.0, 3: 4.0, 4: 3.0, 5: 3.0,
                    6: 2.0, 7: 2.0, 8: float("nan"), 9: float("nan")},
        ntc_wells=(8, 9),
        # From E.C LC.xlsx (26-07-10): E5=B5/C5/D5, E4=B6/C6/D6, E3=B7/C7/D7, E2=B8/C8/D8, NTC=F5/F6/F7
        lc96_well_map={
            0: ["B5", "C5", "D5"], 1: ["B5", "C5", "D5"],
            2: ["B6", "C6", "D6"], 3: ["B6", "C6", "D6"],
            4: ["B7", "C7", "D7"], 5: ["B7", "C7", "D7"],
            6: ["B8", "C8", "D8"], 7: ["B8", "C8", "D8"],
            8: ["F5", "F6", "F7"], 9: ["F5", "F6", "F7"],
        },
        lc96_unreliable=set(),
        lc96_export=None,
        plate_features=DATA_ROOT / "26-07-10 E. Coli Serial Dilution Synthetic" / "plate_features.json",
    ),
    # 26-07-28 EC — same layout as we processed before but chip folder renamed
    #               E_coli_03 → E_coli_04 (per user 2026-07-30). Chip conc E5/E4/E3/E2 (labeled 1e5/1e4/5e3/1e2 on chip)
    #               NOTE: the "5e3" concentration on the chip corresponds to LC96 E3 (4.17e2 copies).
    ChipConfig(
        chip_tag="uti_260728_EC_SD",
        chip_dir=DATA_ROOT / "26-07-28 E. Coli Serial Dilution Synthetic" / "D20260728_E00_C00_F4500KHz_U_E_coli_04",
        well_labels={
            0: "1e5 (L)", 1: "1e5 (R)",
            2: "1e4 (L)", 3: "1e4 (R)",
            4: "5e3 (L)", 5: "5e3 (R)",
            6: "1e2 (L)", 7: "1e2 (R)",
            8: "NTC (L)", 9: "NTC (R)",
        },
        well_log10={0: 5.0, 1: 5.0, 2: 4.0, 3: 4.0,
                    4: float(np.log10(5e3)), 5: float(np.log10(5e3)),
                    6: 2.0, 7: 2.0, 8: float("nan"), 9: float("nan")},
        ntc_wells=(8, 9),
        # From 26-07-28 E.C LC Results.xlsx: E6=B3/C3/D3, E5=B4/C4/D4, E4=B5/C5/D5, E3=B6/C6/D6, NTC=F4/F5/F6
        lc96_well_map={
            0: ["B3", "C3", "D3"], 1: ["B3", "C3", "D3"],
            2: ["B4", "C4", "D4"], 3: ["B4", "C4", "D4"],
            4: ["B5", "C5", "D5"], 5: ["B5", "C5", "D5"],
            6: ["B6", "C6", "D6"], 7: ["B6", "C6", "D6"],
            8: ["F4", "F5", "F6"], 9: ["F4", "F5", "F6"],
        },
        lc96_unreliable={4, 5, 6, 7},  # supervisor flag for low-conc wet loading
        lc96_export=None,
        plate_features=DATA_ROOT / "26-07-28 E. Coli Serial Dilution Synthetic" / "plate_features.json",
    ),
    # 26-08-08 KP Serial Dilution — chip 03 (KP_03 worked; KP_04 excluded — no amp).
    # Layout from 26-08-08 KP_chip_order.xlsx (5-row × 2-col, L/R = top rows L then R).
    #   row 0: 1e6 (L, well 0)   1e5 (R, well 1)
    #   row 1: 1e4 (L, well 2)   1e3 (R, well 3)
    #   row 2: 1e6 (L, well 4)   1e5 (R, well 5)
    #   row 3: 1e4 (L, well 6)   1e3 (R, well 7)
    #   row 4: NC  (L, well 8)   NC  (R, well 9)
    # Plate qPCR TTPs from Book.xlsx (Kp_3 rows) — no .lc96p export available, so
    # plate_features.json is not built via the A1 parser. Reference numbers only:
    #   1e6 → avg 5.05 min (4 replicates, SD 0.25)
    #   1e5 → avg 5.70 min (SD 0.25)
    #   1e4 → avg 8.38 min (SD 2.66)
    #   1e3 → avg 26.65 min (SD 16.16 — 1 of 4 replicates failed at "-")
    ChipConfig(
        chip_tag="uti_260808_KP",
        chip_dir=DATA_ROOT / "26-08-08 KP Serial Dilution" / "D20260808_E00_C00_F4500KHz_U_UTI_KP_03",
        well_labels={
            0: "1e6 (L)", 1: "1e5 (R)",
            2: "1e4 (L)", 3: "1e3 (R)",
            4: "1e6 (L)", 5: "1e5 (R)",
            6: "1e4 (L)", 7: "1e3 (R)",
            8: "NC (L)",  9: "NC (R)",
        },
        well_log10={0: 6.0, 1: 5.0, 2: 4.0, 3: 3.0,
                    4: 6.0, 5: 5.0, 6: 4.0, 7: 3.0,
                    8: float("nan"), 9: float("nan")},
        ntc_wells=(8, 9),
        lc96_well_map={},  # no .lc96p export → no LC96 position mapping
        lc96_unreliable=set(),
        lc96_export=None,
        # Path is set for consistency with other entries but the file does not
        # exist for KP. _load_lc96_from_plate_features() returns {} when missing.
        plate_features=DATA_ROOT / "26-08-08 KP Serial Dilution" / "plate_features.json",
        # KP_03 has a firmware settle transient that lasts ~5.5 min (not ~2 min
        # like the SD chips). With the default search_max_min=5.0 min, the trim
        # algorithm cuts at 2.17 min (inside the still-elevated spike plateau),
        # which makes BS subtract an inflated reference and produces a ~-150 mV
        # step at t≈3.2 min post-trim across every well. Raising to 8.0 lets
        # trim_chip_artefact find the true end at ~5.97 min. (Debug plot:
        # /tmp/kp_raw.png — SD chip drops to plateau at t≈2.5 min; KP at ~5.5.)
        trim_search_max_min=8.0,
    ),
]


# ---------------------------------------------------------------------------
# LC96 lookup — use A1's plate_features.json (rule-free Cq_takeoff) rather than
# the LC96 export's auto-computed Cq (which the software sometimes marks as "-"
# for late amplification).
# ---------------------------------------------------------------------------

def _load_lc96_from_plate_features(cfg: ChipConfig) -> dict[str, dict]:
    """Return {LC96_position: {cq: float or None, call: str, takeoff_min: float or None}}
    keyed by LC96 well positions the chip cares about.
    """
    if not cfg.plate_features.exists():
        return {}
    data = json.loads(cfg.plate_features.read_text())
    wells = data.get("wells", {})
    out: dict[str, dict] = {}
    for w_idx, positions in cfg.lc96_well_map.items():
        for pos in positions:
            if pos in wells:
                w = wells[pos]
                # Use Cq_takeoff (peak-of-first-derivative) as our Cq.
                # If is_positive=False, mark call as Negative.
                out[pos] = {
                    "cq": w.get("Cq_takeoff"),
                    "call": "Positive" if w.get("is_positive") else "Negative",
                    "takeoff_min": w.get("Cq_takeoff_min"),
                    "plateau_min": w.get("Cq_plateau_min"),
                    "start_min": w.get("Cq_start_min"),
                }
    return out


def _lc96_block_for_well(w_idx: int, cfg: ChipConfig,
                          cq_by_pos: dict[str, dict]) -> dict | None:
    positions = cfg.lc96_well_map.get(w_idx)
    if positions is None:
        return None
    reps = []
    for pos in positions:
        entry = cq_by_pos.get(pos)
        if entry is None:
            reps.append({"position": pos, "cq": None, "call": "Missing"})
        else:
            reps.append({"position": pos, "cq": entry["cq"], "call": entry["call"]})
    finite_cqs = [r["cq"] for r in reps if r["cq"] is not None]
    return {
        "positions": positions,
        "replicates": reps,
        "cq_mean": (sum(finite_cqs) / len(finite_cqs)) if finite_cqs else None,
        "n_positive": len(finite_cqs),
        "n_replicates": len(reps),
        "unreliable": w_idx in cfg.lc96_unreliable,
    }


# ---------------------------------------------------------------------------
# Monkey-patched process_chip → result_to_payload flow
# ---------------------------------------------------------------------------

def _patched_result_to_payload(result, cfg: ChipConfig, decimate_to: int = 300) -> dict:
    """Version of pipe.result_to_payload that uses THIS chip's LC96 map."""
    def _to_list(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return [None if not np.isfinite(v) else float(v) for v in x.tolist()]
        return x

    cq_by_pos = _load_lc96_from_plate_features(cfg)

    wells = []
    for w in result.wells:
        wells.append({
            "well": w.well,
            "label": w.label,
            "log10_conc": (None if not np.isfinite(w.log10_conc) else w.log10_conc),
            "n_active_raw": w.n_active_raw,
            "n_kept": w.n_kept,
            "search_start_min": w.search_start_min,
            "time_min":       _to_list(pipe._decimate(w.time_min, decimate_to)),
            "time_min_diff":  _to_list(pipe._decimate(w.time_min[:-1], decimate_to)),
            "time_min_diff2": _to_list(pipe._decimate(w.time_min[:-2], decimate_to)),
            "mean_bs_raw":     _to_list(pipe._decimate(w.mean_bs_raw, decimate_to)),
            "mean_after_qc":   _to_list(pipe._decimate(w.mean_after_qc, decimate_to)),
            "mean_after_spat": _to_list(pipe._decimate(w.mean_after_spat, decimate_to)),
            "smoothed_signal": _to_list(pipe._decimate(w.smoothed_signal, decimate_to)),
            "diff_smooth":     _to_list(pipe._decimate(w.diff_smooth, decimate_to)),
            "diff2_smooth":    _to_list(pipe._decimate(w.diff2_smooth, decimate_to)),
            "ttp": {
                "threshold_derivative": (None if not np.isfinite(w.ttp_threshold)
                                          else w.ttp_threshold),
                "threshold_deriv_peak": (None if not np.isfinite(w.ttp_threshold_peak)
                                          else w.ttp_threshold_peak),
                "cy0": None if not np.isfinite(w.ttp_cy0) else w.ttp_cy0,
                "sdm": None if not np.isfinite(w.ttp_sdm) else w.ttp_sdm,
            },
            "lc96": _lc96_block_for_well(w.well, cfg, cq_by_pos),
        })

    return {
        "chip_name": result.chip_name,
        "chip_tag": cfg.chip_tag,
        "chip_cut_min": result.chip_cut_min,
        "per_well_cut_min": [
            (None if not np.isfinite(v) else float(v))
            for v in result.per_well_cut_min
        ],
        "per_well_had_artefact": [bool(v) for v in result.per_well_had_artefact],
        "filter_info": result.filter_info,
        "wells": wells,
        "config": {
            "n_wells": pipe.N_WELLS,
            "n_a_type": pipe.N_A_TYPE,
            "end_time_min": pipe.END_TIME_MIN,
            "mad_k": pipe.MAD_K,
            "filter_layers": pipe.FILTER_LAYERS,
            "spat_order": pipe.SPAT_ORDER,
            "smooth_order": pipe.SMOOTH_ORDER,
            "search_end_min": pipe.SEARCH_END_MIN,
            "threshold_frac": pipe.THRESHOLD_FRAC,
        },
    }


def process_one(cfg: ChipConfig, decimate_to: int = 300) -> None:
    print(f"\n{'=' * 70}\nProcessing {cfg.chip_tag}\n  chip_dir: {cfg.chip_dir}\n{'=' * 70}")
    if not cfg.chip_dir.exists():
        print(f"  [SKIP] chip_dir not found")
        return

    # Monkey-patch pipe's chip-config globals so pipe.process_chip() uses our config.
    old_labels = pipe.WELL_LABELS
    old_log10  = pipe.WELL_LOG10
    old_ntc    = pipe.NTC_WELLS
    old_trim   = pipe.TRIM_SEARCH_MAX_MIN
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10  = cfg.well_log10
        pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min

        result = pipe.process_chip(cfg.chip_dir)
        payload = _patched_result_to_payload(result, cfg, decimate_to=decimate_to)

        out_json = OUT_DIR / f"{cfg.chip_tag}_stages.json"
        out_json.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"[ok] wrote {out_json}  ({out_json.stat().st_size:,} bytes)")

        # Compact per-well TTP summary
        summary_rows = []
        for w in result.wells:
            summary_rows.append({
                "well": w.well, "label": w.label,
                "log10_conc": None if not np.isfinite(w.log10_conc) else w.log10_conc,
                "n_active_raw": w.n_active_raw, "n_kept": w.n_kept,
                "search_start_min": w.search_start_min,
                "ttp_threshold": None if not np.isfinite(w.ttp_threshold) else w.ttp_threshold,
                "ttp_threshold_peak": None if not np.isfinite(w.ttp_threshold_peak) else w.ttp_threshold_peak,
                "ttp_cy0": None if not np.isfinite(w.ttp_cy0) else w.ttp_cy0,
                "ttp_sdm": None if not np.isfinite(w.ttp_sdm) else w.ttp_sdm,
            })
        (OUT_DIR / f"{cfg.chip_tag}_ttp_summary.json").write_text(
            json.dumps(summary_rows, indent=2))

    finally:
        pipe.WELL_LABELS = old_labels
        pipe.WELL_LOG10  = old_log10
        pipe.NTC_WELLS   = old_ntc
        pipe.TRIM_SEARCH_MAX_MIN = old_trim


def main() -> None:
    for cfg in CHIPS:
        try:
            process_one(cfg)
        except Exception as e:
            print(f"[ERROR] {cfg.chip_tag}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
