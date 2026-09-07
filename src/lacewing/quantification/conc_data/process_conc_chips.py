"""Data · KP dataset chip processing pipeline. [Cat A] Report §NewData.

Process the new "Concentration Data Experiment" chips through the same
preprocessing pipeline used for UTI (BS + MAD-ABCD + spatA3 + artefact-trim).

Per supervisor guidance (2026-08-14):
  - Two chips, one xlsx per chip (Chip 1.xlsx, Chip 2.xlsx).
  - Chip layout (from Chip Order.xlsx) is 5 rows × 2 cols per chip:
      Row 0:  1e6 (L, w0)   1e5 (R, w1)
      Row 1:  1e4 (L, w2)   1e3 (R, w3)
      Row 2:  1e6 (L, w4)   1e5 (R, w5)
      Row 3:  1e4 (L, w6)   1e3 (R, w7)
      Row 4:  PTC/NTC (w8)  NTC/PTC (w9)   ← Chip 1: PTC/NTC ; Chip 2: NTC/PTC
  - Plate TTP for a chip well = per-concentration Cq mean × 0.5 min (parse_chip_xlsx.py).
  - Both chip wells at the same concentration receive the SAME plate TTP.

Output: Analysis/quantification/conc_data/output/<chip_tag>_stages.json (per chip).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lacewing.quantification.chip_pipeline import process_chip as pipe


# src/lacewing/quantification/conc_data/process_conc_chips.py
# -> repo root is parents[4]
_env_data = os.environ.get("LACEWING_DATA_ROOT")
PROJECT_ROOT = Path(_env_data) if _env_data else Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "Data" / "Concentration Data Experiment"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATE_FEATURES_JSON = DATA_ROOT / "plate_features.json"

# Mapping from chip well concentration (as log10) to the Kp_N label in the xlsx.
_LOG10_TO_KP = {3.0: "Kp_3", 4.0: "Kp_4", 5.0: "Kp_5", 6.0: "Kp_6"}


@dataclass
class ChipConfig:
    chip_tag: str
    chip_dir: Path
    chip_label: str
    plate_key: str                        # "chip1" | "chip2" — index into plate_features.json
    well_labels: dict[int, str]
    well_log10: dict[int, float]          # {well_idx: log10(copies)} — NaN for PTC/NTC
    ntc_wells: tuple[int, ...]            # wells treated as "no amplification" in the pipeline
    trim_search_max_min: float | None = None


def _build_chip_configs() -> list[ChipConfig]:
    # Common layout (concentrations shared across both chips for wells 0-7).
    base_labels = {
        0: "1e6 (L)", 1: "1e5 (R)",
        2: "1e4 (L)", 3: "1e3 (R)",
        4: "1e6 (L)", 5: "1e5 (R)",
        6: "1e4 (L)", 7: "1e3 (R)",
    }
    base_log10 = {0: 6.0, 1: 5.0, 2: 4.0, 3: 3.0,
                  4: 6.0, 5: 5.0, 6: 4.0, 7: 3.0}

    chip1 = ChipConfig(
        chip_tag="conc_260812_KP_01",
        chip_dir=DATA_ROOT / "D20260812_E00_C00_F4500KHz_U_KP_conc_01",
        chip_label="Chip 1 (2026-08-12, KP)",
        plate_key="chip1",
        well_labels={**base_labels, 8: "PTC", 9: "NTC"},
        well_log10={**base_log10, 8: float("nan"), 9: float("nan")},
        ntc_wells=(9,),   # only NTC treated as no-amp for pipeline QC
    )
    chip2 = ChipConfig(
        chip_tag="conc_260813_KP_DDM_02",
        chip_dir=DATA_ROOT / "D20260813_E00_C00_F4500KHz_U_DDM_KP_con_02",
        chip_label="Chip 2 (2026-08-13, KP + DDM)",
        plate_key="chip2",
        well_labels={**base_labels, 8: "NTC", 9: "PTC"},
        well_log10={**base_log10, 8: float("nan"), 9: float("nan")},
        ntc_wells=(8,),
    )
    # Chips 3, 4, 5 have no plate data — all use the pooled (Chip1 + Chip2 mean)
    # qLAMP TTP anchor. Per Chip Order.xlsx, all three share the "PTC then NTC"
    # ordering seen on Chip 1 (well 8 = PTC, well 9 = NTC).
    pooled_layout = {
        "well_labels": {**base_labels, 8: "PTC", 9: "NTC"},
        "well_log10":  {**base_log10, 8: float("nan"), 9: float("nan")},
        "ntc_wells":   (9,),
    }
    chip3 = ChipConfig(
        chip_tag="conc_260820_KP_DDM_03",
        chip_dir=DATA_ROOT / "D20260820_E00_C00_F4500KHz_U_DDM_KP_conc_03",
        chip_label="Chip 3 (2026-08-20, KP + DDM)  ·  pooled qLAMP",
        plate_key="pooled_chip1_chip2",
        **pooled_layout,
    )
    chip4 = ChipConfig(
        chip_tag="conc_260823_KP_DDM_04",
        chip_dir=DATA_ROOT / "D20260823_E00_C00_F4500KHz_U_DDM_KP_04_05",
        chip_label="Chip 4 (2026-08-23, KP + DDM)  ·  pooled qLAMP",
        plate_key="pooled_chip1_chip2",
        **pooled_layout,
    )
    chip5 = ChipConfig(
        chip_tag="conc_260827_KP_DDM_05",
        chip_dir=DATA_ROOT / "D20260827_E00_C00_F4500KHz_U_DDM_KP_Conc_05",
        chip_label="Chip 5 (2026-08-27, KP + DDM)  ·  pooled qLAMP",
        plate_key="pooled_chip1_chip2",
        **pooled_layout,
    )
    return [chip1, chip2, chip3, chip4, chip5]


def _plate_block_for_well(w_idx: int, cfg: ChipConfig, plate_per_conc: dict) -> dict | None:
    """Return the plate TTP block for a chip well, or None for PTC/NTC/unknown."""
    log10 = cfg.well_log10.get(w_idx)
    if log10 is None or not np.isfinite(log10):
        return None
    kp_label = _LOG10_TO_KP.get(float(log10))
    if kp_label is None:
        return None
    entry = plate_per_conc.get(kp_label)
    if entry is None:
        return None
    is_pooled = cfg.plate_key == "pooled_chip1_chip2"
    if is_pooled:
        note = (
            f"pooled qLAMP TTP for {kp_label}: mean of Chip 1 "
            f"({entry.get('ttp_min_chip1'):.3f} min) and Chip 2 "
            f"({entry.get('ttp_min_chip2'):.3f} min); "
            f"used because Chip {cfg.chip_tag[-2:]} has no direct plate data."
        )
    else:
        note = (
            f"plate TTP = mean(Cq) of all {entry.get('n_positive')} {kp_label} rows in this chip's "
            f"xlsx × 0.5 min; both chip wells at {kp_label} get the same value."
        )
    return {
        "kp_label": kp_label,
        "cq_mean": entry.get("cq_mean"),  # None for pooled entries
        "takeoff_min": entry.get("ttp_min"),
        "n_positive": entry.get("n_positive"),  # None for pooled entries
        "n_replicates": entry.get("n_replicates"),  # None for pooled entries
        "raw_cqs": entry.get("raw_cqs"),  # None for pooled entries
        "assignment_note": note,
    }


def _decimate_or_none(x, n):
    if x is None:
        return None
    return [None if not np.isfinite(v) else float(v) for v in pipe._decimate(x, n).tolist()]


def _result_to_payload(result, cfg: ChipConfig, plate_per_conc: dict, decimate_to: int = 300) -> dict:
    wells = []
    for w in result.wells:
        wells.append({
            "well": w.well,
            "label": w.label,
            "log10_conc": (None if not np.isfinite(w.log10_conc) else w.log10_conc),
            "n_active_raw": w.n_active_raw,
            "n_kept": w.n_kept,
            "search_start_min": w.search_start_min,
            "time_min":        _decimate_or_none(w.time_min, decimate_to),
            "smoothed_signal": _decimate_or_none(w.smoothed_signal, decimate_to),
            "mean_after_spat": _decimate_or_none(w.mean_after_spat, decimate_to),
            "ttp": {
                "threshold_derivative": (None if not np.isfinite(w.ttp_threshold)
                                         else w.ttp_threshold),
                "cy0": None if not np.isfinite(w.ttp_cy0) else w.ttp_cy0,
                "sdm": None if not np.isfinite(w.ttp_sdm) else w.ttp_sdm,
            },
            "plate": _plate_block_for_well(w.well, cfg, plate_per_conc),
        })
    return {
        "chip_name": result.chip_name,
        "chip_tag": cfg.chip_tag,
        "chip_label": cfg.chip_label,
        "chip_dir": str(cfg.chip_dir.relative_to(PROJECT_ROOT)),
        "chip_cut_min": result.chip_cut_min,
        "per_well_cut_min": [(None if not np.isfinite(v) else float(v))
                              for v in result.per_well_cut_min],
        "wells": wells,
    }


def process_one(cfg: ChipConfig, decimate_to: int = 300) -> None:
    print(f"\n{'=' * 70}\nProcessing {cfg.chip_tag}\n  chip_dir: {cfg.chip_dir}\n{'=' * 70}")
    if not cfg.chip_dir.exists():
        print(f"  [SKIP] chip_dir not found")
        return

    plate_data = json.loads(PLATE_FEATURES_JSON.read_text())
    plate_per_conc = plate_data[cfg.plate_key]["per_conc_kp"]

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
        payload = _result_to_payload(result, cfg, plate_per_conc, decimate_to=decimate_to)

        out_json = OUT_DIR / f"{cfg.chip_tag}_stages.json"
        out_json.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"[ok] wrote {out_json}  ({out_json.stat().st_size:,} bytes)")

    finally:
        pipe.WELL_LABELS = old_labels
        pipe.WELL_LOG10  = old_log10
        pipe.NTC_WELLS   = old_ntc
        pipe.TRIM_SEARCH_MAX_MIN = old_trim


def main() -> None:
    for cfg in _build_chip_configs():
        process_one(cfg)


if __name__ == "__main__":
    main()
