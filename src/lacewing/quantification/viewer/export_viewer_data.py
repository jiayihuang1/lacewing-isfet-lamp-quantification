"""Export a single JSON sidecar for the quantification results viewer.

Walks Analysis/quantification/methods/results/ and Analysis/quantification/results/
looking for dirs with canonical predictions.npz + labels.npz.  Groups by method
family (rule-based / f4 / p1_fb / p1_fc / p1_fd / p1_fe / p2 / p3 / etc.) with
seed as a sub-key.

For each (method_family, seed) cell, emits:
  1. All scoreboard metrics (from the harness — recomputed here so the viewer
     is self-contained).
  2. Sample of per-pixel test-set traces + per-pixel predicted TTP + per-pixel
     true TTP + concentration group.
  3. Well-mean predicted vs. true TTP for the plot.

Compression:
  * Traces downsampled by DOWN=4 (450 → 113 samples per pixel).
  * Cap per-well pixel sample at N_PER_WELL (default 100).
  * Round floats to 3 dp.

The output file is Analysis/quantification/viewer/quant_viewer_data.json.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.schema import read_predictions, read_labels
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
METHODS_RESULTS = LACEWING_PKG_DIR / "quantification" / "methods" / "results"
RULE_RESULTS = LACEWING_PKG_DIR / "quantification" / "results"
CACHE_PATH = LACEWING_PKG_DIR / "regression" / "data" / "cache" / "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz"
SCOREBOARD_CSV = LACEWING_PKG_DIR / "quantification" / "eval" / "scoreboard.csv"
OUT_PATH = LACEWING_PKG_DIR / "quantification" / "viewer" / "quant_viewer_data.json"
HTML_TEMPLATE = LACEWING_PKG_DIR / "quantification" / "viewer" / "quant_viewer.html"
HTML_STANDALONE = LACEWING_PKG_DIR / "quantification" / "viewer" / "quant_viewer_standalone.html"

DOWN = 4
N_PER_WELL = 100
SAMPLES_PER_MIN = 15
N_SAMPLES = 450


def _family_of(method_id: str) -> str | None:
    """Group scoreboard method_ids into families for the dropdown."""
    if method_id in ("rule_ttp", "rule_sdm", "rule_cy0"):
        return method_id  # Rule-based methods are their own "family" for viewer.
    m = re.match(r"^(f4_wmse_cnn1d_spatA3)_seed(\d+)$", method_id)
    if m:
        return "f4_wmse"
    m = re.match(r"^(p1_fb_slide_cls_w\d+_stride\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return "p1_fb"
    m = re.match(r"^(p1_fc_slide_reg_w\d+_stride\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return "p1_fc"
    m = re.match(r"^(p1_fd_unet_seg_ow\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return "p1_fd"
    m = re.match(r"^(p1_fe_bigru_full_spatA3)_seed(\d+)$", method_id)
    if m:
        return "p1_fe"
    return None


def _seed_of(method_id: str) -> int | None:
    m = re.search(r"_seed(\d+)$", method_id)
    return int(m.group(1)) if m else None


def _run_dir_of(method_id: str) -> Path | None:
    """Reverse the scoreboard method_id back to its run dir."""
    if method_id == "rule_ttp":
        return RULE_RESULTS / "ttp"
    if method_id == "rule_sdm":
        return RULE_RESULTS / "sdm_cy0" / "sdm"
    if method_id == "rule_cy0":
        return RULE_RESULTS / "sdm_cy0" / "cy0"
    m = re.match(r"^f4_wmse_cnn1d_spatA3_seed(\d+)$", method_id)
    if m:
        return METHODS_RESULTS / "f6_pdf_cnn1d_spatA3_wmse" / f"seed{m.group(1)}_sigma5_wmse0.1"
    m = re.match(r"^(p1_fb_slide_cls_w\d+_stride\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return METHODS_RESULTS / m.group(1) / f"seed{m.group(2)}"
    m = re.match(r"^(p1_fc_slide_reg_w\d+_stride\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return METHODS_RESULTS / m.group(1) / f"seed{m.group(2)}"
    m = re.match(r"^(p1_fd_unet_seg_ow\d+_spatA3)_seed(\d+)$", method_id)
    if m:
        return METHODS_RESULTS / m.group(1) / f"seed{m.group(2)}"
    m = re.match(r"^(p1_fe_bigru_full_spatA3)_seed(\d+)$", method_id)
    if m:
        return METHODS_RESULTS / m.group(1) / f"seed{m.group(2)}"
    return None


def _load_scoreboard_rows() -> list[dict]:
    rows: list[dict] = []
    if not SCOREBOARD_CSV.exists():
        return rows
    with SCOREBOARD_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("method_id", "config_json"):
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def _load_cache_traces() -> dict:
    """Return dict[chip_id][well_id] = list of traces (each length 450)."""
    data = np.load(CACHE_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32)
    chip_id = data["chip_id"]
    well_id = data["well_id"].astype(int)
    split = data["split"].astype(int)

    # SD chip only (split == 1 == test).
    mask = split == 1
    X = X[mask]
    chip_id = chip_id[mask]
    well_id = well_id[mask]

    out: dict[str, dict[int, list[list[float]]]] = {}
    unique_chips = np.unique(chip_id)
    for c in unique_chips:
        c_key = str(c)
        out[c_key] = {}
        cmask = chip_id == c
        for w in np.unique(well_id[cmask]):
            w_int = int(w)
            wmask = cmask & (well_id == w)
            traces = X[wmask]  # (n_px, 450)
            out[c_key][w_int] = traces
    return out


def _build_cell(preds, labels, cache_traces):
    """Assemble the viewer data for a single (method_id / seed) cell.

    Returns dict with:
      - per_pixel: sampled traces + prediction + true TTP + concentration.
      - per_well: {well_key: {log_conc, y_true, y_pred_mean, y_pred_std, n_px}}
    """
    # Test-only mask.
    mask = labels.split == "test"
    ttp_pred = preds.ttp_pred_min[mask]
    ttp_true = labels.ttp_true_min[mask]
    chip_id = labels.chip_id[mask]
    well_id = labels.well_id[mask]
    log_c = labels.log10_concentration[mask]

    per_well: dict[str, dict] = {}
    per_pixel_samples: list[dict] = []

    rng = np.random.default_rng(0)

    unique_chip_wells = set(zip(chip_id.tolist(), well_id.tolist()))
    for c, w in sorted(unique_chip_wells, key=lambda t: (t[0], t[1])):
        m = (chip_id == c) & (well_id == w)
        if int(m.sum()) == 0:
            continue
        ttp_pred_w = ttp_pred[m]
        ttp_true_w = ttp_true[m]
        log_c_w = log_c[m]
        key = f"{c}::{int(w)}"

        # Per-well summary.
        per_well[key] = {
            "chip_id": str(c),
            "well_id": int(w),
            "log10_conc": round(float(log_c_w.mean()), 3),
            "y_true_min": round(float(ttp_true_w.mean()), 3),
            "y_pred_mean_min": round(float(ttp_pred_w.mean()), 3),
            "y_pred_std_min": round(float(ttp_pred_w.std()), 3),
            "y_pred_median_min": round(float(np.median(ttp_pred_w)), 3),
            "n_px": int(m.sum()),
        }

        # Sample per-pixel trace + prediction (cap at N_PER_WELL).
        n_available = int(m.sum())
        n_sample = min(N_PER_WELL, n_available)
        idx = rng.choice(np.where(m)[0], size=n_sample, replace=False)

        # Get traces from cache. Reverse-map: build a lookup by chip+well.
        c_str = str(c)
        if c_str not in cache_traces or int(w) not in cache_traces[c_str]:
            traces_w = None
        else:
            traces_w = cache_traces[c_str][int(w)]

        # The order in labels may not match the cache order; we assume they do
        # (both derived from the same cache in dataset order).
        # We just need `n_sample` traces from the well — take random ones.
        if traces_w is None or len(traces_w) < n_sample:
            trace_arr = np.zeros((n_sample, N_SAMPLES // DOWN), dtype=np.float32)
        else:
            trace_idx = rng.choice(len(traces_w), size=n_sample, replace=False)
            trace_arr = traces_w[trace_idx][:, ::DOWN]  # downsample

        # Pack.
        per_pixel_samples.append({
            "well_key": key,
            "log10_conc": round(float(log_c_w.mean()), 3),
            "y_true_min": round(float(ttp_true_w.mean()), 3),
            "n_sample": int(n_sample),
            "n_total": n_available,
            "traces": [[round(v, 3) for v in t.tolist()] for t in trace_arr],
            "pred_ttp_min": [round(float(v), 3) for v in ttp_pred[idx].tolist()],
        })

    return {"per_well": per_well, "per_pixel_samples": per_pixel_samples}


def main() -> None:
    print("== Exporting quantification viewer data ==")

    scoreboard = _load_scoreboard_rows()
    if not scoreboard:
        print(f"[warn] scoreboard.csv missing at {SCOREBOARD_CSV}")

    print(f"Loading trace cache from {CACHE_PATH.name} ...")
    cache_traces = _load_cache_traces()

    # Group scoreboard rows by family.
    families: dict[str, dict] = {}
    scoreboard_by_id = {r["method_id"]: r for r in scoreboard}

    for row in scoreboard:
        method_id = row["method_id"]
        family = _family_of(method_id)
        if family is None:
            print(f"[skip] {method_id}: unknown family")
            continue
        run_dir = _run_dir_of(method_id)
        if run_dir is None or not (run_dir / "predictions.npz").exists():
            print(f"[skip] {method_id}: run dir missing ({run_dir})")
            continue

        preds = read_predictions(run_dir / "predictions.npz")
        labels = read_labels(run_dir / "labels.npz")
        cell = _build_cell(preds, labels, cache_traces)

        seed = _seed_of(method_id)
        if family not in families:
            families[family] = {
                "family_name": family,
                "seeds": {},
            }
        seed_key = "-" if seed is None else str(seed)
        cell["metrics"] = {k: v for k, v in row.items() if k not in ("method_id", "config_json")}
        cell["method_id"] = method_id
        families[family]["seeds"][seed_key] = cell
        print(f"[ok]   {method_id}: n_wells={len(cell['per_well'])}, n_samples={sum(p['n_sample'] for p in cell['per_pixel_samples'])}")

    # Time axis in minutes (downsampled).
    # `X[:, ::DOWN]` gives ceil(N_SAMPLES / DOWN) samples, not N_SAMPLES // DOWN.
    n_ds = int(np.ceil(N_SAMPLES / DOWN))
    t_axis = [round((i * DOWN) / SAMPLES_PER_MIN, 3) for i in range(n_ds)]

    out = {
        "schema": 1,
        "generated_at": "auto",
        "n_families": len(families),
        "families": list(families.keys()),
        "cells": families,
        "t_axis_min": t_axis,
        "down_factor": DOWN,
        "n_samples_per_trace": n_ds,
        "scoreboard": {
            "method_ids": [r["method_id"] for r in scoreboard],
            "rows": scoreboard,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(out)
    OUT_PATH.write_text(json_str)
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUT_PATH.name} ({size_mb:.1f} MB) — {len(families)} families, {sum(len(f['seeds']) for f in families.values())} seeds")

    # Also produce a standalone HTML with the JSON embedded inline so it opens
    # by double-click without needing an HTTP server (browsers block fetch()
    # from file:// URLs to sibling JSON for CORS reasons).
    # The template `quant_viewer.html` is left unchanged (it can still be used
    # via a local HTTP server); the standalone variant is what most users want.
    if HTML_TEMPLATE.exists():
        html = HTML_TEMPLATE.read_text()
        import re
        replacement = (
            '// Embedded data (auto-injected by export_viewer_data.py).\n'
            f'DATA = {json_str};\n'
            'document.getElementById("status").textContent = "loaded (embedded).";\n'
            'initControls();\n'
            'render();\n'
            'renderSummary();'
        )
        pattern = re.compile(
            r'fetch\("quant_viewer_data\.json"\)\s*\.then\([^}]*?\}\)\s*\.catch\([^}]*?\}\);',
            re.DOTALL,
        )
        # Lambda replacement so backslash sequences in JSON (e.g. \uXXXX)
        # aren't interpreted as regex group refs.
        new_html = pattern.sub(lambda m: replacement, html, count=1)
        if new_html == html:
            print(f"[warn] no fetch(...) block found in {HTML_TEMPLATE.name}; standalone not built.")
        else:
            HTML_STANDALONE.write_text(new_html)
            html_mb = HTML_STANDALONE.stat().st_size / 1024 / 1024
            print(f"Wrote {HTML_STANDALONE.name} ({html_mb:.1f} MB — self-contained, double-clickable)")


if __name__ == "__main__":
    main()
