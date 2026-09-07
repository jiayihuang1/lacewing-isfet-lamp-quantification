"""Viewer · systematic-TTP labelling interactive viewer. [Cat A] The primary labelling-inspection tool.

Build the systematic-TTP viewer HTML.

Layout matches the existing segmentation viewer (LHS scrollable chip/well list,
RHS 2-panel view). Each well shows:
  - TOP:    well-mean signal with overlays for plate TTP, systematic TTP,
            and manual TTP (where available).
  - BOTTOM: first derivative (Savitzky-Golay smoothed) with the systematic peak marked.

Datasets grouped in LHS:
  NEW · Concentration Data Experiment  (Chip 1, Chip 2, Chip 3, Chip 4, Chip 5)
        Chips 1 and 2 use per-chip plate qLAMP TTP.
        Chips 3, 4, 5 use pooled Chip1+Chip2 mean qLAMP TTP (no plate collected).
  OLD · UTI Trial Data                  (06-30, 07-10, 07-28)

Manual TTPs (UTI only) are pulled from output/manual_chip_ttps.json.
Systematic TTPs are computed on the fly via systematic_ttp.compute_systematic_ttp.

Output: Analysis/quantification/output/systematic_ttp_viewer.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from lacewing.quantification.systematic_ttp import (
    compute_all_variants,
    second_derivative,
    VARIANTS,
    DEFAULT_SG_WINDOW,
    DEFAULT_SG_POLY,
)
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

VARIANT_LABELS = {
    "closest_peak_d1":       "V1 · closest d1 peak",
    "zero_cross_d2":         "V4 · d2 zero-cross (+→−)",
    "cy0_after_plate":       "V6 · cy0 after plate",
    "thr_deriv_after_plate": "V7 · thr_deriv after plate",
}
VARIANT_COLOURS = {
    "closest_peak_d1":       "#b30000",   # red
    "zero_cross_d2":         "#8a2be2",   # blue-violet
    "cy0_after_plate":       "#ff8c00",   # dark orange
    "thr_deriv_after_plate": "#008b8b",   # dark cyan
}


PROJECT_ROOT = DATA_ROOT  # was: parents[2]
UTI_OUT = LACEWING_PKG_DIR / "quantification" / "chip_pipeline" / "output"
UTI_STAGES = [
    UTI_OUT / "uti_260630_EC_SD_stages.json",
    UTI_OUT / "uti_260710_EC_SD_stages.json",
    UTI_OUT / "uti_260728_EC_SD_stages.json",
]
MANUAL_LABELS_JSON = UTI_OUT / "manual_chip_ttps.json"

CONC_OUT = Path(__file__).resolve().parent / "conc_data" / "output"
CONC_STAGES = [
    CONC_OUT / "conc_260812_KP_01_stages.json",
    CONC_OUT / "conc_260813_KP_DDM_02_stages.json",
    CONC_OUT / "conc_260820_KP_DDM_03_stages.json",
    CONC_OUT / "conc_260823_KP_DDM_04_stages.json",
    CONC_OUT / "conc_260827_KP_DDM_05_stages.json",
]

OUT_DIR = LACEWING_PKG_DIR / "quantification" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "systematic_ttp_viewer.html"


UTI_CHIP_LABELS = {
    "uti_260630_EC_SD": "06-30 EC SD",
    "uti_260710_EC_SD": "07-10 EC SD",
    "uti_260728_EC_SD": "07-28 EC SD",
}


def _load_manual_labels() -> dict:
    if not MANUAL_LABELS_JSON.exists():
        return {}
    return json.loads(MANUAL_LABELS_JSON.read_text()).get("labels", {})


def _finite(x):
    return [None if v is None or not np.isfinite(v) else float(v) for v in x]


def _pack_variants(sig: np.ndarray, t: np.ndarray, plate_ttp) -> dict:
    """Return {variant: {ttp_min, reason}} plus the derivative arrays for plotting."""
    results = compute_all_variants(sig, t, plate_ttp_min=plate_ttp)
    per_variant = {v: {"ttp_min": r.ttp_min, "reason": r.reason}
                    for v, r in results.items()}
    # Use V1's smoothed / d1 for plotting (all variants share the same smoothing)
    v1 = results["closest_peak_d1"]
    _, d2 = second_derivative(sig, t)
    return {
        "variants": per_variant,
        "smoothed": v1.smoothed_signal,
        "d1": v1.first_derivative,
        "d2": d2 if d2 is not None else np.zeros_like(v1.first_derivative),
    }


def _prepare_uti_well(chip_tag: str, well: dict, manual: dict) -> dict:
    key = f"{chip_tag}__well{well['well']}"
    manual_lbl = manual.get(key) or {}
    manual_ttp = manual_lbl.get("chip_ttp_min")
    plate_ttp = manual_lbl.get("plate_ttp_mean_min")

    t_full = np.asarray(well["time_min"], dtype=np.float64)
    sig = np.asarray(well["smoothed_signal"], dtype=np.float64)
    mask = np.isfinite(t_full) & np.isfinite(sig)
    t = t_full[mask]; sig = sig[mask]

    packed = _pack_variants(sig, t, plate_ttp)

    return {
        "well_id": int(well["well"]),
        "well_label": well["label"],
        "log10_conc": well.get("log10_conc"),
        "conc_str": _format_uti_conc(well.get("log10_conc"), well["label"]),
        "unit_note": "copies / reaction (chip well loading)",
        "n_active": well.get("n_active_raw"),
        "n_kept": well.get("n_kept"),
        "time_min":         _finite(t.tolist()),
        "signal":           _finite(packed["smoothed"].tolist()),
        "derivative":       _finite(packed["d1"].tolist()),
        "second_derivative": _finite(packed["d2"].tolist()),
        "manual_ttp_min":   None if manual_ttp is None else float(manual_ttp),
        "plate_ttp_min":    None if plate_ttp is None else float(plate_ttp),
        "variants":         packed["variants"],
        "is_no_amp":        bool(manual_lbl.get("is_no_amp", False)),
    }


def _prepare_conc_well(chip_tag: str, well: dict) -> dict:
    plate_info = well.get("plate") or {}
    plate_ttp = plate_info.get("takeoff_min")

    t_full = np.asarray(well["time_min"], dtype=np.float64)
    sig = np.asarray(well["smoothed_signal"], dtype=np.float64)
    mask = np.isfinite(t_full) & np.isfinite(sig)
    t = t_full[mask]; sig = sig[mask]

    packed = _pack_variants(sig, t, plate_ttp)

    return {
        "well_id": int(well["well"]),
        "well_label": well["label"],
        "log10_conc": well.get("log10_conc"),
        "conc_str": _format_conc_new(well.get("log10_conc"), well["label"]),
        "unit_note": "10^N copies (unit TBD: copies/rxn or copies/mL — awaiting supervisor)",
        "n_active": well.get("n_active_raw"),
        "n_kept": well.get("n_kept"),
        "time_min":   _finite(t.tolist()),
        "signal":     _finite(packed["smoothed"].tolist()),
        "derivative": _finite(packed["d1"].tolist()),
        "second_derivative": _finite(packed["d2"].tolist()),
        "manual_ttp_min": None,   # never manually labelled on new data
        "plate_ttp_min":  None if plate_ttp is None else float(plate_ttp),
        "variants":       packed["variants"],
        "is_no_amp": False,
        "plate_info": {
            "kp_label":       plate_info.get("kp_label"),
            "cq_mean":        plate_info.get("cq_mean"),
            "n_positive":     plate_info.get("n_positive"),
            "n_replicates":   plate_info.get("n_replicates"),
            "raw_cqs":        plate_info.get("raw_cqs"),
            "assignment_note": plate_info.get("assignment_note"),
        },
    }


def _format_uti_conc(log10_conc, well_label: str) -> str:
    if log10_conc is None or not np.isfinite(log10_conc):
        return f"{well_label} — no concentration"
    return f"10^{log10_conc:.2g} copies/rxn  ({well_label})"


def _format_conc_new(log10_conc, well_label: str) -> str:
    if log10_conc is None or not np.isfinite(log10_conc):
        return f"{well_label} — no concentration"
    return f"10^{log10_conc:.2g} copies (unit TBD)  ({well_label})"


def _build_payload() -> dict:
    manual = _load_manual_labels()

    datasets: list[dict] = []

    # NEW: Concentration Data Experiment
    new_chips: list[dict] = []
    for path in CONC_STAGES:
        if not path.exists():
            new_chips.append({
                "chip_tag": path.stem.replace("_stages", ""),
                "chip_label": path.stem.replace("_stages", ""),
                "notes_top": (
                    "PLACEHOLDER — awaiting Book.xlsx ↔ chip mapping clarification "
                    "from supervisor before this chip can be processed. "
                    "See Meeting_Notes.md 2026-08-14 Q1-Q3."
                ),
                "wells": [],
            })
            continue
        stages = json.loads(path.read_text())
        wells = [_prepare_conc_well(stages["chip_tag"], w) for w in stages["wells"]]
        chip_label = stages.get("chip_label") or stages["chip_tag"]
        is_pooled = "pooled qLAMP" in chip_label
        if is_pooled:
            note_top = (
                "Plate TTP = mean of Chip 1 and Chip 2 per-concentration TTP "
                "(this chip has no direct plate data collected). "
                "Both chip wells at the same concentration share one plate TTP."
            )
        else:
            note_top = (
                "Plate TTP = mean(Cq) of all Kp_N rows in this chip's xlsx × 0.5 min "
                "(30-sec cycles). Both chip wells at the same concentration share one plate TTP. "
                "Raw .lc96p not delivered → we rely on the vendor Cq column."
            )
        new_chips.append({
            "chip_tag": stages["chip_tag"],
            "chip_label": chip_label,
            "notes_top": note_top,
            "wells": wells,
        })
    datasets.append({
        "id": "new",
        "label": "NEW · Concentration Data Experiment",
        "colour": "#7A3C8A",
        "chips": new_chips,
    })

    # OLD: UTI trial
    uti_chips: list[dict] = []
    for path in UTI_STAGES:
        if not path.exists():
            continue
        stages = json.loads(path.read_text())
        wells = [_prepare_uti_well(stages["chip_tag"], w, manual) for w in stages["wells"]]
        uti_chips.append({
            "chip_tag": stages["chip_tag"],
            "chip_label": UTI_CHIP_LABELS.get(stages["chip_tag"], stages["chip_tag"]),
            "notes_top": (
                "Plate TTP from parse_lc96_features.py (derivative-argmax on raw .lc96p). "
                "Manual TTP overlays available for wells that were hand-labelled."
            ),
            "wells": wells,
        })
    datasets.append({
        "id": "old",
        "label": "OLD · UTI Trial Data",
        "colour": "#1976D2",
        "chips": uti_chips,
    })

    return {
        "generated_at": "2026-08-14",
        "sg_window": DEFAULT_SG_WINDOW,
        "sg_poly": DEFAULT_SG_POLY,
        "variants": [
            {"key": v, "label": VARIANT_LABELS[v], "colour": VARIANT_COLOURS[v]}
            for v in VARIANTS
        ],
        "systematic_rule": (
            "Every variant uses the same anchor: plate TTP. Then it picks a "
            "specific feature of the chip signal in [plate_TTP, plate_TTP+15 min]."
        ),
        "datasets": datasets,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Systematic TTP viewer — new + UTI data</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --panel: #f6f7fb; --border: #d5d8e0;
    --fg: #1a1d24; --muted: #666; --accent: #1f3a6b;
    --systematic: #b30000; --manual: #2ca02c; --plate: #0369a1;
    --new-badge: #7A3C8A; --old-badge: #1976D2;
    --warn: #d97706;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171d; --panel: #1b1f27; --border: #2a2f38;
      --fg: #e6e8ee; --muted: #9aa2ad; --accent: #7dbfff;
    }
  }
  html, body {
    margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px; line-height: 1.4;
  }
  header {
    padding: 12px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  header h1 { margin: 0 0 4px 0; font-size: 17px; color: var(--accent); }
  header .subtitle { color: var(--muted); font-size: 12px; }
  main { padding: 12px 20px; }

  .layout {
    display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start;
  }
  .sidebar {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px; max-height: 90vh; overflow-y: auto;
  }
  .dataset-block { margin-bottom: 14px; }
  .dataset-title {
    font-size: 12px; font-weight: 700; color: #fff;
    padding: 4px 8px; border-radius: 4px; margin-bottom: 6px;
  }
  .chip-block { margin: 6px 0 10px 0; }
  .chip-title {
    font-size: 12px; color: var(--accent); font-weight: 600;
    margin-bottom: 4px;
  }
  .chip-notes {
    color: var(--muted); font-size: 10.5px; font-style: italic;
    margin-bottom: 4px; padding: 0 2px;
  }
  .well-list {
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
  }
  .well-cell {
    padding: 6px 5px; border-radius: 3px; text-align: center;
    background: var(--bg); border: 1px solid var(--border);
    font-size: 10.5px; cursor: pointer; user-select: none; line-height: 1.25;
  }
  .well-cell .wl-label { display: block; font-family: 'SF Mono','Consolas',monospace; font-size: 10px; }
  .well-cell .wl-status { display: block; font-size: 9px; color: var(--muted); margin-top: 1px; }
  .well-cell.selected {
    outline: 2px solid var(--accent);
    background: color-mix(in srgb, var(--accent) 15%, var(--bg));
  }
  .well-cell.no-plate { border-color: var(--warn); }
  .well-cell.placeholder {
    color: var(--muted); font-style: italic; cursor: default;
    grid-column: 1 / span 2;
    padding: 8px 6px;
  }

  .info-strip {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; margin-bottom: 10px;
    font-family: 'SF Mono','Consolas',monospace; font-size: 12px;
  }
  .info-strip .row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 3px; }
  .info-strip .row > span { color: var(--muted); }
  .info-strip .row > span > b { color: var(--fg); font-weight: 600; }

  .legend {
    display: flex; gap: 16px; align-items: center; font-size: 11px;
    padding: 6px 10px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 4px; margin-bottom: 8px; flex-wrap: wrap;
  }
  .legend .item { display: inline-flex; align-items: center; gap: 5px; }
  .legend .swatch {
    width: 14px; height: 3px; display: inline-block; border-radius: 2px;
  }

  .plot-panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; margin-bottom: 12px;
  }
  .plot-panel h4 { margin: 0 0 4px 0; font-size: 12px; color: var(--accent); }
  .plot-panel .caption { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
  .variant-table {
    width: 100%; border-collapse: collapse; font-size: 11px;
    font-family: 'SF Mono','Consolas',monospace;
  }
  .variant-table th, .variant-table td {
    padding: 3px 8px; text-align: left; border-bottom: 1px solid var(--border);
  }
  .variant-table th { color: var(--muted); font-weight: 600; }
  .variant-table td.ttp { text-align: right; }
  .variant-table td .swatch {
    display: inline-block; width: 12px; height: 3px; margin-right: 6px;
    vertical-align: middle; border-radius: 2px;
  }
  .variant-toggles { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  .variant-toggles label { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; cursor: pointer; }
  .variant-toggles input { margin: 0; }
</style>
</head>
<body>
<header>
  <h1>Systematic-TTP viewer — 4 plate-anchored variants</h1>
  <div class="subtitle">
    Every variant uses plate TTP as the anchor and searches the chip signal near it.
    V1: closest 1st-derivative peak · V4: first 2nd-derivative zero-crossing (inflection) ·
    V6: cy0 tangent-intercept · V7: MATLAB thr_deriv 0.4-threshold on 1st derivative.
    V6/V7 use the classical centered-MA smoothing (span 100) and allow the search to start
    up to 2 min BEFORE the plate anchor to accommodate the pipeline's smoothing shift.
    SG window (V1/V4) = <span id="sg-info"></span>.
  </div>
</header>
<main>
  <div class="layout">
    <aside class="sidebar" id="sidebar"></aside>
    <section>
      <div class="info-strip" id="info-strip">Select a well from the sidebar to display its curves.</div>
      <div class="plot-panel">
        <h4>Systematic-TTP variants — anchor + 4 methods</h4>
        <div class="variant-toggles" id="variant-toggles"></div>
        <table class="variant-table" id="variant-table"></table>
      </div>
      <div class="plot-panel">
        <h4>Well-mean signal (smoothed) + all overlays</h4>
        <div class="caption">Solid line = Savitzky-Golay smoothed chip well-mean. Vertical lines = manual (green), plate (blue dotted), and each systematic variant (see legend colours).</div>
        <div id="plot-signal" style="height: 320px;"></div>
      </div>
      <div class="plot-panel">
        <h4>First derivative</h4>
        <div class="caption">d(signal)/dt. V1 picks the closest peak to plate; V6 uses cy0 tangent-at-inflection; V7 uses the MATLAB 0.4-threshold crossing.</div>
        <div id="plot-deriv" style="height: 240px;"></div>
      </div>
      <div class="plot-panel">
        <h4>Second derivative</h4>
        <div class="caption">d²(signal)/dt². V4 uses the first zero-crossing (+→−) after plate.</div>
        <div id="plot-deriv2" style="height: 240px;"></div>
      </div>
    </section>
  </div>
</main>
<script>
const DATA = __PAYLOAD__;
document.getElementById('sg-info').textContent = `${DATA.sg_window} samples, polyorder ${DATA.sg_poly}`;

const COLORS = {
  manual:     '#2ca02c',
  plate:      '#0369a1',
  signal:     '#1a1d24',
  derivative: '#5c6a7a',
};
const VARIANTS_CFG = DATA.variants;
const VARIANT_KEYS = VARIANTS_CFG.map(v => v.key);
// Which variants are toggled ON (default: all)
const variantVisible = Object.fromEntries(VARIANT_KEYS.map(k => [k, true]));

// --- Build the sidebar tree ---------------------------------------------
const sidebar = document.getElementById('sidebar');

// Look up map: (chip_tag, well_id) -> well payload
const wellIndex = new Map();

function _statusForWell(w) {
  if (w.is_no_amp) return {text: 'no-amp', ok: false};
  // Show V1 TTP as the primary in the sidebar (compact)
  const v1 = w.variants && w.variants.closest_peak_d1;
  if (!v1 || v1.ttp_min == null) return {text: 'no-peak', ok: false};
  return {text: `V1 ${v1.ttp_min.toFixed(2)}`, ok: true};
}

for (const ds of DATA.datasets) {
  const block = document.createElement('div');
  block.className = 'dataset-block';

  const title = document.createElement('div');
  title.className = 'dataset-title';
  title.style.background = ds.colour;
  title.textContent = ds.label;
  block.appendChild(title);

  for (const chip of ds.chips) {
    const cblock = document.createElement('div');
    cblock.className = 'chip-block';

    const ctitle = document.createElement('div');
    ctitle.className = 'chip-title';
    ctitle.textContent = chip.chip_label;
    cblock.appendChild(ctitle);

    if (chip.notes_top) {
      const note = document.createElement('div');
      note.className = 'chip-notes';
      note.textContent = chip.notes_top;
      cblock.appendChild(note);
    }

    const list = document.createElement('div');
    list.className = 'well-list';
    if (!chip.wells || chip.wells.length === 0) {
      const ph = document.createElement('div');
      ph.className = 'well-cell placeholder';
      ph.textContent = '(no wells — awaiting mapping)';
      list.appendChild(ph);
    } else {
      for (const w of chip.wells) {
        const cell = document.createElement('div');
        const status = _statusForWell(w);
        const classes = ['well-cell'];
        if (w.plate_ttp_min == null) classes.push('no-plate');
        cell.className = classes.join(' ');
        cell.dataset.chip = chip.chip_tag;
        cell.dataset.well = String(w.well_id);
        cell.innerHTML = `
          <span class="wl-label">w${w.well_id} ${w.well_label}</span>
          <span class="wl-status">${status.text}</span>
        `;
        cell.addEventListener('click', () => selectWell(chip.chip_tag, w.well_id));
        list.appendChild(cell);
        wellIndex.set(`${chip.chip_tag}__${w.well_id}`, {well: w, chip});
      }
    }
    cblock.appendChild(list);
    block.appendChild(cblock);
  }
  sidebar.appendChild(block);
}

function _vline(x, color, dash) {
  return {type: 'line', xref: 'x', yref: 'paper',
          x0: x, x1: x, y0: 0, y1: 1,
          line: {color, dash, width: 1.8}};
}

// Track the current well so toggle changes can re-render without re-selecting.
let _currentKey = null;

function _renderVariantToggles() {
  const el = document.getElementById('variant-toggles');
  el.innerHTML = '';
  const label0 = document.createElement('span');
  label0.style.color = 'var(--muted)';
  label0.textContent = 'show:';
  el.appendChild(label0);
  for (const cfg of VARIANTS_CFG) {
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = variantVisible[cfg.key];
    cb.addEventListener('change', () => {
      variantVisible[cfg.key] = cb.checked;
      if (_currentKey) {
        const [c, w] = _currentKey.split('|');
        selectWell(c, parseInt(w, 10));
      }
    });
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = cfg.colour;
    sw.style.width = '12px'; sw.style.height = '3px';
    sw.style.display = 'inline-block'; sw.style.marginRight = '2px';
    lab.appendChild(cb);
    lab.appendChild(sw);
    lab.appendChild(document.createTextNode(' ' + cfg.label));
    el.appendChild(lab);
  }
}

function _renderVariantTable(w) {
  const el = document.getElementById('variant-table');
  const rows = [];
  rows.push(`<tr><th>variant</th><th>TTP (min)</th><th>Δ vs manual</th><th>reason</th></tr>`);
  for (const cfg of VARIANTS_CFG) {
    const r = w.variants && w.variants[cfg.key];
    const ttp = (r && r.ttp_min != null) ? r.ttp_min.toFixed(2) : '—';
    let delta = '—';
    if (r && r.ttp_min != null && w.manual_ttp_min != null) {
      const d = r.ttp_min - w.manual_ttp_min;
      delta = (d >= 0 ? '+' : '') + d.toFixed(2);
    }
    const reason = (r && r.reason) ? r.reason : '—';
    rows.push(`<tr>
      <td><span class="swatch" style="background:${cfg.colour};"></span>${cfg.label}</td>
      <td class="ttp">${ttp}</td>
      <td class="ttp">${delta}</td>
      <td style="color:var(--muted); font-size:10.5px;">${reason}</td>
    </tr>`);
  }
  el.innerHTML = rows.join('');
}

function selectWell(chip_tag, well_id) {
  document.querySelectorAll('.well-cell.selected').forEach(el => el.classList.remove('selected'));
  const cell = document.querySelector(
    `.well-cell[data-chip="${chip_tag}"][data-well="${well_id}"]`);
  if (cell) cell.classList.add('selected');

  const entry = wellIndex.get(`${chip_tag}__${well_id}`);
  if (!entry) return;
  const {well: w, chip} = entry;
  _currentKey = `${chip_tag}|${well_id}`;

  // ---------- Info strip ----------
  const rows = [];
  rows.push(`<div class="row">
    <span><b>chip:</b> ${chip.chip_label} <small style="color: var(--muted)">(${chip.chip_tag})</small></span>
    <span><b>well:</b> ${w.well_id}</span>
    <span><b>label:</b> ${w.well_label}</span>
    <span><b>conc:</b> ${w.conc_str}</span>
  </div>`);
  rows.push(`<div class="row">
    <span><b>unit:</b> ${w.unit_note}</span>
  </div>`);
  const p = w.plate_ttp_min == null ? '—' : `${w.plate_ttp_min.toFixed(2)} min`;
  const m = w.manual_ttp_min == null ? '—' : `${w.manual_ttp_min.toFixed(2)} min`;
  rows.push(`<div class="row">
    <span><b>plate TTP:</b> ${p}</span>
    <span><b>manual TTP:</b> ${m}</span>
  </div>`);
  if (w.plate_info) {
    const pi = w.plate_info;
    const raw = pi.raw_cqs ? pi.raw_cqs.map(v => v.toFixed(2)).join(', ') : '—';
    rows.push(`<div class="row">
      <span><b>plate conc label:</b> ${pi.kp_label ?? '—'}</span>
      <span><b>Cq mean (n=${pi.n_positive ?? 0}/${pi.n_replicates ?? 0}):</b>
        ${pi.cq_mean == null ? '—' : pi.cq_mean.toFixed(3)}</span>
      <span><b>raw Cqs:</b> [${raw}]</span>
    </div>`);
    if (pi.assignment_note) {
      rows.push(`<div class="row"><span style="color: var(--muted); font-style: italic;">
        ${pi.assignment_note}</span></div>`);
    }
  }
  document.getElementById('info-strip').innerHTML = rows.join('');

  _renderVariantTable(w);

  // ---------- shared overlay helpers ----------
  function _plateAndManualLines() {
    const shapes = [];
    const anns = [];
    if (w.plate_ttp_min != null) {
      shapes.push(_vline(w.plate_ttp_min, COLORS.plate, 'dot'));
      anns.push({x: w.plate_ttp_min, y: 1, xref: 'x', yref: 'paper',
                 text: `plate ${w.plate_ttp_min.toFixed(2)}`, showarrow: false,
                 font: {size: 10, color: COLORS.plate}, xanchor: 'right', yanchor: 'top'});
    }
    if (w.manual_ttp_min != null) {
      shapes.push(_vline(w.manual_ttp_min, COLORS.manual, 'dash'));
      anns.push({x: w.manual_ttp_min, y: 0.90, xref: 'x', yref: 'paper',
                 text: `manual ${w.manual_ttp_min.toFixed(2)}`, showarrow: false,
                 font: {size: 10, color: COLORS.manual}, xanchor: 'left', yanchor: 'top'});
    }
    return {shapes, anns};
  }

  function _variantLines(ystart) {
    const shapes = [];
    const anns = [];
    let yPos = ystart;
    for (const cfg of VARIANTS_CFG) {
      if (!variantVisible[cfg.key]) continue;
      const r = w.variants && w.variants[cfg.key];
      if (!r || r.ttp_min == null) continue;
      shapes.push(_vline(r.ttp_min, cfg.colour, 'dash'));
      anns.push({x: r.ttp_min, y: yPos, xref: 'x', yref: 'paper',
                 text: `${cfg.key} ${r.ttp_min.toFixed(2)}`,
                 showarrow: false, font: {size: 9.5, color: cfg.colour},
                 xanchor: 'left', yanchor: 'top'});
      yPos -= 0.07;
    }
    return {shapes, anns};
  }

  const t = w.time_min;

  // ---------- Signal panel ----------
  {
    const overlays = _plateAndManualLines();
    const vlines = _variantLines(0.82);
    Plotly.newPlot('plot-signal', [
      {x: t, y: w.signal, mode: 'lines',
       line: {color: COLORS.signal, width: 2}, name: 'smoothed signal'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 320,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'signal (a.u.)', gridcolor: 'rgba(120,120,120,0.15)'},
      shapes: overlays.shapes.concat(vlines.shapes),
      annotations: overlays.anns.concat(vlines.anns),
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }

  // ---------- First derivative panel ----------
  {
    const overlays = _plateAndManualLines();
    // Show V1 + V6 + V7 lines on d1 panel (all operate on 1st derivative)
    const shapes = overlays.shapes.slice();
    const anns = overlays.anns.slice();
    let yPos = 0.82;
    for (const key of ['closest_peak_d1', 'cy0_after_plate', 'thr_deriv_after_plate']) {
      if (!variantVisible[key]) continue;
      const cfg = VARIANTS_CFG.find(v => v.key === key);
      const r = w.variants && w.variants[key];
      if (!r || r.ttp_min == null) continue;
      shapes.push(_vline(r.ttp_min, cfg.colour, 'dash'));
      anns.push({x: r.ttp_min, y: yPos, xref: 'x', yref: 'paper',
                 text: `${key} ${r.ttp_min.toFixed(2)}`,
                 showarrow: false, font: {size: 9.5, color: cfg.colour},
                 xanchor: 'left', yanchor: 'top'});
      yPos -= 0.09;
    }
    Plotly.newPlot('plot-deriv', [
      {x: t, y: w.derivative, mode: 'lines',
       line: {color: COLORS.derivative, width: 1.6}, name: 'd/dt smoothed'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 240,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'd(signal)/dt', gridcolor: 'rgba(120,120,120,0.15)',
              zerolinecolor: 'rgba(120,120,120,0.3)'},
      shapes, annotations: anns,
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }

  // ---------- Second derivative panel ----------
  {
    const overlays = _plateAndManualLines();
    const shapes = overlays.shapes.slice();
    const anns = overlays.anns.slice();
    let yPos = 0.82;
    for (const key of ['zero_cross_d2']) {
      if (!variantVisible[key]) continue;
      const cfg = VARIANTS_CFG.find(v => v.key === key);
      const r = w.variants && w.variants[key];
      if (!r || r.ttp_min == null) continue;
      shapes.push(_vline(r.ttp_min, cfg.colour, 'dash'));
      anns.push({x: r.ttp_min, y: yPos, xref: 'x', yref: 'paper',
                 text: `${key} ${r.ttp_min.toFixed(2)}`,
                 showarrow: false, font: {size: 9.5, color: cfg.colour},
                 xanchor: 'left', yanchor: 'top'});
      yPos -= 0.09;
    }
    Plotly.newPlot('plot-deriv2', [
      {x: t, y: w.second_derivative, mode: 'lines',
       line: {color: '#7a3c8a', width: 1.4}, name: 'd²/dt²'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 240,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'd²(signal)/dt²', gridcolor: 'rgba(120,120,120,0.15)',
              zerolinecolor: 'rgba(120,120,120,0.4)'},
      shapes, annotations: anns,
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }
}

_renderVariantToggles();

// Auto-select the first available well
(function autoSelect() {
  for (const ds of DATA.datasets) {
    for (const chip of ds.chips) {
      if (chip.wells && chip.wells.length > 0) {
        selectWell(chip.chip_tag, chip.wells[0].well_id);
        return;
      }
    }
  }
})();
</script>
</body>
</html>
"""


def main() -> None:
    payload = _build_payload()
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUT_HTML.write_text(html)
    print(f"[ok] wrote {OUT_HTML}  ({len(html):,} chars)")
    for ds in payload["datasets"]:
        print(f"  {ds['label']}")
        for chip in ds["chips"]:
            n = len(chip.get("wells", []))
            print(f"    {chip['chip_label']}: {n} wells")


if __name__ == "__main__":
    main()
