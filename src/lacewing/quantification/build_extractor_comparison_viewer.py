"""Viewer · computing-extractor comparison interactive viewer. [Cat A] Referenced in presentation.

Build the extractor-comparison viewer HTML.

Same LHS/RHS layout as systematic_ttp_viewer.html but the overlays show
the extractors' TTP picks so we can see WHY they miss the ground truth.

Overlays per well:
  Ground truth (from stages json): cy0_after_plate (V6) — matches the
    per-pixel regression target used by every ML framing in the report.
  Computing-based methods (fresh compute on the well-mean signal):
    - Threshold-derivative TTP (Moser 2022 / deployed Lacewing)
    - Cy0 tangent-at-inflection
    - SDM (second-derivative maximum)
  ML method:
    - F-B sliding-window classifier — per-well TTP is the median across
      the well's amp-positive pixels of predictions.npz['ttp_pred_min'],
      then median across the 5 seeds. Only shown for wells that were the
      held-out fold's amp-positive wells.

Plots (each panel is a Plotly figure):
  1) Smoothed well-mean signal + all extractor vertical lines
  2) First derivative + TTP-family markers (TTP, Cy0)
  3) Second derivative + SDM marker
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lacewing.quantification.methods.cy0 import extract_cy0
from lacewing.quantification.methods.sdm import extract_sdm
from lacewing.quantification.methods.ttp_threshold_derivative import extract_ttp
from lacewing.quantification.systematic_ttp import (
    second_derivative,
    compute_all_variants,
    DEFAULT_SG_WINDOW,
    DEFAULT_SG_POLY,
)
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[2]
CONC_OUT = Path(__file__).resolve().parent / "conc_data" / "output"
CONC_STAGES = [
    CONC_OUT / "conc_260812_KP_01_stages.json",
    CONC_OUT / "conc_260813_KP_DDM_02_stages.json",
    CONC_OUT / "conc_260820_KP_DDM_03_stages.json",
    CONC_OUT / "conc_260823_KP_DDM_04_stages.json",
    CONC_OUT / "conc_260827_KP_DDM_05_stages.json",
]

FB_RESULTS_ROOT = (
    LACEWING_PKG_DIR / "quantification" / "methods" / "results"
)
FB_DIR_STEM = "p1_fb_slide_cls_w120_stride10_spatA3_bb_unet_kthr0.7_conc_loco"

OUT_DIR = LACEWING_PKG_DIR / "quantification" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = OUT_DIR / "extractor_comparison_viewer.html"


EXTRACTORS = [
    {"key": "gt_cy0_after_plate", "label": "Ground truth (V6 cy0_after_plate)", "colour": "#2ca02c", "family": "gt"},
    {"key": "ttp_thr_deriv",      "label": "Threshold-derivative TTP",           "colour": "#008b8b", "family": "d1"},
    {"key": "cy0",                 "label": "Cy0 (tangent intercept)",            "colour": "#ff8c00", "family": "d1"},
    {"key": "sdm",                 "label": "SDM (second-derivative max)",       "colour": "#8a2be2", "family": "d2"},
    {"key": "fb_ml",               "label": "F-B sliding-window classifier",     "colour": "#b30000", "family": "ml"},
]


def _finite(x):
    return [None if v is None or not np.isfinite(v) else float(v) for v in x]


def _load_fb_per_well_ttps(fold_index: int) -> dict[int, float]:
    """Median-over-seeds of median-over-pixels F-B TTP for each amp-positive well
    on the fold's held-out chip.

    fold_index: 1..5 (matches conc_loco{N} directory naming).
    Returns {well_id: ttp_min} for amp+ wells only (wells 0-7).
    """
    per_seed_per_well: dict[int, list[float]] = {}
    for seed in range(5):
        seed_dir = FB_RESULTS_ROOT / f"{FB_DIR_STEM}{fold_index}" / f"seed{seed}"
        pred = seed_dir / "predictions.npz"
        lab = seed_dir / "labels.npz"
        if not (pred.exists() and lab.exists()):
            continue
        P = np.load(pred, allow_pickle=False)
        L = np.load(lab, allow_pickle=False)
        ttp = P["ttp_pred_min"]
        well = L["well_id"]
        true_ttp = L["ttp_true_min"]
        # amp+ wells are the ones with finite ttp_true_min
        for w in np.unique(well):
            m = (well == w) & np.isfinite(ttp) & np.isfinite(true_ttp)
            if not m.any():
                continue
            per_seed_per_well.setdefault(int(w), []).append(float(np.median(ttp[m])))
    return {w: float(np.median(vals)) for w, vals in per_seed_per_well.items() if vals}


def _compute_extractors(sig: np.ndarray, t: np.ndarray, plate_ttp) -> dict:
    """Run the three computing-based extractors + the systematic V6 ground truth
    on the well-mean smoothed signal. Return {extractor_key: {ttp_min, reason}}."""
    out: dict[str, dict] = {}

    # Ground truth = V6 cy0_after_plate (the label every ML framing consumes)
    variants = compute_all_variants(sig, t, plate_ttp_min=plate_ttp)
    v6 = variants["cy0_after_plate"]
    out["gt_cy0_after_plate"] = {
        "ttp_min": v6.ttp_min, "reason": v6.reason,
    }

    # Computing-based method 1: threshold-derivative TTP (deployed Lacewing)
    try:
        ttp_val, peak_val = extract_ttp(t, sig)
    except Exception as e:
        ttp_val, peak_val = float("nan"), float("nan")
    out["ttp_thr_deriv"] = {
        "ttp_min": None if not np.isfinite(ttp_val) else float(ttp_val),
        "reason": (
            f"0.4-threshold on smoothed d1; peak at {peak_val:.2f} min"
            if np.isfinite(peak_val) else "no d1 peak"
        ),
    }

    # Computing-based method 2: Cy0
    try:
        cy0_val = extract_cy0(t, sig)
    except Exception:
        cy0_val = float("nan")
    out["cy0"] = {
        "ttp_min": None if not np.isfinite(cy0_val) else float(cy0_val),
        "reason": "tangent-at-inflection intercept",
    }

    # Computing-based method 3: SDM
    try:
        sdm_val = extract_sdm(t, sig)
    except Exception:
        sdm_val = float("nan")
    out["sdm"] = {
        "ttp_min": None if not np.isfinite(sdm_val) else float(sdm_val),
        "reason": "argmax of second derivative",
    }

    return out


def _prepare_well(chip_tag: str, well: dict, fb_per_well: dict[int, float]) -> dict:
    """Build the per-well payload with signal + extractor overlays."""
    plate_info = well.get("plate") or {}
    plate_ttp = plate_info.get("takeoff_min")

    t_full = np.asarray(well["time_min"], dtype=np.float64)
    sig = np.asarray(well["smoothed_signal"], dtype=np.float64)
    mask = np.isfinite(t_full) & np.isfinite(sig)
    t = t_full[mask]
    sig = sig[mask]

    # Compute derivatives for plotting (V1's smoothed d1 + our own d2)
    variants = compute_all_variants(sig, t, plate_ttp_min=plate_ttp)
    v1 = variants["closest_peak_d1"]
    _, d2 = second_derivative(sig, t)
    if d2 is None:
        d2 = np.zeros_like(v1.first_derivative)

    extractors = _compute_extractors(sig, t, plate_ttp)

    # F-B is only available for amp+ wells on the held-out fold (which is
    # always the "current chip is being evaluated" case in a per-chip viewer);
    # for the KP LOCO evaluation each fold produces predictions on its own
    # held-out chip's amp+ wells only.
    fb_ttp = fb_per_well.get(int(well["well"]))
    extractors["fb_ml"] = {
        "ttp_min": None if fb_ttp is None else float(fb_ttp),
        "reason": (
            "median across amp+ pixels of median across 5 seeds"
            if fb_ttp is not None
            else "not applicable (well 8/9 or excluded from F-B run)"
        ),
    }

    return {
        "well_id": int(well["well"]),
        "well_label": well["label"],
        "log10_conc": well.get("log10_conc"),
        "conc_str": _format_conc(well.get("log10_conc"), well["label"]),
        "n_active": well.get("n_active_raw"),
        "n_kept": well.get("n_kept"),
        "time_min": _finite(t.tolist()),
        "signal": _finite(v1.smoothed_signal.tolist()),
        "derivative": _finite(v1.first_derivative.tolist()),
        "second_derivative": _finite(d2.tolist()),
        "plate_ttp_min": None if plate_ttp is None else float(plate_ttp),
        "extractors": extractors,
        "plate_info": {
            "kp_label": plate_info.get("kp_label"),
            "cq_mean": plate_info.get("cq_mean"),
            "n_positive": plate_info.get("n_positive"),
            "n_replicates": plate_info.get("n_replicates"),
            "raw_cqs": plate_info.get("raw_cqs"),
            "assignment_note": plate_info.get("assignment_note"),
        },
    }


def _format_conc(log10_conc, well_label: str) -> str:
    if log10_conc is None or not np.isfinite(log10_conc):
        return f"{well_label} — no concentration"
    return f"10^{log10_conc:.2g} copies  ({well_label})"


def _build_payload() -> dict:
    chips: list[dict] = []
    # Map each conc_stages file to its LOCO fold index (matches conc_ei script order):
    # conc_260812_KP_01 -> fold 1, ..., conc_260827_KP_DDM_05 -> fold 5
    for fold_idx, path in enumerate(CONC_STAGES, start=1):
        if not path.exists():
            continue
        stages = json.loads(path.read_text())
        fb_per_well = _load_fb_per_well_ttps(fold_idx)
        wells = [_prepare_well(stages["chip_tag"], w, fb_per_well) for w in stages["wells"]]
        chips.append({
            "chip_tag": stages["chip_tag"],
            "chip_label": stages.get("chip_label") or stages["chip_tag"],
            "fold_index": fold_idx,
            "n_fb_wells": len(fb_per_well),
            "wells": wells,
        })

    return {
        "generated_at": "2026-09-03",
        "sg_window": DEFAULT_SG_WINDOW,
        "sg_poly": DEFAULT_SG_POLY,
        "extractors": EXTRACTORS,
        "chips": chips,
        "note": (
            "Ground truth is V6 cy0_after_plate (the systematic labelling recipe "
            "the report treats as ground truth). Computing-based methods "
            "(threshold-derivative TTP, Cy0, SDM) are run on the well-mean "
            "smoothed signal — same trace shown in the plot. F-B sliding-window "
            "classifier per-well TTP is the median across amp+ pixels of the "
            "median across 5 LOCO seeds, computed only on the fold where each "
            "chip is the held-out test chip."
        ),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extractor comparison viewer — KP dataset</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --panel: #f6f7fb; --border: #d5d8e0;
    --fg: #1a1d24; --muted: #666; --accent: #1f3a6b;
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
  .chip-block { margin: 6px 0 10px 0; }
  .chip-title {
    font-size: 12px; color: var(--accent); font-weight: 600;
    margin-bottom: 4px;
  }
  .chip-notes {
    color: var(--muted); font-size: 10.5px; font-style: italic;
    margin-bottom: 4px; padding: 0 2px;
  }
  .well-list { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
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
  .well-cell.no-gt { border-color: var(--warn); }

  .info-strip {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; margin-bottom: 10px;
    font-family: 'SF Mono','Consolas',monospace; font-size: 12px;
  }
  .info-strip .row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 3px; }
  .info-strip .row > span { color: var(--muted); }
  .info-strip .row > span > b { color: var(--fg); font-weight: 600; }

  .plot-panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px; margin-bottom: 12px;
  }
  .plot-panel h4 { margin: 0 0 4px 0; font-size: 12px; color: var(--accent); }
  .plot-panel .caption { color: var(--muted); font-size: 11px; margin-bottom: 4px; }

  .extractor-table {
    width: 100%; border-collapse: collapse; font-size: 11px;
    font-family: 'SF Mono','Consolas',monospace;
  }
  .extractor-table th, .extractor-table td {
    padding: 3px 8px; text-align: left; border-bottom: 1px solid var(--border);
  }
  .extractor-table th { color: var(--muted); font-weight: 600; }
  .extractor-table td.ttp { text-align: right; }
  .extractor-table td .swatch {
    display: inline-block; width: 12px; height: 3px; margin-right: 6px;
    vertical-align: middle; border-radius: 2px;
  }
  .extractor-toggles { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
  .extractor-toggles label { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; cursor: pointer; }
  .extractor-toggles input { margin: 0; }
</style>
</head>
<body>
<header>
  <h1>Extractor comparison viewer — KP dataset</h1>
  <div class="subtitle">
    Ground truth (green) is the V6 cy0-after-plate systematic label. Computing-based methods
    (threshold-derivative TTP, Cy0, SDM) run on the well-mean smoothed signal shown below.
    F-B (red) is the sliding-window classifier's median per-well TTP over 5 seeds on the
    fold where each chip is the held-out test chip. SG smoothing window = <span id="sg-info"></span>.
  </div>
</header>
<main>
  <div class="layout">
    <aside class="sidebar" id="sidebar"></aside>
    <section>
      <div class="info-strip" id="info-strip">Select a well from the sidebar to display its curves.</div>
      <div class="plot-panel">
        <h4>Per-well TTP predictions vs ground truth</h4>
        <div class="extractor-toggles" id="extractor-toggles"></div>
        <table class="extractor-table" id="extractor-table"></table>
      </div>
      <div class="plot-panel">
        <h4>Well-mean signal (smoothed) + all overlays</h4>
        <div class="caption">Solid line = Savitzky-Golay smoothed chip well-mean. Vertical lines = ground truth (green) and each extractor's TTP pick.</div>
        <div id="plot-signal" style="height: 320px;"></div>
      </div>
      <div class="plot-panel">
        <h4>First derivative + d1-family extractors</h4>
        <div class="caption">d(signal)/dt. Threshold-derivative TTP walks back from the d1 peak; Cy0 is the tangent-at-inflection intercept.</div>
        <div id="plot-deriv" style="height: 240px;"></div>
      </div>
      <div class="plot-panel">
        <h4>Second derivative + SDM extractor</h4>
        <div class="caption">d²(signal)/dt². SDM fires at the second-derivative maximum on the rising flank.</div>
        <div id="plot-deriv2" style="height: 240px;"></div>
      </div>
    </section>
  </div>
</main>
<script>
const DATA = __PAYLOAD__;
document.getElementById('sg-info').textContent = `${DATA.sg_window} samples, polyorder ${DATA.sg_poly}`;

const EXTRACTORS_CFG = DATA.extractors;
const EXTRACTOR_KEYS = EXTRACTORS_CFG.map(e => e.key);
const extractorVisible = Object.fromEntries(EXTRACTOR_KEYS.map(k => [k, true]));

// (chip_tag, well_id) -> {well, chip}
const wellIndex = new Map();

function _statusForWell(w) {
  const gt = w.extractors && w.extractors.gt_cy0_after_plate;
  if (!gt || gt.ttp_min == null) return {text: 'no-gt', ok: false};
  return {text: `gt ${gt.ttp_min.toFixed(2)}`, ok: true};
}

const sidebar = document.getElementById('sidebar');
for (const chip of DATA.chips) {
  const cblock = document.createElement('div');
  cblock.className = 'chip-block';

  const ctitle = document.createElement('div');
  ctitle.className = 'chip-title';
  ctitle.textContent = chip.chip_label;
  cblock.appendChild(ctitle);

  const note = document.createElement('div');
  note.className = 'chip-notes';
  note.textContent = `LOCO fold ${chip.fold_index}. F-B predictions available for ${chip.n_fb_wells} amp+ wells.`;
  cblock.appendChild(note);

  const list = document.createElement('div');
  list.className = 'well-list';
  for (const w of chip.wells) {
    const cell = document.createElement('div');
    const status = _statusForWell(w);
    const classes = ['well-cell'];
    const gt = w.extractors && w.extractors.gt_cy0_after_plate;
    if (!gt || gt.ttp_min == null) classes.push('no-gt');
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
  cblock.appendChild(list);
  sidebar.appendChild(cblock);
}

function _vline(x, color, dash) {
  return {type: 'line', xref: 'x', yref: 'paper',
          x0: x, x1: x, y0: 0, y1: 1,
          line: {color, dash, width: 1.8}};
}

let _currentKey = null;

function _renderExtractorToggles() {
  const el = document.getElementById('extractor-toggles');
  el.innerHTML = '';
  const label0 = document.createElement('span');
  label0.style.color = 'var(--muted)';
  label0.textContent = 'show:';
  el.appendChild(label0);
  for (const cfg of EXTRACTORS_CFG) {
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = extractorVisible[cfg.key];
    cb.addEventListener('change', () => {
      extractorVisible[cfg.key] = cb.checked;
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

function _renderExtractorTable(w) {
  const el = document.getElementById('extractor-table');
  const rows = [];
  rows.push(`<tr><th>extractor</th><th>TTP (min)</th><th>Δ vs ground truth</th><th>notes</th></tr>`);
  const gt = w.extractors && w.extractors.gt_cy0_after_plate;
  const gt_ttp = (gt && gt.ttp_min != null) ? gt.ttp_min : null;
  for (const cfg of EXTRACTORS_CFG) {
    const r = w.extractors && w.extractors[cfg.key];
    const ttp = (r && r.ttp_min != null) ? r.ttp_min.toFixed(2) : '—';
    let delta = '—';
    if (r && r.ttp_min != null && gt_ttp != null && cfg.key !== 'gt_cy0_after_plate') {
      const d = r.ttp_min - gt_ttp;
      delta = (d >= 0 ? '+' : '') + d.toFixed(2);
    } else if (cfg.key === 'gt_cy0_after_plate') {
      delta = '(reference)';
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

  // Info strip
  const rows = [];
  rows.push(`<div class="row">
    <span><b>chip:</b> ${chip.chip_label} <small style="color: var(--muted)">(${chip.chip_tag})</small></span>
    <span><b>well:</b> ${w.well_id}</span>
    <span><b>label:</b> ${w.well_label}</span>
    <span><b>conc:</b> ${w.conc_str}</span>
  </div>`);
  const p = w.plate_ttp_min == null ? '—' : `${w.plate_ttp_min.toFixed(2)} min`;
  rows.push(`<div class="row"><span><b>plate anchor:</b> ${p}</span></div>`);
  if (w.plate_info && w.plate_info.assignment_note) {
    rows.push(`<div class="row"><span style="color: var(--muted); font-style: italic;">
      ${w.plate_info.assignment_note}</span></div>`);
  }
  document.getElementById('info-strip').innerHTML = rows.join('');

  _renderExtractorTable(w);

  // Helper: vertical lines for the extractors that hit this panel's y-axis
  function _extractorLines(keys, ystart) {
    const shapes = [];
    const anns = [];
    let yPos = ystart;
    for (const key of keys) {
      if (!extractorVisible[key]) continue;
      const cfg = EXTRACTORS_CFG.find(e => e.key === key);
      const r = w.extractors && w.extractors[key];
      if (!r || r.ttp_min == null) continue;
      shapes.push(_vline(r.ttp_min, cfg.colour, key === 'gt_cy0_after_plate' ? 'solid' : 'dash'));
      anns.push({x: r.ttp_min, y: yPos, xref: 'x', yref: 'paper',
                 text: `${key} ${r.ttp_min.toFixed(2)}`,
                 showarrow: false, font: {size: 9.5, color: cfg.colour},
                 xanchor: 'left', yanchor: 'top'});
      yPos -= 0.07;
    }
    return {shapes, anns};
  }

  const t = w.time_min;

  // Signal panel — show ALL extractors
  {
    const lines = _extractorLines(EXTRACTOR_KEYS, 0.95);
    Plotly.newPlot('plot-signal', [
      {x: t, y: w.signal, mode: 'lines',
       line: {color: '#1a1d24', width: 2}, name: 'smoothed signal'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 320,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'signal (a.u.)', gridcolor: 'rgba(120,120,120,0.15)'},
      shapes: lines.shapes, annotations: lines.anns,
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }

  // First derivative panel — GT + TTP + Cy0 (d1-family)
  {
    const lines = _extractorLines(['gt_cy0_after_plate', 'ttp_thr_deriv', 'cy0', 'fb_ml'], 0.95);
    Plotly.newPlot('plot-deriv', [
      {x: t, y: w.derivative, mode: 'lines',
       line: {color: '#5c6a7a', width: 1.6}, name: 'd/dt smoothed'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 240,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'd(signal)/dt', gridcolor: 'rgba(120,120,120,0.15)',
              zerolinecolor: 'rgba(120,120,120,0.3)'},
      shapes: lines.shapes, annotations: lines.anns,
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }

  // Second derivative panel — GT + SDM + F-B
  {
    const lines = _extractorLines(['gt_cy0_after_plate', 'sdm', 'fb_ml'], 0.95);
    Plotly.newPlot('plot-deriv2', [
      {x: t, y: w.second_derivative, mode: 'lines',
       line: {color: '#7a3c8a', width: 1.4}, name: 'd²/dt²'},
    ], {
      margin: {l: 55, r: 10, t: 10, b: 40}, height: 240,
      xaxis: {title: 'time (min)', gridcolor: 'rgba(120,120,120,0.15)'},
      yaxis: {title: 'd²(signal)/dt²', gridcolor: 'rgba(120,120,120,0.15)',
              zerolinecolor: 'rgba(120,120,120,0.4)'},
      shapes: lines.shapes, annotations: lines.anns,
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      font: {color: getComputedStyle(document.body).color},
      showlegend: false,
    }, {responsive: true, displaylogo: false});
  }
}

_renderExtractorToggles();

// Auto-select the first available well
(function autoSelect() {
  for (const chip of DATA.chips) {
    if (chip.wells && chip.wells.length > 0) {
      selectWell(chip.chip_tag, chip.wells[0].well_id);
      return;
    }
  }
})();
</script>
</body>
</html>
"""


def main() -> None:
    payload = _build_payload()
    html = HTML_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, separators=(",", ":"))
    )
    OUT_HTML.write_text(html)
    print(f"[ok] wrote {OUT_HTML}  ({len(html):,} chars)")
    for chip in payload["chips"]:
        n_wells = len(chip.get("wells", []))
        n_fb = chip.get("n_fb_wells", 0)
        print(f"  {chip['chip_label']}: {n_wells} wells ({n_fb} with F-B TTP)")


if __name__ == "__main__":
    main()
