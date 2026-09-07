"""Per-pixel-overlay viewer for the 5 Concentration Data chips.

For each (chip, well): plots every surviving pixel trace as a thin
translucent line, with the well-mean drawn on top in bold. Vertical
markers show the cy0_after_plate anchor (per-well systematic TTP)
and the plate qLAMP TTP.

Purpose: visually judge how much per-pixel timing variation actually
exists in the data after the deployed preprocessing pipeline.

Output:
  Analysis/quantification/output/per_pixel_overlay_viewer.html
"""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from lacewing.quantification.conc_data.process_conc_chips import (  # noqa: E402
    _build_chip_configs,
    PLATE_FEATURES_JSON,
    _LOG10_TO_KP,
)
from lacewing.quantification.chip_pipeline import process_chip as pipe  # noqa: E402
from lacewing.quantification.systematic_ttp import compute_all_variants  # noqa: E402
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
OUT_HTML = LACEWING_PKG_DIR / "quantification" / "output" / "per_pixel_overlay_viewer.html"


def _plate_ttp_for_well(cfg, plate_per_conc, w_idx):
    log10 = cfg.well_log10.get(w_idx)
    if log10 is None or not np.isfinite(log10):
        return None
    kp_label = _LOG10_TO_KP.get(float(log10))
    if kp_label is None:
        return None
    entry = plate_per_conc.get(kp_label)
    if entry is None:
        return None
    return float(entry.get("ttp_min"))


def _decimate_axis(x, n_target=300):
    """Downsample a 1D array to at most n_target points (index stride)."""
    if len(x) <= n_target:
        return x
    step = int(np.ceil(len(x) / n_target))
    return x[::step]


def _decimate_2d(X, n_target=300):
    """Downsample a 2D array (n_pixels, T) along axis 1."""
    if X.shape[1] <= n_target:
        return X
    step = int(np.ceil(X.shape[1] / n_target))
    return X[:, ::step]


def process_chip(cfg):
    print(f"\nProcessing {cfg.chip_tag} ...")
    if not cfg.chip_dir.exists():
        print(f"  [SKIP] chip_dir not found: {cfg.chip_dir}")
        return None

    plate_data = json.loads(PLATE_FEATURES_JSON.read_text())
    plate_per_conc = plate_data[cfg.plate_key]["per_conc_kp"]

    old = (pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN)
    try:
        pipe.WELL_LABELS = cfg.well_labels
        pipe.WELL_LOG10  = cfg.well_log10
        pipe.NTC_WELLS   = tuple(cfg.ntc_wells)
        pipe.TRIM_SEARCH_MAX_MIN = cfg.trim_search_max_min
        result = pipe.process_chip(cfg.chip_dir, save_per_pixel=True)
    finally:
        pipe.WELL_LABELS, pipe.WELL_LOG10, pipe.NTC_WELLS, pipe.TRIM_SEARCH_MAX_MIN = old

    wells_payload = []
    for w in result.wells:
        plate_ttp = _plate_ttp_for_well(cfg, plate_per_conc, w.well)
        cy0_after_plate = None
        if plate_ttp is not None:
            variants = compute_all_variants(
                w.mean_after_spat, w.time_min, plate_ttp_min=plate_ttp,
            )
            v = variants["cy0_after_plate"].ttp_min
            if np.isfinite(v):
                cy0_after_plate = float(v)

        # Decimate for browser payload (300 samples max per pixel).
        t = _decimate_axis(w.time_min, 300)
        wm = _decimate_axis(w.mean_after_spat, 300)
        if w.per_pixel_qc is not None and w.per_pixel_qc.shape[0] > 0:
            px = _decimate_2d(w.per_pixel_qc, 300).astype(np.float32)
            n_pix = int(px.shape[0])
            # Pack float32 array as base64-encoded gzip'd bytes for compact transport.
            # Also send the flat shape so JS can unpack.
            raw = px.tobytes(order="C")
            gz = gzip.compress(raw, compresslevel=9)
            px_b64 = base64.b64encode(gz).decode("ascii")
            px_shape = list(px.shape)  # [n_pix, T]
        else:
            n_pix = 0
            px_b64 = ""
            px_shape = [0, 0]

        wells_payload.append({
            "well_idx": int(w.well),
            "well_label": w.label,
            "log10_conc": (None if not np.isfinite(w.log10_conc) else float(w.log10_conc)),
            "n_pix": n_pix,
            "time_min": [float(x) for x in t.tolist()],
            "well_mean": [float(x) for x in wm.tolist()],
            "pixels_b64_gz": px_b64,
            "pixels_shape": px_shape,
            "plate_ttp_min": plate_ttp,
            "cy0_after_plate_min": cy0_after_plate,
        })

    return {
        "chip_tag": cfg.chip_tag,
        "chip_label": cfg.chip_label,
        "wells": wells_payload,
    }


HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Per-pixel overlay — Concentration Data</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }
  #app { display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
  #lhs { border-right: 1px solid #ddd; overflow-y: auto; padding: 10px; background: #f7f7f9; }
  #rhs { padding: 12px; overflow-y: auto; }
  .chip-header { font-weight: 600; margin: 12px 0 6px 0; color: #333; }
  .well-item { padding: 4px 8px; margin: 2px 0; cursor: pointer; border-radius: 4px; font-size: 12px; }
  .well-item:hover { background: #e6e6ea; }
  .well-item.active { background: #4c72b0; color: white; }
  .well-item .conc { color: #888; float: right; font-family: monospace; }
  .well-item.active .conc { color: #ddd; }
  h1 { font-size: 15px; margin: 0 0 4px 0; }
  .meta { font-size: 12px; color: #666; margin: 4px 0 12px 0; }
  #plot { height: calc(100vh - 100px); }
</style>
</head>
<body>
<div id="app">
  <div id="lhs"></div>
  <div id="rhs">
    <h1 id="title">Select a well</h1>
    <div class="meta" id="meta"></div>
    <div id="plot"></div>
  </div>
</div>
<script>
const DATA = __DATA__;

function unpackPixels(w) {
  if (!w.pixels_b64_gz || w.pixels_shape[0] === 0) return null;
  const gz = Uint8Array.from(atob(w.pixels_b64_gz), c => c.charCodeAt(0));
  const raw = pako.ungzip(gz);
  const [n, T] = w.pixels_shape;
  const flat = new Float32Array(raw.buffer, raw.byteOffset, n * T);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(flat.slice(i * T, (i + 1) * T));
  }
  return out;
}

function render(chipIdx, wellIdx) {
  const chip = DATA[chipIdx];
  const w = chip.wells[wellIdx];
  document.getElementById("title").textContent =
    chip.chip_label + " · " + w.well_label + " (well " + w.well_idx + ")";
  const meta = [];
  meta.push("n_pixels = " + w.n_pix);
  if (w.log10_conc !== null) meta.push("10^" + w.log10_conc + " copies");
  if (w.plate_ttp_min !== null)
    meta.push("plate qLAMP TTP = " + w.plate_ttp_min.toFixed(3) + " min");
  if (w.cy0_after_plate_min !== null)
    meta.push("cy0_after_plate = " + w.cy0_after_plate_min.toFixed(3) + " min");
  document.getElementById("meta").innerHTML = meta.join("&nbsp;&nbsp;·&nbsp;&nbsp;");

  const pixels = unpackPixels(w);
  const traces = [];
  const DIM_COLOR = "rgba(70,110,180,0.15)";
  const HL_COLOR  = "rgba(20,60,140,0.95)";
  if (pixels) {
    for (let i = 0; i < pixels.length; i++) {
      traces.push({
        x: w.time_min,
        y: Array.from(pixels[i]),
        mode: "lines",
        type: "scatter",           // scatter (not scattergl) so hover works reliably per trace
        line: { width: 0.6, color: DIM_COLOR },
        name: "pixel " + i,
        hovertemplate: "pixel " + i + "<br>t=%{x:.2f} min<br>v=%{y:.4f}<extra></extra>",
        showlegend: false,
      });
    }
  }
  traces.push({
    x: w.time_min,
    y: w.well_mean,
    mode: "lines",
    type: "scatter",
    line: { width: 2.2, color: "#111" },
    name: "well mean",
    hovertemplate: "well mean<br>t=%{x:.2f} min<br>v=%{y:.4f}<extra></extra>",
  });
  const shapes = [];
  if (w.plate_ttp_min !== null) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: w.plate_ttp_min, x1: w.plate_ttp_min, y0: 0, y1: 1,
      line: { color: "#c81c1c", width: 1.4, dash: "dash" },
    });
  }
  if (w.cy0_after_plate_min !== null) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: w.cy0_after_plate_min, x1: w.cy0_after_plate_min, y0: 0, y1: 1,
      line: { color: "#ff8c00", width: 1.4, dash: "dash" },
    });
  }
  const annotations = [];
  if (w.plate_ttp_min !== null) {
    annotations.push({
      x: w.plate_ttp_min, y: 1.0, xref: "x", yref: "paper", yshift: 12,
      text: "plate TTP", showarrow: false, font: { size: 10, color: "#c81c1c" },
    });
  }
  if (w.cy0_after_plate_min !== null) {
    annotations.push({
      x: w.cy0_after_plate_min, y: 0.94, xref: "x", yref: "paper", yshift: 12,
      text: "cy0_after_plate", showarrow: false, font: { size: 10, color: "#ff8c00" },
    });
  }
  const plotEl = document.getElementById("plot");
  Plotly.newPlot(plotEl, traces, {
    margin: { t: 30, r: 12, b: 42, l: 50 },
    xaxis: { title: "time (min)" },
    yaxis: { title: "linearised signal (V, anchored)" },
    shapes: shapes,
    annotations: annotations,
    hovermode: "closest",
  }, { displayModeBar: true, responsive: true });

  // Per-trace highlight-on-hover. When cursor is over a pixel trace,
  // that trace thickens + darkens; on unhover it snaps back to the
  // dim style. Well-mean (last trace) is always excluded from the effect.
  const nPixels = pixels ? pixels.length : 0;
  const wellMeanIdx = nPixels;   // well-mean is the last trace
  plotEl.on("plotly_hover", ev => {
    const pt = ev.points && ev.points[0];
    if (!pt) return;
    const idx = pt.curveNumber;
    if (idx === wellMeanIdx) return;
    Plotly.restyle(plotEl, { line: { width: 1.8, color: HL_COLOR } }, [idx]);
  });
  plotEl.on("plotly_unhover", ev => {
    const pt = ev.points && ev.points[0];
    if (!pt) return;
    const idx = pt.curveNumber;
    if (idx === wellMeanIdx) return;
    Plotly.restyle(plotEl, { line: { width: 0.6, color: DIM_COLOR } }, [idx]);
  });

  document.querySelectorAll(".well-item").forEach(el => el.classList.remove("active"));
  const activeEl = document.querySelector(
    ".well-item[data-chip='" + chipIdx + "'][data-well='" + wellIdx + "']"
  );
  if (activeEl) activeEl.classList.add("active");
}

function buildLhs() {
  const lhs = document.getElementById("lhs");
  DATA.forEach((chip, ci) => {
    const h = document.createElement("div");
    h.className = "chip-header";
    h.textContent = chip.chip_label;
    lhs.appendChild(h);
    chip.wells.forEach((w, wi) => {
      const el = document.createElement("div");
      el.className = "well-item";
      el.dataset.chip = ci;
      el.dataset.well = wi;
      const conc = w.log10_conc !== null ? "10^" + w.log10_conc : w.well_label;
      el.innerHTML = "w" + w.well_idx + " · " + w.well_label +
                     "<span class='conc'>" + conc + "</span>";
      el.onclick = () => render(ci, wi);
      lhs.appendChild(el);
    });
  });
}

buildLhs();
render(0, 0);
</script>
</body></html>
"""


def main():
    payload = []
    for cfg in _build_chip_configs():
        chip_payload = process_chip(cfg)
        if chip_payload is not None:
            payload.append(chip_payload)

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html)
    print(f"\nwrote {OUT_HTML}  ({OUT_HTML.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
