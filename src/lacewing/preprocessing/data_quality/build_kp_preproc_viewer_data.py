"""Preprocessing · builds the KP preprocessing viewer data. [Cat A] Report §NewData; interactive viewer at src/lacewing/viewers/kp_preproc_viewer.html.

Extract per-(chip, well) trace arrays across the three KP preprocessing
states and emit one JSON per chip + a manifest for the browser viewer.

For each of the 5 KP chips we produce three panels of traces per well:
    (1) deployed  = raw z-score only (Layer A alone)
    (2) +MAD      = deployed + chip-relative MAD-ABCD pixel-quality filter
    (3) +spatA3   = MAD + order-3 spatial averaging (the deployed pipeline
                    used everywhere downstream)

Source caches (already produced by the RQ1 preprocessing ablation):
    Analysis/classification/data/cache/dataset_conc_deployed.npz
    Analysis/classification/data/cache/dataset_conc_mad.npz
    Analysis/classification/data/cache/dataset_conc.npz

Only rows whose chip_id begins with 'conc_' (i.e. the 5 KP chips) are kept.

To keep JSON small we downsample the time axis by a factor DOWNSAMPLE and
cap the number of traces plotted per (well, state) by MAX_TRACES_PER_PANEL
(pixels sub-sampled uniformly at random with a fixed seed).

Output:
    Analysis/preprocessing/data_quality/kp_preproc_viewer_data/manifest.json
    Analysis/preprocessing/data_quality/kp_preproc_viewer_data/<chip>.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT

ROOT = DATA_ROOT  # was: parents[3]
CACHE_DIR = LACEWING_PKG_DIR / "classification" / "data" / "cache"
STAGES_DIR = LACEWING_PKG_DIR / "quantification" / "conc_data" / "output"
OUT_DIR = LACEWING_PKG_DIR / "preprocessing" / "data_quality" / "kp_preproc_viewer_data"
OUT_DIR.mkdir(exist_ok=True)

STATES = [
    ("deployed", "dataset_conc_deployed.npz", "(1) Deployed (raw z-score only)"),
    ("mad",      "dataset_conc_mad.npz",      "(2) + MAD-ABCD filter"),
    ("full",     "dataset_conc.npz",          "(3) + order-3 spatial averaging"),
]
KP_CHIPS = [
    "conc_260812_KP_01",
    "conc_260813_KP_DDM_02",
    "conc_260820_KP_DDM_03",
    "conc_260823_KP_DDM_04",
    "conc_260827_KP_DDM_05",
]
CHIP_SHORT_LABELS = {
    "conc_260812_KP_01":     "Chip 1 (2026-08-12, KP)",
    "conc_260813_KP_DDM_02": "Chip 2 (2026-08-13, KP+DDM)",
    "conc_260820_KP_DDM_03": "Chip 3 (2026-08-20, KP+DDM)",
    "conc_260823_KP_DDM_04": "Chip 4 (2026-08-23, KP+DDM)",
    "conc_260827_KP_DDM_05": "Chip 5 (2026-08-27, KP+DDM)",
}

DOWNSAMPLE = 3
MAX_TRACES_PER_PANEL = 200
RNG = np.random.default_rng(0)


def _load_stages_json(chip_tag: str) -> dict:
    """Return the per-well concentration + time-axis mapping for a chip."""
    f = STAGES_DIR / f"{chip_tag}_stages.json"
    if not f.exists():
        return {"wells": []}
    return json.loads(f.read_text())


def _load_cache(name: str) -> dict:
    z = np.load(CACHE_DIR / name)
    out = {k: z[k] for k in z.keys()}
    z.close()
    return out


def main() -> None:
    print("Loading 3 caches...")
    caches = {tag: _load_cache(name) for tag, name, _ in STATES}
    for tag in caches:
        print(f"  {tag}: X={caches[tag]['X'].shape} n_chips={len(np.unique(caches[tag]['chip_id']))}")

    T_full = caches["deployed"]["X"].shape[1]
    time_axis_idx = np.arange(0, T_full, DOWNSAMPLE).tolist()

    manifest = {
        "chips": [],
        "states": [{"tag": tag, "label": label} for tag, _, label in STATES],
        "downsample": DOWNSAMPLE,
        "max_traces_per_panel": MAX_TRACES_PER_PANEL,
    }

    for chip in KP_CHIPS:
        chip_stages = _load_stages_json(chip)
        stage_wells = {w["well"]: w for w in chip_stages.get("wells", [])}
        # Time axis in minutes (per-well but constant across wells within a chip)
        time_min_full = None
        for w in stage_wells.values():
            if "time_min" in w:
                time_min_full = w["time_min"]
                break
        if time_min_full is None or len(time_min_full) < T_full:
            time_min_ds = [i * 5.0 / 60 for i in range(T_full)][::DOWNSAMPLE]
        else:
            time_min_ds = time_min_full[::DOWNSAMPLE]

        # Discover wells present in the deployed cache for this chip.
        m0 = caches["deployed"]["chip_id"] == chip
        wells = sorted(int(w) for w in np.unique(caches["deployed"]["well_id"][m0]).tolist())
        print(f"\n{chip}: wells={wells}")

        chip_wells_payload = []
        for w in wells:
            info = stage_wells.get(w, {})
            label = info.get("label", f"well {w}")
            log10 = info.get("log10_conc", None)
            y_label = int(caches["deployed"]["y"][m0 & (caches["deployed"]["well_id"] == w)][0])

            panels = {}
            for tag in caches:
                c = caches[tag]
                mask = (c["chip_id"] == chip) & (c["well_id"] == w)
                n = int(mask.sum())
                if n == 0:
                    panels[tag] = {"n_kept": 0, "traces": []}
                    continue
                X = c["X"][mask]  # (n, T)
                if n > MAX_TRACES_PER_PANEL:
                    idx = RNG.choice(n, size=MAX_TRACES_PER_PANEL, replace=False)
                    X_plot = X[idx][:, ::DOWNSAMPLE]
                else:
                    X_plot = X[:, ::DOWNSAMPLE]
                # Round to 4 dp for JSON size
                traces = np.round(X_plot, 4).tolist()
                panels[tag] = {
                    "n_kept": n,           # total pixels for this (well, state)
                    "n_plotted": len(traces),
                    "traces": traces,
                }
            n_dep = panels["deployed"]["n_kept"]
            n_mad = panels["mad"]["n_kept"]
            n_full = panels["full"]["n_kept"]
            frac_mad = (n_mad / n_dep) if n_dep else 0.0
            frac_full = (n_full / n_dep) if n_dep else 0.0
            chip_wells_payload.append({
                "well": w,
                "label": label,
                "log10_conc": log10,
                "y": y_label,
                "n_deployed": n_dep,
                "n_mad": n_mad,
                "n_full": n_full,
                "frac_mad": round(frac_mad, 4),
                "frac_full": round(frac_full, 4),
                "panels": panels,
            })
            print(f"  well {w:>2} ({label:>10}) y={y_label} "
                  f"deployed={n_dep} mad={n_mad} ({frac_mad:.2%}) "
                  f"full={n_full} ({frac_full:.2%})")

        chip_payload = {
            "chip":       chip,
            "chip_label": CHIP_SHORT_LABELS[chip],
            "time_min":   time_min_ds,
            "wells":      chip_wells_payload,
        }
        out_path = OUT_DIR / f"{chip}.json"
        out_path.write_text(json.dumps(chip_payload, separators=(",", ":")))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  wrote {out_path.name} ({size_mb:.2f} MB)")

        manifest["chips"].append({
            "chip":       chip,
            "chip_label": CHIP_SHORT_LABELS[chip],
            "json":       out_path.name,
            "n_wells":    len(chip_wells_payload),
        })

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
