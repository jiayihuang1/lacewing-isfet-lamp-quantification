"""Export data for the F-B methodology + extraction-rule viewer.

For each of {W=30, 60, 90, 120} F-B checkpoints (seed 0):
  * All test pixels' raw ISFET traces (downsampled by 4).
  * All test pixels' per-window sigmoid probabilities.
  * All test pixels' predicted TTP for every extraction rule.
  * Well-mean trace and well-mean P(t) curve.
  * Per-well qLAMP TTP.

Output: Analysis/quantification/viewer/fb_viewer_data.json (embedded into
Analysis/quantification/viewer/fb_viewer_standalone.html by the exporter).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from lacewing.quantification.eval.extraction_rules import ALL_RULES
from lacewing.quantification.eval.schema import read_labels, read_predictions
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
METHODS_RESULTS = LACEWING_PKG_DIR / "quantification" / "methods" / "results"
CACHE_PATH = LACEWING_PKG_DIR / "regression" / "data" / "cache" / "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3.npz"
JSON_OUT = LACEWING_PKG_DIR / "quantification" / "viewer" / "fb_viewer_data.json"
HTML_TEMPLATE = LACEWING_PKG_DIR / "quantification" / "viewer" / "fb_viewer.html"
HTML_STANDALONE = LACEWING_PKG_DIR / "quantification" / "viewer" / "fb_viewer_standalone.html"

SAMPLES_PER_MIN = 15
N_SAMPLES = 450
DOWN_FACTOR = 4      # downsample traces by 4x for transport (450 -> 113 samples)
SEED = 0             # seed to visualise (viewer note: F-B is seed-variable)

FB_DIRS = [
    ("w30_stride3", "p1_fb_slide_cls_w30_stride3_spatA3", 30, 3),
    ("w60_stride5", "p1_fb_slide_cls_w60_stride5_spatA3", 60, 5),
    ("w90_stride8", "p1_fb_slide_cls_w90_stride8_spatA3", 90, 8),
    ("w120_stride10", "p1_fb_slide_cls_w120_stride10_spatA3", 120, 10),
]


def _load_test_traces() -> dict[int, np.ndarray]:
    """Return {well_id: (n_pixels_in_well, N_SAMPLES) traces} for SD test chip."""
    data = np.load(CACHE_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32)
    well_id = data["well_id"].astype(int)
    split = data["split"].astype(int)  # 0=train, 1=test

    test_mask = split == 1
    X_te = X[test_mask]
    well_te = well_id[test_mask]

    out: dict[int, np.ndarray] = {}
    for w in np.unique(well_te):
        w_int = int(w)
        out[w_int] = X_te[well_te == w_int]
    return out


def _apply_all_rules(probs: np.ndarray, window: int, stride: int) -> dict[str, np.ndarray]:
    """Apply every rule in ALL_RULES to `probs`.  Returns {rule_id: (n_pixels,) ttp_pred}."""
    out: dict[str, np.ndarray] = {}
    for rule in ALL_RULES:
        try:
            out[rule.id] = rule.fn(probs, window, stride).astype(np.float32)
        except Exception as e:
            print(f"[warn] rule {rule.id} failed: {e}")
            out[rule.id] = np.full(probs.shape[0], np.nan, dtype=np.float32)
    return out


def _round_list(arr: np.ndarray, dp: int = 3) -> list:
    """Round floats to dp decimal places for JSON compactness."""
    return [round(float(v), dp) if np.isfinite(v) else None for v in arr]


def main() -> None:
    print(f"== F-B viewer exporter ==")
    print(f"Loading test-set traces from {CACHE_PATH.name}...")
    well_traces = _load_test_traces()
    print(f"  Loaded {len(well_traces)} test wells:")
    for w in sorted(well_traces):
        print(f"    well {w}: {well_traces[w].shape[0]} pixels")

    # Downsampled time axis for the traces.
    n_ds = int(np.ceil(N_SAMPLES / DOWN_FACTOR))
    t_axis_min = [round((i * DOWN_FACTOR) / SAMPLES_PER_MIN, 3) for i in range(n_ds)]

    # Data structure keyed by window config (dropdown 1).
    windows: dict[str, dict] = {}

    for key, dir_name, W, S in FB_DIRS:
        seed_dir = METHODS_RESULTS / dir_name / f"seed{SEED}"
        preds_path = seed_dir / "predictions.npz"
        labels_path = seed_dir / "labels.npz"
        if not preds_path.exists() or not labels_path.exists():
            print(f"[skip] {key}: missing outputs at {seed_dir}")
            continue

        preds = read_predictions(preds_path)
        labels = read_labels(labels_path)

        # Sanity: labels split should be all "test".
        mask = labels.split == "test"
        if not mask.all():
            print(f"[warn] {key}: labels contain non-test rows; filtering.")
        well_id = labels.well_id[mask].astype(int)
        y_true = labels.ttp_true_min[mask]

        window_probs = preds.extras.get("window_probs")
        if window_probs is None:
            print(f"[skip] {key}: no window_probs in extras.")
            continue
        # window_probs may have been saved for all pixels or only test — trust its shape.
        assert window_probs.shape[0] == len(y_true), (
            f"{key}: window_probs shape {window_probs.shape} vs labels {len(y_true)}"
        )

        # Apply every extraction rule to all test pixels.
        print(f"  {key}: applying {len(ALL_RULES)} rules to {window_probs.shape[0]} test pixels...")
        rules_ttp = _apply_all_rules(window_probs, W, S)

        # Per-window centres in minutes.
        n_windows = window_probs.shape[1]
        window_centres_min = [
            round((i * S + (W - 1) / 2) / SAMPLES_PER_MIN, 3)
            for i in range(n_windows)
        ]

        # Bucket by well.
        wells_out: dict[str, dict] = {}
        for w in sorted(np.unique(well_id)):
            w_int = int(w)
            wmask = well_id == w_int
            if not wmask.any():
                continue

            traces_full = well_traces.get(w_int)  # (n_pix, 450) from cache
            if traces_full is None:
                print(f"[skip] well {w_int}: no cache traces.")
                continue

            # Traces: downsample by DOWN_FACTOR for transport.
            traces_ds = traces_full[:, ::DOWN_FACTOR]  # (n_pix, ~113)
            # Mean trace.
            mean_trace = traces_ds.mean(axis=0)

            # Per-window probs for this well.
            probs_w = window_probs[wmask]  # (n_pix, n_windows)
            mean_probs = probs_w.mean(axis=0)

            # Per-rule TTP for this well.
            rules_ttp_w: dict[str, list] = {
                rid: _round_list(rules_ttp[rid][wmask]) for rid in rules_ttp
            }

            wells_out[str(w_int)] = {
                "well_id": w_int,
                "n_pixels": int(wmask.sum()),
                "y_true_min": round(float(y_true[wmask].mean()), 3),  # constant per well
                "traces_ds": [_round_list(t, 3) for t in traces_ds],
                "mean_trace": _round_list(mean_trace, 3),
                "probs": [_round_list(p, 3) for p in probs_w],
                "mean_probs": _round_list(mean_probs, 3),
                "rules_ttp": rules_ttp_w,
            }

        windows[key] = {
            "key": key,
            "dir_name": dir_name,
            "W": W,
            "stride": S,
            "window_centres_min": window_centres_min,
            "n_windows": n_windows,
            "wells": wells_out,
        }
        print(f"    OK: {len(wells_out)} wells packed for {key}")

    # Rule metadata for the dropdown.
    rule_meta = [
        {"id": r.id, "label": r.label, "family": r.family} for r in ALL_RULES
    ]

    out = {
        "schema": 1,
        "note": (
            "F-B methodology viewer.  Shows per-pixel window probabilities + "
            f"predicted TTP per extraction rule for seed {SEED}.  Windows "
            "downsampled to " + str(DOWN_FACTOR) + "x for transport."
        ),
        "seed": SEED,
        "samples_per_min": SAMPLES_PER_MIN,
        "n_samples": N_SAMPLES,
        "down_factor": DOWN_FACTOR,
        "t_axis_min": t_axis_min,
        "rules": rule_meta,
        "windows": windows,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(out)
    JSON_OUT.write_text(json_str)
    size_mb = JSON_OUT.stat().st_size / 1024 / 1024
    n_wins = len(windows)
    n_wells_total = sum(len(w["wells"]) for w in windows.values())
    print(f"\nWrote {JSON_OUT.name} ({size_mb:.1f} MB) — {n_wins} windows, {n_wells_total} well-cells total")

    # Standalone HTML with embedded JSON.
    if HTML_TEMPLATE.exists():
        html = HTML_TEMPLATE.read_text()
        replacement = (
            "// Embedded data (auto-injected by export_fb_viewer_data.py).\n"
            f"DATA = {json_str};\n"
            'document.getElementById("status").textContent = "loaded (embedded).";\n'
            "initControls();\n"
            "render();"
        )
        pattern = re.compile(
            r'fetch\("fb_viewer_data\.json"\)\s*\.then\([^}]*?\}\)\s*\.catch\([^}]*?\}\);',
            re.DOTALL,
        )
        # Use a lambda replacement so backslash sequences in the JSON string
        # (e.g. \uXXXX escapes) aren't interpreted as regex group refs.
        new_html = pattern.sub(lambda m: replacement, html, count=1)
        if new_html == html:
            print(f"[warn] no fetch(...) block found in {HTML_TEMPLATE.name}; standalone not built.")
        else:
            HTML_STANDALONE.write_text(new_html)
            html_mb = HTML_STANDALONE.stat().st_size / 1024 / 1024
            print(f"Wrote {HTML_STANDALONE.name} ({html_mb:.1f} MB — self-contained)")
    else:
        print(f"[warn] template {HTML_TEMPLATE.name} not found; standalone not built.")


if __name__ == "__main__":
    main()
