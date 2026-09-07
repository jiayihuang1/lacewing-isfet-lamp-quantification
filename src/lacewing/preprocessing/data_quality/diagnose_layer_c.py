"""Diagnose Layer C collateral.

We saw that Layer C (the spatial kNN-disagreement filter) drops some
pixels that the classifier always gets right.  At the default 95th
percentile cut on the PnG_Bead chip, that's 359 always-correct y=1
pixels - the question is whether those drops are *real collateral*
(clean amplifying signals being thrown away) or *legitimate-but-rare
edge cases* (pixels that look fine to the classifier but are near a
well boundary, or have anomalous shape, and would be dropped by a
human inspector too).

This script produces three figures per chip:

  1. Trace overlay:  random sample of always_correct dropped-by-C
     pixels next to a same-sized sample of always_correct kept-by-C
     pixels.  Visual sanity check - do they look the same?

  2. Spatial heatmap per well, coloured by:
         always_correct + kept
         always_correct + dropped by C
         non-correct + kept
         non-correct + dropped by C
     To see whether C-drops cluster near well boundaries.

  3. Distance-to-well-edge histogram per category.  Quantifies the
     "near the edge" intuition: if always_correct dropped pixels sit
     closer to the well's pixel-bounding-box edge than always_correct
     kept pixels, the supervisor's "outliers cluster at edges" hint
     is corroborated.

Run::

    python -m lacewing.preprocessing.data_quality.diagnose_layer_c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "filter_data"
OUT_DIR  = THIS_DIR / "diagnostics_layer_c"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def load_chip(json_path: Path) -> dict:
    return json.loads(json_path.read_text())


def collect_pixels(chip: dict) -> list[dict]:
    """Flatten the per-well pixel lists into one list, annotating each
    pixel with its well_id (so we can group by well later)."""
    out = []
    for w_str, w in chip["wells"].items():
        for p in w["pixels"]:
            q = dict(p)
            q["well_id"] = int(w_str)
            q["well_label"] = int(w["label"])
            out.append(q)
    return out


def category(p: dict) -> str:
    """One of {always_correct_kept, always_correct_dropC,
                other_kept,         other_dropC,
                dropped_AB}.
    Useful for colouring spatial maps."""
    if p["dropped_layer"] == "A" or p["dropped_layer"] == "B":
        return "dropped_AB"
    if p["status"] == "always_correct":
        return ("always_correct_dropC"
                if p["dropped_layer"] == "C" else "always_correct_kept")
    else:
        return ("other_dropC"
                if p["dropped_layer"] == "C" else "other_kept")


CAT_COLOURS = {
    "always_correct_kept":   "#bbbbbb",       # grey - the "background" of good pixels
    "always_correct_dropC":  "#1f77b4",       # blue - the pixels we want to investigate
    "other_kept":            "#d4af37",       # gold
    "other_dropC":           "#d35400",       # orange (C drops)
    "dropped_AB":            "#ececec",       # very light grey
}


# --------------------------------------------------------------------
# Figure 1: trace overlay of dropped vs kept always-correct
# --------------------------------------------------------------------

def fig_traces(pixels: list[dict], chip_name: str, n_show: int = 25,
               seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dropC = [p for p in pixels
             if p["status"] == "always_correct" and p["dropped_layer"] == "C"]
    kept  = [p for p in pixels
             if p["status"] == "always_correct" and p["dropped_layer"] is None]

    n_show = min(n_show, len(dropC), len(kept))
    if n_show == 0:
        return None
    dropC_sub = rng.choice(dropC, n_show, replace=False).tolist() \
        if isinstance(dropC, list) else list(rng.choice(dropC, n_show, replace=False))
    kept_sub  = rng.choice(kept,  n_show, replace=False).tolist() \
        if isinstance(kept, list) else list(rng.choice(kept, n_show, replace=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, samples, title, col in [
        (axes[0], kept_sub,  f"always_correct  kept by C   (n={len(kept_sub)})",
         "#7f7f7f"),
        (axes[1], dropC_sub, f"always_correct  dropped by C  (n={len(dropC_sub)})",
         "#1f77b4"),
    ]:
        for p in samples:
            ax.plot(p["trace"], color=col, linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("sample (downsampled)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("voltage")
    fig.suptitle(
        f"{chip_name}  -  trace shapes of always_correct pixels, "
        "kept vs dropped by Layer C",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"traces__{chip_name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------
# Figure 2: spatial heatmap per well
# --------------------------------------------------------------------

def fig_spatial(chip: dict, chip_name: str) -> Path:
    wells = sorted(chip["wells"].keys(), key=int)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=False, sharey=False)
    axes = axes.flat
    for ax, w_str in zip(axes, wells):
        w = chip["wells"][w_str]
        nrows, ncols = w["nrows"], w["ncols"]

        # Build a (nrows, ncols, 3) RGB image; default to white.
        img = np.ones((nrows, ncols, 3), dtype=np.float32)

        # Draw each surviving pixel.
        for p in w["pixels"]:
            cat = category(p)
            colour = CAT_COLOURS[cat]
            r, g, b = _hex_to_rgb01(colour)
            img[p["row"], p["col"]] = (r, g, b)

        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_title(f"well {w_str} (label {w['label']}, n={w['n_total']})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend.
    fig.suptitle(
        f"{chip_name}  -  spatial layout of pixels coloured by filter outcome",
        fontsize=11,
    )
    handles = []
    for cat in ["always_correct_kept", "always_correct_dropC",
                "other_kept", "other_dropC", "dropped_AB"]:
        handles.append(plt.matplotlib.patches.Patch(
            color=CAT_COLOURS[cat], label=cat.replace("_", " ")))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = OUT_DIR / f"spatial__{chip_name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _hex_to_rgb01(s: str) -> tuple[float, float, float]:
    s = s.lstrip("#")
    return (int(s[0:2], 16) / 255.0,
            int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0)


# --------------------------------------------------------------------
# Figure 3: distance-to-edge histograms
# --------------------------------------------------------------------

def distance_to_well_edge(pixels: list[dict]) -> np.ndarray:
    """Per pixel, distance (in pixel units) to the nearest edge of its
    well's *occupied bounding box*.  Uses bounding box of surviving
    pixels in that well; not the full nrows x ncols frame, because most
    of the grid is outside the actual well stencil.
    """
    # Build per-well bounding box first.
    per_well_bbox = {}
    for p in pixels:
        w = p["well_id"]
        bb = per_well_bbox.setdefault(w, [p["row"], p["row"], p["col"], p["col"]])
        bb[0] = min(bb[0], p["row"]); bb[1] = max(bb[1], p["row"])
        bb[2] = min(bb[2], p["col"]); bb[3] = max(bb[3], p["col"])

    out = np.empty(len(pixels), dtype=np.float32)
    for i, p in enumerate(pixels):
        rmin, rmax, cmin, cmax = per_well_bbox[p["well_id"]]
        d_top    = p["row"] - rmin
        d_bottom = rmax   - p["row"]
        d_left   = p["col"] - cmin
        d_right  = cmax   - p["col"]
        out[i] = min(d_top, d_bottom, d_left, d_right)
    return out


def fig_edge_distance(pixels: list[dict], chip_name: str) -> Path:
    cats = [category(p) for p in pixels]
    dist = distance_to_well_edge(pixels)

    # Focus on the comparison of interest: always_correct_kept vs always_correct_dropC.
    targets = [
        ("always_correct_kept",  "always_correct kept",   "#7f7f7f"),
        ("always_correct_dropC", "always_correct drop C", "#1f77b4"),
        ("other_dropC",          "non-correct drop C",    "#d35400"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cat, label, colour in targets:
        d = dist[[i for i, c in enumerate(cats) if c == cat]]
        if len(d) == 0:
            continue
        bins = np.arange(0, max(35, d.max() + 1))
        ax.hist(d, bins=bins, alpha=0.55, density=True, color=colour,
                label=f"{label} (n={len(d)}, median={np.median(d):.1f})")
    ax.set_xlabel("distance to well bounding-box edge (pixels)")
    ax.set_ylabel("density")
    ax.set_title(f"{chip_name}  -  edge proximity by filter outcome")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"edge_distance__{chip_name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# --------------------------------------------------------------------
# Numeric summary
# --------------------------------------------------------------------

def print_summary(pixels: list[dict], chip_name: str) -> None:
    cats = [category(p) for p in pixels]
    dist = distance_to_well_edge(pixels)

    print(f"\n=== {chip_name} ===")
    print(f"{'category':<25} {'n':>6} {'edge dist median':>18} {'edge dist mean':>15}")
    print("-" * 70)
    for cat in ["always_correct_kept", "always_correct_dropC",
                "other_kept", "other_dropC", "dropped_AB"]:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        d = dist[idx]
        print(f"{cat:<25} {len(idx):>6} {np.median(d):>18.2f} {d.mean():>15.2f}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-show", type=int, default=25,
                        help="number of traces to show per panel in fig 1")
    args = parser.parse_args()

    chip_files = sorted(p for p in DATA_DIR.glob("*.json")
                        if p.name != "manifest.json")
    if not chip_files:
        raise SystemExit(
            f"no chip JSONs in {DATA_DIR}; run build_filter_pipeline.py first")

    for cp in chip_files:
        chip = load_chip(cp)
        pixels = collect_pixels(chip)
        chip_name = chip["chip"]

        out_traces = fig_traces(pixels, chip_name, n_show=args.n_show,
                                seed=args.seed)
        out_sp     = fig_spatial(chip, chip_name)
        out_edge   = fig_edge_distance(pixels, chip_name)

        print_summary(pixels, chip_name)
        print(f"  wrote {out_traces.name}")
        print(f"  wrote {out_sp.name}")
        print(f"  wrote {out_edge.name}")


if __name__ == "__main__":
    main()
