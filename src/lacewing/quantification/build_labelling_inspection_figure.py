"""Build · labelling-variant visual inspection figure. [Cat A] Report §Appendix labelling-variants.

Generate the labelling-variant inspection figure for the report appendix.

Produces a 4x2 grid of representative KP wells covering log10-conc 3-6 across
the five KP chips. Each panel shows:
  - smoothed chip well-mean signal (black)
  - plate anchor (blue dotted vertical)
  - V4 zero_cross_d2 (purple dashed)
  - V6 cy0_after_plate (orange dashed)
  - V7 thr_deriv_after_plate (grey dashed)

Output: Report/Final Report/Drafts/latex/images/labelling_variants_inspection.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lacewing.quantification.systematic_ttp import compute_all_variants
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


KP_CHIPS = [
    "conc_260812_KP_01",
    "conc_260813_KP_DDM_02",
    "conc_260820_KP_DDM_03",
    "conc_260823_KP_DDM_04",
    "conc_260827_KP_DDM_05",
]
CHIP_LABEL = {c: f"chip {i+1}" for i, c in enumerate(KP_CHIPS)}

# 8 wells covering log10-conc 3-6 across chips: two per concentration
# from different chips, so the figure shows both chip diversity and
# concentration diversity in one page.
SELECTION = [
    ("conc_260812_KP_01",     0),  # chip 1 well 0, log10=6
    ("conc_260820_KP_DDM_03", 0),  # chip 3 well 0, log10=6
    ("conc_260813_KP_DDM_02", 1),  # chip 2 well 1, log10=5
    ("conc_260827_KP_DDM_05", 5),  # chip 5 well 5, log10=5
    ("conc_260820_KP_DDM_03", 2),  # chip 3 well 2, log10=4
    ("conc_260823_KP_DDM_04", 6),  # chip 4 well 6, log10=4
    ("conc_260813_KP_DDM_02", 3),  # chip 2 well 3, log10=3
    ("conc_260827_KP_DDM_05", 7),  # chip 5 well 7, log10=3
]

STAGES_DIR = LACEWING_PKG_DIR / "quantification" / "conc_data" / "output"
OUT_PATH = Path("Report/Final Report/Drafts/latex/images/labelling_variants_inspection.pdf")

# Colour palette matches the systematic_ttp_viewer where possible.
COL_PLATE = "#0369a1"     # blue
COL_V4    = "#7A3C8A"     # purple
COL_V6    = "#d97706"     # orange
COL_V7    = "#606060"     # grey


def load_well(chip: str, well_idx: int):
    stages = json.loads((STAGES_DIR / f"{chip}_stages.json").read_text())
    for w in stages["wells"]:
        if w["well"] == well_idx:
            return w
    raise KeyError(f"well {well_idx} not found in {chip}")


def draw_panel(ax, chip: str, well_idx: int):
    w = load_well(chip, well_idx)
    t = np.asarray(w["time_min"], dtype=np.float64)
    sig = np.asarray(w["mean_after_spat"], dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(sig)
    t, sig = t[mask], sig[mask]

    plate = (w.get("plate") or {}).get("takeoff_min")

    variants = compute_all_variants(sig, t, plate_ttp_min=plate)
    v4 = variants["zero_cross_d2"].ttp_min
    v6 = variants["cy0_after_plate"].ttp_min
    v7 = variants["thr_deriv_after_plate"].ttp_min

    # Signal
    ax.plot(t, sig, color="black", linewidth=0.9, zorder=2)

    ymin, ymax = sig.min(), sig.max() * 1.05
    ax.set_ylim(ymin - 0.001, ymax)
    ax.set_xlim(t.min(), t.max())
    ax.grid(True, linestyle=":", alpha=0.4, zorder=1)

    # Collect all landmarks and assign each to a y-slot that doesn't
    # collide with any already-placed label. Each label occupies an
    # x-range starting at its line's x and extending by label_width; two
    # labels can share the same y-slot only if their x-ranges don't overlap.
    landmarks = [
        (plate, COL_PLATE, "plate", ":"),
        (v4,    COL_V4,    "SDM", "--"),
        (v6,    COL_V6,    "Cy0 (chosen)", "--"),
        (v7,    COL_V7,    "threshold-deriv", "--"),
    ]
    valid = [(x, c, lbl, s) for (x, c, lbl, s) in landmarks
             if x is not None and np.isfinite(x)]
    valid.sort(key=lambda r: r[0])  # left-to-right

    x_span = float(t.max() - t.min())
    slots = [0.97, 0.82, 0.67, 0.52]  # top-down y positions (fraction of ymax)
    # Approx label width in data units: 1.2 char per digit * ~0.017*xspan per char
    # We keep it slightly generous to prevent visual overlap.
    def label_width(text: str) -> float:
        return len(text) * 0.017 * x_span + 0.06 * x_span

    # slot_occupied[i] = list of (x_start, x_end) already placed on slot i
    slot_occupied: list[list[tuple[float, float]]] = [[] for _ in slots]

    for x, colour, lbl, style in valid:
        ax.axvline(x, color=colour, linestyle=style, linewidth=1.1,
                   alpha=0.9, zorder=3)
        full_text = f"{lbl} {x:.2f}"
        width = label_width(full_text)
        label_x_start = x + 0.008 * x_span
        label_x_end = label_x_start + width
        # Pick the topmost slot with no x-overlap.
        chosen = 0
        for i in range(len(slots)):
            if all(not (label_x_start < e and label_x_end > s)
                   for (s, e) in slot_occupied[i]):
                chosen = i
                break
        slot_occupied[chosen].append((label_x_start, label_x_end))
        y = ymax * slots[chosen]
        ax.text(label_x_start, y, full_text,
                color=colour, fontsize=7, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    conc = int(w.get("log10_conc", 0))
    ax.set_title(f"{CHIP_LABEL[chip]}, well {well_idx}, $10^{{{conc}}}$",
                 fontsize=9, pad=2)


def build():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 10.5), sharey=False)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
    })
    for ax, (chip, well_idx) in zip(axes.flat, SELECTION):
        draw_panel(ax, chip, well_idx)

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("time (min)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("signal (a.u.)", fontsize=9)

    # Single shared legend at the top
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COL_PLATE, linestyle=":",  label="plate anchor (LC96 takeoff)"),
        Line2D([0], [0], color=COL_V4,    linestyle="--", label="plate-anchored SDM ($d^2$ zero-crossing)"),
        Line2D([0], [0], color=COL_V6,    linestyle="--", label=r"plate-anchored Cy$_0$ (tangent-at-inflection, chosen)"),
        Line2D([0], [0], color=COL_V7,    linestyle="--", label="plate-anchored threshold-derivative"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.995))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
