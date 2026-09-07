"""Manual well-mean TTP labelling UI (matplotlib clicker).

Walks through every well across the 4 UTI chips, shows:
  - TOP: chip well-mean (spatA3-smoothed) — the signal to label on.
  - BOTTOM: chip smoothed first derivative — for visual context.

Controls:
  - Left-click anywhere on the TOP plot: sets the chip TTP for this well at that time.
  - Right-click: mark well as "no amplification" (skip).
  - Press 'n': confirm current pick and go to next well.
  - Press 'r': re-open plate overlay as a sanity check for this well.
  - Press 'q': quit + save progress.

Save file: Analysis/quantification/chip_pipeline/output/manual_chip_ttps.json
Format: {"labels": {"<chip_tag>__well<idx>": {"chip_ttp_min": float | None, "notes": str}}, ...}

Resume: re-run the script. Wells already in the JSON are skipped.

Usage
-----
    python -m lacewing.quantification.chip_pipeline.manual_label_ttp_ui
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
# Force macosx backend before importing pyplot so the window pops as a native macOS window.
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from lacewing.quantification.chip_pipeline import process_chip as pipe
from lacewing.quantification.chip_pipeline import process_all_uti_chips as multi


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS_JSON = OUT_DIR / "manual_chip_ttps.json"

CHIP_TTPS_JSON = OUT_DIR / "chip_ttps.json"  # from A2b, used as data source

# Chips to exclude from labelling. 26-06-24 is the negative-control test —
# 10 wells with different primer sets + water; no meaningful amplification
# to label, and no matching plate concentration for calibration.
SKIP_CHIP_TAGS = {"uti_2606024_NC_test"}


def _load_existing_labels() -> dict:
    if LABELS_JSON.exists():
        return json.loads(LABELS_JSON.read_text())
    return {"labels": {}}


def _save_labels(data: dict) -> None:
    LABELS_JSON.write_text(json.dumps(data, indent=2))


def _load_wells_from_a2b() -> list[dict]:
    """Reuse the well-mean + derivative arrays A2b already computed."""
    if not CHIP_TTPS_JSON.exists():
        raise SystemExit(
            f"Missing {CHIP_TTPS_JSON}. "
            "Run `python -m lacewing.quantification.chip_pipeline.compute_chip_ttps` first."
        )
    data = json.loads(CHIP_TTPS_JSON.read_text())
    return data["wells"]


@dataclass
class LabelState:
    """Per-well transient state for the click handler."""
    chip_ttp_min: float | None = None
    is_no_amp: bool = False
    show_plate: bool = False


def label_wells() -> None:
    wells_all = _load_wells_from_a2b()
    # Filter out chips we don't want to label (e.g. NC test with no matching plate concs).
    wells = [w for w in wells_all if w["chip_tag"] not in SKIP_CHIP_TAGS]
    n_skipped = len(wells_all) - len(wells)
    if n_skipped > 0:
        print(f"[filter] skipped {n_skipped} wells from {sorted(SKIP_CHIP_TAGS)}")

    existing = _load_existing_labels()
    labels = existing.setdefault("labels", {})

    remaining = [w for w in wells if f"{w['chip_tag']}__well{w['well_id']}" not in labels]

    if not remaining:
        print("[done] all wells already labelled. Nothing to do.")
        print(f"       Load from {LABELS_JSON}")
        return

    print(f"[start] {len(remaining)} wells to label out of {len(wells)} total")
    print(f"        Progress saved to {LABELS_JSON}")
    print()
    print("Controls:")
    print("  LEFT CLICK on TOP plot   → set chip TTP at that time")
    print("  RIGHT CLICK              → mark as 'no amplification'")
    print("  Press 'n'                → confirm + next well")
    print("  Press 'r'                → toggle plate overlay for sanity check")
    print("  Press 'u'                → undo current pick (re-label this well)")
    print("  Press 'q'                → quit + save progress")
    print()

    # Set up the figure ONCE, reuse it across wells
    fig, (ax_sig, ax_deriv) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=True,
    )
    state = LabelState()
    current_idx = [0]  # boxed so nested closures can mutate
    quit_flag = [False]

    def _plot_well(w: dict) -> None:
        """Redraw both axes for the given well record."""
        ax_sig.clear()
        ax_deriv.clear()

        t = np.array(w["time_min"], dtype=float)
        mean_spat = np.array(w["well_mean_spat"], dtype=float)
        smoothed = np.array(w["smoothed_signal"], dtype=float)
        deriv = np.array(w["derivative"], dtype=float)

        # TOP plot — well-mean + smoothed
        ax_sig.plot(t, mean_spat, color="#1a1d24", linewidth=1.2, alpha=0.5, label="well-mean (spatA3)")
        ax_sig.plot(t, smoothed, color="#b30000", linewidth=1.6, linestyle=":", label="smoothed (SG w=15, o=3)")

        # If we've already picked a TTP for this well, mark it
        if state.chip_ttp_min is not None:
            ax_sig.axvline(state.chip_ttp_min, color="#2ca02c", linewidth=2, linestyle="--",
                           label=f"picked chip TTP = {state.chip_ttp_min:.2f} min")

        # Optional plate overlay (right-panel style, normalised amplitude so it fits on the same axis)
        if state.show_plate and w.get("plate") and w["plate"].get("replicates"):
            reps = w["plate"]["replicates"]
            for rep in reps:
                cycles = rep.get("cycles")
                fluor = rep.get("fluor")
                if cycles and fluor:
                    plate_t = np.array(cycles, dtype=float) * 0.5  # LC96 30s cycles
                    fluor_arr = np.array(fluor, dtype=float)
                    # Normalise plate fluor to the chip signal range for visual overlay
                    chip_range = smoothed.max() - smoothed.min()
                    plate_range = fluor_arr.max() - fluor_arr.min()
                    if plate_range > 0 and chip_range > 0:
                        fluor_norm = (fluor_arr - fluor_arr.min()) / plate_range * chip_range + smoothed.min()
                        ax_sig.plot(plate_t, fluor_norm, color="#0369a1", linewidth=1.0, alpha=0.4,
                                    linestyle="-", label=f"plate {rep['position']} (normalised)")
            if w["plate"].get("plate_ttp_mean_min") is not None:
                ax_sig.axvline(w["plate"]["plate_ttp_mean_min"], color="#0369a1", linewidth=1.4,
                               linestyle=":", alpha=0.7,
                               label=f"plate TTP mean = {w['plate']['plate_ttp_mean_min']:.2f} min")

        conc_str = f"log10={w['log10_conc']:.2f}" if w["log10_conc"] is not None else "log10=NaN (NTC)"
        # Count labelled wells within scope (exclude any pre-existing labels from skipped chips)
        n_labelled = sum(1 for k in labels
                         if not any(k.startswith(f"{skip}__") for skip in SKIP_CHIP_TAGS))
        n_total = len(wells)
        title = (f"[{n_labelled}/{n_total} labelled] "
                 f"{w['chip_tag']}  |  well {w['well_id']} ({w['well_label']})  |  {conc_str}  "
                 f"|  {w['n_kept_pixels']} kept pixels")
        if state.is_no_amp:
            title += "  |  ⚠ marked as no amp"
        ax_sig.set_title(title, fontsize=11)
        ax_sig.set_ylabel("chip signal (V)")
        ax_sig.legend(loc="best", fontsize=9)
        ax_sig.grid(True, alpha=0.3, linewidth=0.5)

        # BOTTOM plot — derivative
        ax_deriv.plot(t, deriv, color="#d97706", linewidth=1.4, label="d/dt smoothed")
        if state.chip_ttp_min is not None:
            ax_deriv.axvline(state.chip_ttp_min, color="#2ca02c", linewidth=2, linestyle="--")
        ax_deriv.set_xlabel("chip time (min)")
        ax_deriv.set_ylabel("d/dt (V/min)")
        ax_deriv.axhline(0, color="#888", linewidth=0.5, alpha=0.5)
        ax_deriv.grid(True, alpha=0.3, linewidth=0.5)
        ax_deriv.legend(loc="best", fontsize=9)

        fig.tight_layout()
        fig.canvas.draw_idle()

    def _commit_current_and_advance() -> None:
        w = remaining[current_idx[0]]
        key = f"{w['chip_tag']}__well{w['well_id']}"
        labels[key] = {
            "chip_tag": w["chip_tag"],
            "well_id": w["well_id"],
            "well_label": w["well_label"],
            "log10_conc": w["log10_conc"],
            "chip_ttp_min": None if state.is_no_amp else state.chip_ttp_min,
            "is_no_amp": state.is_no_amp,
            "plate_ttp_mean_min": (w["plate"]["plate_ttp_mean_min"] if w.get("plate") else None),
        }
        _save_labels({"labels": labels})
        print(f"  [saved] {key}: chip_ttp={labels[key]['chip_ttp_min']}  no_amp={labels[key]['is_no_amp']}")

        # Advance
        current_idx[0] += 1
        state.chip_ttp_min = None
        state.is_no_amp = False
        state.show_plate = False
        if current_idx[0] >= len(remaining):
            print("[done] all remaining wells labelled — closing.")
            plt.close(fig)
            return
        _plot_well(remaining[current_idx[0]])

    def _on_click(event) -> None:
        if event.inaxes != ax_sig:
            return
        if event.xdata is None:
            return
        if event.button == 1:  # left click
            state.chip_ttp_min = float(event.xdata)
            state.is_no_amp = False
            print(f"  [pick] chip TTP = {state.chip_ttp_min:.2f} min")
        elif event.button == 3:  # right click
            state.is_no_amp = True
            state.chip_ttp_min = None
            print(f"  [pick] marked as NO AMPLIFICATION")
        _plot_well(remaining[current_idx[0]])

    def _on_key(event) -> None:
        if event.key == "n":
            if state.chip_ttp_min is None and not state.is_no_amp:
                print("  [warn] no pick yet — left-click to set TTP or right-click to mark no-amp")
                return
            _commit_current_and_advance()
        elif event.key == "r":
            state.show_plate = not state.show_plate
            print(f"  [plate overlay] {'ON' if state.show_plate else 'OFF'}")
            _plot_well(remaining[current_idx[0]])
        elif event.key == "u":
            state.chip_ttp_min = None
            state.is_no_amp = False
            print(f"  [undo] pick cleared")
            _plot_well(remaining[current_idx[0]])
        elif event.key == "q":
            quit_flag[0] = True
            print("[quit] saving + closing")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    _plot_well(remaining[current_idx[0]])

    # Set window title so it's easy to find in the Dock / Cmd-Tab.
    try:
        fig.canvas.manager.set_window_title("UTI TTP labelling — LEFT-click to pick, 'n' next, 'q' quit")
    except Exception:
        pass

    print()
    print("*** matplotlib window should be open now.")
    print("*** If you don't see it, check your Dock for a Python rocket icon and click it.")
    print("*** The window title will start with 'UTI TTP labelling'.")
    print()

    plt.show()

    # Final save state message
    labels_now = _load_existing_labels()["labels"]
    n_ttp = sum(1 for v in labels_now.values() if v.get("chip_ttp_min") is not None)
    n_noamp = sum(1 for v in labels_now.values() if v.get("is_no_amp"))
    print(f"\n[final] labelled: {len(labels_now)} wells "
          f"(TTP set: {n_ttp}  |  no-amp: {n_noamp})")
    print(f"        saved to {LABELS_JSON}")


if __name__ == "__main__":
    label_wells()
