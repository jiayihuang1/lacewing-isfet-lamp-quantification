"""Per-chip diagnostic showing the artefact-trim cut points.

For each chip, produce one PNG with two panels:

  Left  (DETECTION VIEW):
    Non-baseline-subtracted active-pixel mean per well (matches the
    units of main_DNA's linearized_signal_combined output -- this is
    the y-axis where the early-time downward dip is visible).  Each
    well's individual cut time is drawn as a thin dashed vertical
    line; the chip-wide cut is drawn thicker and solid.  Used to
    inspect whether the detector picks the right spot.

  Right (CLASSIFIER-INPUT VIEW):
    Baseline-subtracted active-pixel mean per well, AFTER applying the
    trim.  Time axis re-zeros at the chip cut.  This is the trace
    shape that would be fed to the classifier window.

Combination: per-Vref active pixel sets are union-combined via
``load_chip_combined`` first, so for multi-Vref chips (e.g. Elena) the
plot reflects the union across all Vref slices.

Output: ``Analysis/multi_data/output/<chip_name>/trim_diagnostic.png``.

Run::

    python -m lacewing.quantification.multi_data_shared.visualise_trim \\
        --chip "Data/Multi/D20260320_E00_C00_F4500KHz_U_Elena_steap_cv"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .load_combined import (DEFAULT_END_TIME_MIN, DEFAULT_N_A_TYPE,
                             DEFAULT_N_WELLS, load_chip_combined)
from .artefact_trim import trim_chip_artefact


OUTPUT_ROOT = Path(__file__).resolve().parent / "output"

WELL_COLORS = plt.cm.tab10.colors  # 10 wells -> tab10 cycle


def _short_chip_name(chip_path: Path) -> str:
    """``D20260320_..._U_Elena_steap_cv`` -> ``Elena_steap_cv``."""
    stem = chip_path.name
    marker = "KHz_U_"
    return stem[stem.index(marker) + len(marker):] if marker in stem else stem


def plot_chip_trim(chip_path: Path, *,
                   n_wells: int = DEFAULT_N_WELLS,
                   n_a_type: str = DEFAULT_N_A_TYPE,
                   end_time_min: int = DEFAULT_END_TIME_MIN,
                   out_path: Path | None = None) -> Path:
    chip_path = Path(chip_path)

    # 1. Load combined experiment.
    exp, diag = load_chip_combined(
        chip_path,
        n_wells=n_wells,
        n_a_type=n_a_type,
        end_time_min=end_time_min,
        print_status=False,
    )

    # 2. Capture the pre-trim non-bs traces for the left panel.
    pre_time_min = [np.asarray(w.time_min, dtype=float).copy()
                    for w in exp.wells_list]
    pre_signals = [np.asarray(w.well_2d_nl_active_mean, dtype=float).copy()
                   for w in exp.wells_list]

    # 3. Detect (but don't apply yet) so we can mark per-well cuts on the
    #    left panel.
    detect = trim_chip_artefact(exp, apply=False)

    # 4. Now apply the trim to get the right panel's post-trim view.
    applied = trim_chip_artefact(exp, apply=True)
    post_time_min = [np.asarray(w.time_min, dtype=float)
                     for w in exp.wells_list]
    post_signals_bs = [np.asarray(w.well_2d_bs_active_mean, dtype=float)
                       for w in exp.wells_list]

    # 5. Build the figure.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 5.6),
                                      gridspec_kw={"width_ratios": [1, 1]})
    short_name = _short_chip_name(chip_path)
    fig.suptitle(
        f"{short_name}   |   n_refs={diag['n_refs']}   |   "
        f"chip cut = {detect.chip_cut_min:.2f} min   |   "
        f"combined active pixels = {sum(diag['combined_active_per_well'])}",
        fontsize=11,
    )

    # ---- Left: detection view (non-bs active mean) ----
    for i_well, (t, y, color) in enumerate(zip(pre_time_min, pre_signals, WELL_COLORS)):
        ax_l.plot(t, y, color=color, linewidth=0.9, alpha=0.85,
                  label=f"well {i_well}")
        # Per-well cut: thin dashed line (only if an artefact was detected)
        if detect.per_well_had_artefact[i_well]:
            ax_l.axvline(detect.per_well_cut_min[i_well],
                         color=color, linewidth=0.6, linestyle=":",
                         alpha=0.6)

    # Chip-wide cut: thick solid line on top.
    ax_l.axvline(detect.chip_cut_min, color="black", linewidth=1.8,
                 linestyle="-",
                 label=f"chip cut = {detect.chip_cut_min:.2f} min")
    ax_l.set_xlabel("time [min]")
    ax_l.set_ylabel("Linearised signal [a.u.] (non-BS)")
    ax_l.set_title("Detection view: combined per-well mean with cut",
                   fontsize=10)
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=7, ncol=2, loc="best")
    # Zoom the left panel to the first 8 min so the artefact and the cut
    # are clearly visible (full trace is shown on the right anyway).
    ax_l.set_xlim(0, 8)

    # ---- Right: classifier-input view (BS active mean, post-trim) ----
    for i_well, (t, y, color) in enumerate(zip(post_time_min, post_signals_bs, WELL_COLORS)):
        ax_r.plot(t, y, color=color, linewidth=0.9, alpha=0.85,
                  label=f"well {i_well}")
    ax_r.axvline(0.0, color="black", linewidth=1.0, linestyle="-",
                 alpha=0.5)
    ax_r.set_xlabel("time [min] (re-zeroed at chip cut)")
    ax_r.set_ylabel("Baseline-subtracted signal (classifier input)")
    ax_r.set_title("Post-trim view: signal as fed to classifier",
                   fontsize=10)
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=7, ncol=2, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if out_path is None:
        out_dir = OUTPUT_ROOT / short_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trim_diagnostic.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # Print a small summary so the CLI run is self-documenting.
    print(f"  -> {out_path}")
    print(f"     chip cut: {detect.chip_cut_min:.3f} min "
          f"(idx={detect.chip_cut_idx}); "
          f"per-well cuts: "
          f"{[round(x, 2) for x in detect.per_well_cut_min]}")
    print(f"     had_artefact per well: {detect.per_well_had_artefact}")
    print(f"     combined active pixels per well: "
          f"{diag['combined_active_per_well']}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chip", required=True, type=Path,
                        help="Path to one chip folder in Data/Multi/")
    parser.add_argument("--n-wells", type=int, default=DEFAULT_N_WELLS)
    parser.add_argument("--n-a-type", default=DEFAULT_N_A_TYPE)
    parser.add_argument("--end-time-min", type=int, default=DEFAULT_END_TIME_MIN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"==> {args.chip.name}")
    plot_chip_trim(args.chip,
                   n_wells=args.n_wells,
                   n_a_type=args.n_a_type,
                   end_time_min=args.end_time_min,
                   out_path=args.out)


if __name__ == "__main__":
    main()
