"""matplotlib click-UI for hand-labelling amp-trace segment boundaries.

Given a trace + qLAMP TTP, the labeller marks 3 boundaries:
  - b1: baseline → drift
  - b2: drift → rising  (this IS the TTP under the ML label — should ≈ qLAMP TTP)
  - b3: rising → post-amp

Empty classes are represented by degenerate boundaries (b1==b2 means no drift,
b1==0 means no baseline, b3==450 means no post-amp).

Keypresses (chosen to avoid matplotlib's default keymap — 's' saves the figure,
'k' toggles log-scale, 'q' closes without our handler running, etc.):
  enter      save current 3 boundaries as status="ok"
  n          save as status="invalid" (no rising visible; exclude from training)
  space      skip / defer (do not save; leave cluster unlabelled)
  backspace  undo last boundary
  escape     quit entire session (returns None; orchestrator checks quit_requested())

Mouse:
  left-click   add boundary at click x (rounded to nearest sample index)
  right-click  remove most-recent boundary
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


CLS_BASELINE = 0
CLS_DRIFT = 1
CLS_RISING = 2
CLS_POST_AMP = 3

N_SAMPLES = 450

_CLASS_LABELS = ["baseline", "drift", "rising", "post-amp"]

# Module-level flag: set to True when user presses 'escape'.
_QUIT_REQUESTED = False


def quit_requested() -> bool:
    return _QUIT_REQUESTED


def _reset_quit() -> None:
    """For tests / new sessions."""
    global _QUIT_REQUESTED
    _QUIT_REQUESTED = False


def validate_boundaries(b1: int, b2: int, b3: int, n_samples: int = N_SAMPLES) -> None:
    """Raises ValueError if 0 <= b1 <= b2 <= b3 <= n_samples not satisfied."""
    for name, val in [("b1", b1), ("b2", b2), ("b3", b3)]:
        if val < 0 or val > n_samples:
            raise ValueError(f"{name}={val} out of range [0, {n_samples}]")
    if not (b1 <= b2 <= b3):
        raise ValueError(
            f"boundaries must be monotonic non-decreasing: b1={b1} b2={b2} b3={b3}"
        )


def label_cluster(
    trace: np.ndarray,
    ttp_true_min: float,
    well_id: str,
    cluster_id: int,
    samples_per_min: int = 15,
) -> dict | None:
    """Show trace + collect 3 boundaries. See module docstring for keybinds.

    Returns dict with keys 'boundaries' (tuple or None) and 'status' ('ok' or 'invalid'),
    OR None if the user skipped/quit.
    """
    state = {
        "boundaries": [],   # list of ints, at most 3
        "result": None,     # dict when finalised
    }

    fig, ax = plt.subplots(figsize=(11, 4))
    t = np.arange(len(trace))
    ax.plot(t, trace, "-", lw=1.2, color="#1f77b4")
    ttp_idx = ttp_true_min * samples_per_min
    ax.axvline(ttp_idx, color="red", linestyle="--", lw=1.4, alpha=0.7,
               label=f"qLAMP TTP = {ttp_true_min:.2f} min ({ttp_idx:.0f} samples)")
    ax.set_xlabel("sample index")
    ax.set_ylabel("ISFET value")
    ax.set_xlim(0, N_SAMPLES)
    ax.legend(loc="upper right", fontsize=9)

    def _redraw_boundaries() -> None:
        # Remove old boundary artists.
        for line in list(ax.lines):
            if getattr(line, "_boundary_marker", False):
                line.remove()
        for txt in list(ax.texts):
            if getattr(txt, "_boundary_marker", False):
                txt.remove()
        # Draw current boundaries.
        for i, b in enumerate(state["boundaries"]):
            ln = ax.axvline(b, color="grey", linestyle="-", lw=1.4, alpha=0.8)
            ln._boundary_marker = True
            label = f"b{i+1}: {_CLASS_LABELS[i]}→{_CLASS_LABELS[i+1]}"
            y_pos = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            tx = ax.text(b + 3, y_pos, label, fontsize=8, color="grey", rotation=90,
                         va="bottom")
            tx._boundary_marker = True
        ax.set_title(
            f"well={well_id} cluster={cluster_id} — {len(state['boundaries'])}/3 boundaries. "
            f"[enter=save  n=invalid  space=skip  backspace=undo  esc=quit]"
        )
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 1:  # left
            if len(state["boundaries"]) >= 3:
                print("  already have 3 boundaries — press enter/n/backspace/space")
                return
            b = int(round(event.xdata))
            b = max(0, min(N_SAMPLES, b))
            # Enforce monotonic: new boundary must be >= last.
            if state["boundaries"] and b < state["boundaries"][-1]:
                print(f"  boundary {b} < previous {state['boundaries'][-1]} — clamping to previous")
                b = state["boundaries"][-1]
            state["boundaries"].append(b)
            _redraw_boundaries()
        elif event.button == 3:  # right = undo
            if state["boundaries"]:
                state["boundaries"].pop()
                _redraw_boundaries()

    def _on_key(event) -> None:
        global _QUIT_REQUESTED
        # Keys chosen to avoid matplotlib default keymap:
        #   's' saves the figure (dialog), 'k' toggles x-log, 'q' closes without
        #   our handler running, 'p' pans, 'l' toggles y-log. Ours use enter /
        #   n / space / backspace / escape.
        if event.key == "enter":
            if len(state["boundaries"]) != 3:
                print(f"  need 3 boundaries; have {len(state['boundaries'])}")
                return
            b1, b2, b3 = state["boundaries"]
            try:
                validate_boundaries(b1, b2, b3)
            except ValueError as e:
                print(f"  invalid: {e}")
                return
            state["result"] = {"boundaries": (b1, b2, b3), "status": "ok"}
            plt.close(fig)
        elif event.key == "n":
            state["result"] = {"boundaries": None, "status": "invalid"}
            plt.close(fig)
        elif event.key == " ":  # matplotlib reports space as literal " "
            state["result"] = None
            plt.close(fig)
        elif event.key == "backspace":
            if state["boundaries"]:
                state["boundaries"].pop()
                _redraw_boundaries()
        elif event.key == "escape":
            _QUIT_REQUESTED = True
            state["result"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw_boundaries()
    plt.show(block=True)

    return state["result"]
