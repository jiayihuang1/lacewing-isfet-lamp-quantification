"""RQ2 · F-D 4-class per-timestep segmentation framing. [Cat B] Report §RQ2 (future-work retained).

F-D — Full-trace 4-class segmentation with 1D U-Net (Paper 19 adaptation).

Adaptation of Joung et al. 2024 (paper 19) §3.6: U-Net trained to label
each timestep of the ISFET trace as one of 4 classes:
  0 = pre-amp        (t < ttp_idx - onset_window_samples)
  1 = onset-window   (ttp_idx - onset_window_samples <= t < ttp_idx)
  2 = amp            (ttp_idx <= t < ttp_idx + 90)
  3 = plateau        (t >= ttp_idx + 90)

TTP extraction at test time: first timestep where argmax ∈ {1, 2}
(onset-window or amp), converted to minutes via / SAMPLES_PER_MIN.

Provenance
----------
Backbone: 1D U-Net from Joung et al. 2024 §3.6; adapted to ISFET-LAMP
4-class per-timestep segmentation for onset detection.

Usage
-----
    python -m lacewing.quantification.methods.unet_segmentation \\
        --seed 0 --epochs 30

Output dir (canonical schema):
    Analysis/quantification/methods/results/
        p1_fd_unet_seg_ow{onset_window}_spatA3/seed{seed}/
            config.json
            history.json
            predictions.npz
            labels.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lacewing.classification.models.unet_1d import UNet1D
from lacewing.quantification.eval.schema import write_predictions, write_labels
from lacewing.quantification.regression_shared.data import dataset as reg_ds
from lacewing.quantification.methods.pdf_regression import _log10_concentration_array
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


SAMPLES_PER_MIN = 15
N_SAMPLES = 450
CACHE_STEM = "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"

# Class indices (semantic constants for readability).
CLS_PRE_AMP = 0
CLS_ONSET_WINDOW = 1
CLS_AMP = 2
CLS_PLATEAU = 3

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _make_4class_labels(
    ttp_min: np.ndarray,
    n_samples: int = N_SAMPLES,
    onset_window_samples: int = 30,
) -> np.ndarray:
    """Build per-pixel, per-timestep 4-class integer labels.

    Parameters
    ----------
    ttp_min:
        (n_pixels,) TTP in minutes.
    n_samples:
        Length of each trace.
    onset_window_samples:
        Width of the onset window in samples (ablation knob, default 30).

    Returns
    -------
    labels: (n_pixels, n_samples) int64
        Class per timestep for each pixel.
    """
    n_pixels = len(ttp_min)
    ttp_idx = (ttp_min * SAMPLES_PER_MIN).astype(np.float64)  # (n_pixels,)

    t = np.arange(n_samples, dtype=np.float64)  # (n_samples,)

    # Broadcast: (n_pixels, n_samples)
    ttp_idx_2d = ttp_idx[:, None]
    t_2d = t[None, :]

    labels = np.full((n_pixels, n_samples), CLS_PRE_AMP, dtype=np.int64)

    # onset-window: ttp_idx - onset_window_samples <= t < ttp_idx
    mask_ow = (t_2d >= ttp_idx_2d - onset_window_samples) & (t_2d < ttp_idx_2d)
    labels[mask_ow] = CLS_ONSET_WINDOW

    # amp: ttp_idx <= t < ttp_idx + 90
    mask_amp = (t_2d >= ttp_idx_2d) & (t_2d < ttp_idx_2d + 90)
    labels[mask_amp] = CLS_AMP

    # plateau: t >= ttp_idx + 90
    mask_plateau = t_2d >= ttp_idx_2d + 90
    labels[mask_plateau] = CLS_PLATEAU

    return labels


def _load_manual_labels_for_pixels(
    cache_pixel_ids: np.ndarray,
    manual_labels_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Match cache pixel ids to manual labels.

    Returns
    -------
    matched_mask : (n_cache_pixels,) bool
        True at indices where cache pixel has a manual label.
    matched_labels : (n_matched, 450) int64
        Per-timestep 4-class labels for those matched pixels, in the same
        order as they appear in cache_pixel_ids (subset via matched_mask).
    """
    with np.load(manual_labels_path, allow_pickle=False) as npz:
        m_pixel_ids = npz["pixel_ids"]     # (M,) int32
        m_labels = npz["labels"]           # (M, 450) int8

    # Build a lookup: manual pixel id -> row in m_labels.
    pixel_to_row = {int(p): i for i, p in enumerate(m_pixel_ids)}
    matched_mask = np.array([int(p) in pixel_to_row for p in cache_pixel_ids],
                            dtype=bool)
    matched_rows = [pixel_to_row[int(cache_pixel_ids[i])]
                    for i in np.flatnonzero(matched_mask)]
    if len(matched_rows) == 0:
        matched_labels = np.zeros((0, m_labels.shape[1]), dtype=np.int64)
    else:
        matched_labels = m_labels[matched_rows].astype(np.int64)
    return matched_mask, matched_labels


# ---------------------------------------------------------------------------
# Per-trace dataset
# ---------------------------------------------------------------------------

class _TraceDataset(Dataset):
    """Yields (1, T) traces with (T,) per-timestep 4-class integer labels."""

    def __init__(
        self,
        X: np.ndarray,
        ttp_min: np.ndarray,
        onset_window_samples: int = 30,
    ):
        # X: (n_pixels, T)  ttp_min: (n_pixels,)
        self.X = X.astype(np.float32)
        self.labels = _make_4class_labels(ttp_min, n_samples=X.shape[1],
                                          onset_window_samples=onset_window_samples)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])[None, :]           # (1, T)
        y = torch.from_numpy(self.labels[idx])                # (T,) int64
        return x, y


class _TraceDatasetFromLabels(Dataset):
    """Yields (1, T) traces with (T,) per-timestep 4-class integer labels,
    given pre-built labels (manual-labels mode)."""

    def __init__(self, X: np.ndarray, labels: np.ndarray):
        # X: (n_pixels, T)  labels: (n_pixels, T) int64
        self.X = X.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])[None, :]
        y = torch.from_numpy(self.labels[idx])
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(base_channels: int = 8) -> nn.Module:
    """1D U-Net backbone for per-timestep 4-class segmentation."""
    return UNet1D(in_channels=1, n_classes=4, base_channels=base_channels)


def _monotonicity_loss(logits: torch.Tensor) -> torch.Tensor:
    """Penalise the model for outputting non-monotonic class progression.

    logits: (B, n_classes=4, T). Computes expected progression per timestep
    (0*P[0]+1*P[1]+2*P[2]+3*P[3]) and penalises any negative time-derivative.
    """
    probs = torch.softmax(logits, dim=1)  # (B, 4, T)
    n_classes = probs.size(1)
    class_idx = torch.arange(n_classes, device=logits.device, dtype=torch.float32).view(1, n_classes, 1)
    progression = (probs * class_idx).sum(dim=1)  # (B, T)
    diffs = progression[:, 1:] - progression[:, :-1]  # (B, T-1)
    return torch.relu(-diffs).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    ckpt_dir: Path | None = None,
    mono_weight: float = 0.0,
    use_crf: bool = False,
) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    crf = None
    if use_crf:
        from lacewing.quantification.methods.monotonic_crf import MonotonicCRF
        crf = MonotonicCRF(n_classes=4).to(device)
    history = []
    for e in range(epochs):
        # Train step
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)             # (B, 1, T)
            yb = yb.to(device)             # (B, T) int64
            logits = model(xb)             # (B, 4, T)
            if use_crf:
                # logits: (B, 4, T) → emissions (B, T, 4)
                emissions = logits.permute(0, 2, 1)
                loss = crf(emissions, yb)
            else:
                # CrossEntropyLoss expects (B, C, T) logits and (B, T) targets.
                loss_ce = loss_fn(logits, yb)
                if mono_weight > 0.0:
                    loss = loss_ce + mono_weight * _monotonicity_loss(logits)
                else:
                    loss = loss_ce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
        avg_train_loss = total_loss / max(total_samples, 1)

        # Validation step (pixel-level accuracy)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)          # (B, 4, T)
                if use_crf:
                    emissions = logits.permute(0, 2, 1)
                    val_loss_acc += crf(emissions, yb).item() * yb.size(0)
                    preds = crf.decode(emissions)   # (B, T)
                else:
                    val_loss_acc += loss_fn(logits, yb).item() * yb.size(0)
                    preds = logits.argmax(dim=1)   # (B, T)
                val_correct += (preds == yb).sum().item()
                val_total += yb.numel()
        avg_val_loss = val_loss_acc / max(val_loader.dataset.__len__(), 1)
        val_acc = val_correct / max(val_total, 1)

        record = {
            "epoch": e,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(
            f"epoch {e + 1:02d}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": e + 1, "model_state": model.state_dict(),
                        "val_loss": avg_val_loss, "val_acc": val_acc},
                       ckpt_dir / "last.pt")
            if not hasattr(_train, "_best_seen") or avg_val_loss < _train._best_seen:
                _train._best_seen = avg_val_loss
                torch.save({"epoch": e + 1, "model_state": model.state_dict(),
                            "val_loss": avg_val_loss, "val_acc": val_acc},
                           ckpt_dir / "best.pt")
    return history


# ---------------------------------------------------------------------------
# Test-time prediction
# ---------------------------------------------------------------------------

def _predict_ttp(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    ttp_classes: frozenset[int] = frozenset({CLS_ONSET_WINDOW, CLS_AMP}),
    crf=None,
) -> np.ndarray:
    """Return (n_pixels,) predicted TTP in minutes.

    TTP = first timestep where argmax(logits) is in ``ttp_classes``,
    converted to minutes via / SAMPLES_PER_MIN.  Falls back to
    ``N_SAMPLES - 1`` if no such timestep exists.

    Default ``ttp_classes = {1, 2}`` matches the qlamp_rule 4-class
    schema (onset-window + amp).  Manual-labels mode should pass
    ``frozenset({2})`` (rising only) since the manual schema is
    {baseline=0, drift=1, rising=2, post-amp=3}.

    If ``crf`` is given (a MonotonicCRF instance), per-timestep class
    predictions come from ``crf.decode()`` on the emissions instead of
    plain argmax.
    """
    model.eval()
    n_pixels = len(X)
    ttp_pred = np.empty(n_pixels, dtype=np.float32)
    target_cls = np.array(sorted(ttp_classes), dtype=np.int64)

    with torch.no_grad():
        for i in range(n_pixels):
            x = torch.from_numpy(X[i].astype(np.float32))[None, None, :].to(device)
            # x: (1, 1, T)
            logits = model(x)               # (1, 4, T)
            if crf is not None:
                emissions = logits.permute(0, 2, 1)          # (1, T, 4)
                preds = crf.decode(emissions).squeeze(0).cpu().numpy()  # (T,)
            else:
                preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (T,)
            # First timestep where class is one of the target classes.
            onset_mask = np.isin(preds, target_cls)
            onset_indices = np.where(onset_mask)[0]
            if len(onset_indices) > 0:
                ttp_sample = int(onset_indices[0])
            else:
                ttp_sample = N_SAMPLES - 1
            ttp_pred[i] = ttp_sample / SAMPLES_PER_MIN

    return ttp_pred


def _predict_ttp_from_argmax(
    preds_argmax: np.ndarray,
    target_cls: frozenset[int] = frozenset({CLS_ONSET_WINDOW, CLS_AMP}),
    k_consecutive: int = 1,
) -> np.ndarray:
    """Extract per-pixel TTP from a pre-computed (N, T) per-timestep argmax.

    Rule: first timestep where K consecutive timesteps all predict a class
    in ``target_cls``.  Fallback = N_SAMPLES - 1 if no such window exists.

    Pure NumPy; used to re-score existing checkpoints under different rules
    without needing to run the model.
    """
    n_pixels, T = preds_argmax.shape
    target_arr = np.array(sorted(target_cls), dtype=np.int64)
    is_target = np.isin(preds_argmax, target_arr)  # (N, T) bool

    ttp_pred = np.full(n_pixels, T - 1, dtype=np.float32)
    for i in range(n_pixels):
        row = is_target[i]
        if k_consecutive <= 1:
            idxs = np.where(row)[0]
            if len(idxs):
                ttp_pred[i] = idxs[0]
        else:
            # Find first index where the next K entries (inclusive) are all True.
            for t in range(T - k_consecutive + 1):
                if row[t : t + k_consecutive].all():
                    ttp_pred[i] = t
                    break

    return (ttp_pred / SAMPLES_PER_MIN).astype(np.float32)


def _save_test_time_argmax(
    model: nn.Module,
    X: np.ndarray,
    device: str,
    out_path: Path,
) -> None:
    """Run model on X, save per-timestep argmax + softmax probs to npz.

    Output file has:
        argmax:  (N, T) int8   — argmax over 4 classes per (pixel, timestep)
        softmax: (N, 4, T) float16 — full posteriors, for smoothed / Viterbi rules

    Emitted alongside predictions.npz; used by fd_extraction_sweep.py.
    """
    model.eval()
    n_pixels = len(X)
    argmax_out = np.empty((n_pixels, N_SAMPLES), dtype=np.int8)
    softmax_out = np.empty((n_pixels, 4, N_SAMPLES), dtype=np.float16)
    with torch.no_grad():
        for i in range(n_pixels):
            x = torch.from_numpy(X[i].astype(np.float32))[None, None, :].to(device)
            logits = model(x)                                    # (1, 4, T)
            argmax_out[i] = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int8)
            softmax_out[i] = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float16)
    np.savez_compressed(out_path, argmax=argmax_out, softmax=softmax_out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--onset_window_samples", type=int, default=30,
        help="Width of the onset-window class in samples (ablation knob, default 30). "
             "Sweep: {30, 60, 90}.",
    )
    parser.add_argument(
        "--base_channels", type=int, default=8,
        help="U-Net base channel width (default 8 → ~500k params; use 16 for ~2M).",
    )
    parser.add_argument(
        "--max_train_pixels", type=int, default=None,
        help="Subsample training set to at most N pixels (for smoke-testing on CPU).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--labels-mode", choices=["qlamp_rule", "manual"],
        default="qlamp_rule",
        help="qlamp_rule = derive 4-class labels from qLAMP TTP (default, backward compat). "
             "manual = load hand-labelled boundaries from manual_labels.npz.",
    )
    parser.add_argument("--backbone", default="unet",
                        choices=["unet", "gru", "cnn_gru_par"],
                        help="F-D backbone. Default 'unet' (backward-compat).")
    parser.add_argument(
        "--manual-labels-path", type=Path,
        default=LACEWING_PKG_DIR / "quantification" / "methods" / "fd_labelling" / "manual_labels.npz",
        help="Path to manual_labels.npz (used only when --labels-mode manual).",
    )
    parser.add_argument(
        "--internal-val-frac", type=float, default=0.1,
        help="In manual mode, fraction of labelled pixels held out for internal val loss.",
    )
    parser.add_argument(
        "--monotonicity-weight", type=float, default=0.0,
        help="Weight for the training-time monotonicity-loss term added to the CE "
             "loss (0.0 = disabled, backward-compat default). See _monotonicity_loss.",
    )
    parser.add_argument(
        "--use-monotonic-crf", action="store_true", default=False,
        help="Replace CE loss with a monotonic-CRF NLL loss and argmax inference "
             "with CRF Viterbi decode (default False, backward-compat). See "
             "lacewing.quantification.methods.monotonic_crf.MonotonicCRF. Mutually "
             "exclusive in practice with --monotonicity-weight (CRF hard-constrains "
             "monotonicity already).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- load + split -----
    print(f"Loading cache: {CACHE_STEM}")
    arr = reg_ds.make_split(
        CACHE_STEM,
        seed=args.seed,
        val_n_chips=1,
        normalise_y=False,
    )
    print(
        f"  n_tr={len(arr.X_tr)}  n_va={len(arr.X_va)}  n_te={len(arr.X_te)}"
    )

    if args.labels_mode == "manual":
        if not args.manual_labels_path.exists():
            raise SystemExit(f"manual labels not found: {args.manual_labels_path}")
        # Match cache-train pixels to manual labels.
        matched_mask, matched_labels = _load_manual_labels_for_pixels(
            arr.pixel_tr, args.manual_labels_path
        )
        n_matched = int(matched_mask.sum())
        if n_matched < 500:
            raise SystemExit(
                f"only {n_matched} training pixels have manual labels — need >=500. "
                f"Label more wells."
            )
        X_labelled = arr.X_tr[matched_mask]  # (n_matched, T)
        # Internal split: last internal_val_frac of shuffled indices for val.
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_matched)
        n_val_internal = max(1, int(n_matched * args.internal_val_frac))
        val_idx = np.sort(perm[:n_val_internal])
        train_idx = np.sort(perm[n_val_internal:])
        X_tr = X_labelled[train_idx]
        y_tr_labels = matched_labels[train_idx]
        X_va = X_labelled[val_idx]
        y_va_labels = matched_labels[val_idx]
        print(
            f"  [manual] {n_matched} training pixels matched; "
            f"internal split {len(train_idx)} train / {len(val_idx)} val"
        )

        # Optional subsampling on the manual-labelled train subset.
        if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
            rng2 = np.random.default_rng(args.seed)
            idx2 = rng2.choice(len(X_tr), size=args.max_train_pixels, replace=False)
            idx2.sort()
            X_tr = X_tr[idx2]
            y_tr_labels = y_tr_labels[idx2]
            print(f"  [debug] subsampled manual train to {len(X_tr)} pixels")

        train_ds = _TraceDatasetFromLabels(X_tr, y_tr_labels)
        val_ds = _TraceDatasetFromLabels(X_va, y_va_labels)
        n_train_pixels = len(X_tr)
        n_val_pixels = len(X_va)
    else:
        X_tr, y_tr = arr.X_tr, arr.y_tr
        X_va, y_va = arr.X_va, arr.y_va

        # Optional subsampling for fast smoke-tests.
        if args.max_train_pixels is not None and len(X_tr) > args.max_train_pixels:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(X_tr), size=args.max_train_pixels, replace=False)
            idx.sort()
            X_tr = X_tr[idx]
            y_tr = y_tr[idx]
            print(f"  [debug] subsampled train to {len(X_tr)} pixels")

        train_ds = _TraceDataset(X_tr, y_tr, onset_window_samples=args.onset_window_samples)
        val_ds = _TraceDataset(X_va, y_va, onset_window_samples=args.onset_window_samples)
        n_train_pixels = len(X_tr)
        n_val_pixels = len(X_va)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(
        f"  total train traces = {len(train_ds)}  "
        f"total val traces = {len(val_ds)}"
    )

    # ----- build model -----
    from lacewing.quantification.methods.unet_segmentation_backbones import build_fd_backbone
    model = build_fd_backbone(args.backbone, n_classes=4).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Built backbone={args.backbone!r}  ({n_params:,} params)")

    # ----- construct out_dir BEFORE training so checkpoints can be saved -----
    bb_tag = f"_bb_{args.backbone}" if args.backbone != "unet" else ""
    mono_tag = f"_monoW{args.monotonicity_weight:g}" if args.monotonicity_weight > 0.0 else ""
    crf_tag = "_crf" if args.use_monotonic_crf else ""
    if args.labels_mode == "manual":
        out_dir = RESULTS_ROOT / f"p1_fd_manual4_spatA3{bb_tag}{mono_tag}{crf_tag}" / f"seed{args.seed}"
    else:
        ow = args.onset_window_samples
        out_dir = (
            RESULTS_ROOT
            / f"p1_fd_unet_seg_ow{ow}_spatA3{bb_tag}{mono_tag}{crf_tag}"
            / f"seed{args.seed}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- train -----
    # Reset _best_seen state to avoid cross-run contamination.
    if hasattr(_train, "_best_seen"):
        delattr(_train, "_best_seen")
    history = _train(
        model, train_loader, val_loader, args.device, args.epochs, args.lr,
        ckpt_dir=out_dir / "checkpoints",
        mono_weight=args.monotonicity_weight,
        use_crf=args.use_monotonic_crf,
    )

    # ----- test-time eval -----
    print("Running test-time segmentation and TTP extraction...")
    if args.labels_mode == "manual":
        # Manual schema: {baseline=0, drift=1, rising=2, post-amp=3}.
        # TTP = first "rising" timestep.
        ttp_classes = frozenset({CLS_AMP})
    else:
        # qlamp_rule schema: {pre-amp=0, onset-window=1, amp=2, plateau=3}.
        ttp_classes = frozenset({CLS_ONSET_WINDOW, CLS_AMP})
    eval_crf = None
    if args.use_monotonic_crf:
        from lacewing.quantification.methods.monotonic_crf import MonotonicCRF
        eval_crf = MonotonicCRF(n_classes=4).to(args.device)
    ttp_pred = _predict_ttp(model, arr.X_te, args.device, ttp_classes=ttp_classes, crf=eval_crf)
    print(f"  ttp_pred_min range: [{ttp_pred.min():.2f}, {ttp_pred.max():.2f}]")

    # ----- write outputs -----
    if args.labels_mode == "manual":
        class_semantics = {
            "0": "baseline",
            "1": "drift",
            "2": "rising",
            "3": "post-amp",
        }
    else:
        class_semantics = {
            "0": "pre-amp",
            "1": "onset-window",
            "2": "amp",
            "3": "plateau",
        }

    cfg = {
        "method": "P1.4 F-D 4-class segmentation (Paper 19 / Joung et al. 2024 adaptation)",
        "backbone": args.backbone,
        "labels_mode": args.labels_mode,
        "manual_labels_path": (
            str(args.manual_labels_path) if args.labels_mode == "manual" else None
        ),
        "seed": args.seed,
        "onset_window_samples": (
            args.onset_window_samples if args.labels_mode == "qlamp_rule" else None
        ),
        "internal_val_frac": (
            args.internal_val_frac if args.labels_mode == "manual" else None
        ),
        "monotonicity_weight": args.monotonicity_weight,
        "use_monotonic_crf": args.use_monotonic_crf,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_train_pixels": args.max_train_pixels,
        "device": args.device,
        "n_train_pixels": n_train_pixels,
        "n_val_pixels": n_val_pixels,
        "n_test_pixels": len(arr.X_te),
        "n_params": n_params,
        "samples_per_min": SAMPLES_PER_MIN,
        "cache_stem": CACHE_STEM,
        "ttp_classes": sorted(ttp_classes),
        "class_semantics": class_semantics,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Canonical schema.
    write_predictions(
        out_dir / "predictions.npz",
        ttp_pred_min=ttp_pred.astype(np.float32),
    )

    log10_conc = _log10_concentration_array(arr.chip_te, arr.well_te)

    write_labels(
        out_dir / "labels.npz",
        ttp_true_min=arr.y_te.astype(np.float32),
        chip_id=arr.chip_te,
        well_id=arr.well_te.astype(np.int32),
        log10_concentration=log10_conc,
        split=np.full(len(arr.X_te), "test", dtype="U16"),
    )

    print(f"[ok] wrote {out_dir}/")
    print(f"  predictions.npz  labels.npz  config.json  history.json")


if __name__ == "__main__":
    main()
