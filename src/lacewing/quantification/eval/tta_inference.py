"""Test-time augmentation for a trained F-A or F-B checkpoint.

Perturbs each test trace via (time-shift × amplitude-scale) grid, runs
inference N times per pixel, averages predictions, applies the framing's
default extraction rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lacewing.quantification.eval.scoreboard import compute_all_metrics


TIME_SHIFTS = [-3, -2, -1, 0, 1, 2, 3]
AMPLITUDE_SCALES = [0.95, 1.00, 1.05]
SAMPLES_PER_MIN = 15

HERE = Path(__file__).resolve().parent
METHODS_RESULTS_ROOT = HERE.parent / "methods" / "results"
OUT_CSV = HERE / "tta_scoreboard.csv"


def generate_augmentations(X: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
    """Return list of (shift, scale, augmented_X)."""
    out = []
    for shift in TIME_SHIFTS:
        rolled = np.roll(X, shift, axis=-1)
        for scale in AMPLITUDE_SCALES:
            out.append((shift, scale, (rolled * scale).astype(np.float32)))
    return out


def _reload_fa_model(ckpt_path: Path, cfg: dict, device: str) -> torch.nn.Module:
    from lacewing.quantification.methods.pdf_regression import CNN1DPerStep, N_SAMPLES
    m = CNN1DPerStep(t=int(cfg.get("n_samples", N_SAMPLES))).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def _reload_fb_model(ckpt_path: Path, cfg: dict, device: str) -> torch.nn.Module:
    from lacewing.quantification.methods.sliding_window_cls import build_model
    m = build_model(
        window=int(cfg["window"]),
        backbone=cfg.get("backbone", "cnn_gru_par"),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # F-B saves state_dict directly (no wrapper).
    if "model_state" in ckpt:
        m.load_state_dict(ckpt["model_state"])
    else:
        m.load_state_dict(ckpt)
    m.eval()
    return m


def _fa_predict_ttp(model, X: np.ndarray, device: str) -> np.ndarray:
    """Return (n_pixels,) TTP in minutes via hard argmax on density."""
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(device)  # (N, 1, T)
        densities = []
        # Chunk for memory.
        for i in range(0, x.shape[0], 256):
            densities.append(model(x[i:i + 256]).cpu().numpy())
        density = np.concatenate(densities, axis=0)
    idx = density.argmax(axis=1)
    return idx.astype(np.float32) / float(SAMPLES_PER_MIN)


def _fb_predict_ttp(model, X: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    from lacewing.quantification.methods._windowing import (
        slide_windows,
        aggregate_to_ttp_first_positive,
    )
    window = int(cfg["window"])
    stride = int(cfg["stride"])
    k_consec = int(cfg.get("k_consecutive", 3))
    k_thr = float(cfg.get("k_thr", 0.5))
    ws = slide_windows(X, window, stride)   # (n_pixels, N_windows, W)
    n_p, n_w, _ = ws.shape
    probs = np.empty((n_p, n_w), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_p):
            xb = torch.from_numpy(ws[i]).unsqueeze(1).to(device)
            logits = model(xb)
            probs[i] = torch.sigmoid(logits).cpu().numpy()
    return aggregate_to_ttp_first_positive(
        probs, threshold=k_thr, k_consecutive=k_consec,
        window=window, stride=stride, samples_per_min=SAMPLES_PER_MIN,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", required=True, help="Seed dir under methods/results/")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    seed_dir = METHODS_RESULTS_ROOT / args.dir
    cfg = json.loads((seed_dir / "config.json").read_text())
    # F-A saves test set as {chip_te, well_te, y_te}. Reload from labels.npz.
    # We also need X_te. Reload from the regression cache.
    from lacewing.quantification.regression_shared.data import dataset as reg_ds
    arr = reg_ds.make_split(
        cfg.get("cache", "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3"),
        seed=int(cfg["seed"]),
        val_n_chips=int(cfg.get("val_n_chips", 1)),
        normalise_y=False,
    )
    X_te = arr.X_te
    y_te = arr.y_te

    # Detect framing.
    is_fa = "sigma_samples" in cfg or "F6 PDF regression" in cfg.get("method", "")
    ckpt_path = seed_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no best.pt found in {seed_dir / 'checkpoints'}; "
            f"F-B runs must be trained after Task 4 (which adds checkpoint saving)."
        )

    if is_fa:
        model = _reload_fa_model(ckpt_path, cfg, args.device)
        predict_fn = lambda X: _fa_predict_ttp(model, X, args.device)
    else:
        model = _reload_fb_model(ckpt_path, cfg, args.device)
        predict_fn = lambda X: _fb_predict_ttp(model, X, cfg, args.device)

    # Run inference on each augmentation.
    augs = generate_augmentations(X_te)
    print(f"[tta] {len(augs)} augmentations × {len(X_te)} pixels")
    preds = []
    for shift, scale, X_aug in augs:
        ttp = predict_fn(X_aug)
        preds.append(ttp)
        print(f"  shift={shift:+d} scale={scale:.2f} → MAE={np.mean(np.abs(ttp - y_te)):.3f}")

    ttp_pred_min = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

    # Save + score.
    np.savez_compressed(seed_dir / "predictions_tta.npz",
                        ttp_pred_min=ttp_pred_min)

    from lacewing.quantification.eval.schema import PredictionArrays, read_labels
    from lacewing.quantification.eval.scoreboard import append_scoreboard_row, compute_all_metrics
    preds_struct = PredictionArrays(ttp_pred_min=ttp_pred_min, extras={})
    labels_struct = read_labels(seed_dir / "labels.npz")
    metrics = compute_all_metrics(preds_struct, labels_struct, split="test")
    method_id = f"tta_{seed_dir.parent.name}_{seed_dir.name}"
    cfg_out = {"framing": "fa" if is_fa else "fb",
               "n_augs": len(augs),
               "time_shifts": TIME_SHIFTS,
               "amplitude_scales": AMPLITUDE_SCALES,
               "source_config": cfg}
    append_scoreboard_row(OUT_CSV, method_id=method_id, config_json=cfg_out, metrics=metrics)
    print(f"[tta] {method_id} MAE={metrics.get('per_well_mae_min', float('nan')):.3f}")


if __name__ == "__main__":
    main()
