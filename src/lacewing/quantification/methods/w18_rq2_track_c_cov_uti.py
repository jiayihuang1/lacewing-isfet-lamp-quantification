"""W18 RQ2 Track C — Joint cls+quant training on COV+UTI mixed data.

Builds the a3+alpha0.5 joint model (Track C winner architecture) and trains
it from scratch (or from an optional SSL-pretrained encoder) on the merged
COV+UTI regression cache: ALL COV amp-positive pixels plus UTI amp-positive
pixels from all chips except a held-out UTI chip (used as test) and one
validation UTI chip, leave-one-UTI-chip-out.

Uses the merged cache (cache/merged_cov_uti_reg_cache.npz), which has
per-pixel signals + per-pixel TTP target (`y_ttp_min`) for both COV and UTI
origins, tagged by `origin` ("cov"/"uti") and `chip_id`.

Unlike b4_finetune_uti.py (which fine-tunes a COV-trained joint-model
checkpoint at low LR), this script optionally loads only the SSL-pretrained
encoder body via `--pretrained-encoder` (state_dict loaded with
strict=False into `model.backbone.unet`, stripping any "encoder." prefix
from key names — mirrors joint_cls_quant.py's `--pretrained-encoder`
handling). The cls/quant heads always start from fresh init. Default LR is
therefore the Track C default (1e-3), not a fine-tuning LR.

Output dir:
    Analysis/quantification/methods/results/w18_rq2_track_c_cov_uti/
        seed{S}_holdout_{chip_tag}/
            config.json / history.json / predictions.npz / labels.npz / checkpoints/

Usage:
    python -m lacewing.quantification.methods.w18_rq2_track_c_cov_uti \\
        --seed 0 --holdout-chip uti_260728_EC_SD --epochs 30 \\
        --pretrained-encoder Analysis/quantification/methods/results/p4_ssl_contrastive_unet_cov_uti_ep100/pretrained.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lacewing.quantification.methods.joint_cls_quant import build_joint_model
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
MERGED_CACHE = LACEWING_PKG_DIR / "quantification" / "methods" / "cache" / "merged_cov_uti_reg_cache.npz"
RESULTS_ROOT = LACEWING_PKG_DIR / "quantification" / "methods" / "results" / "w18_rq2_track_c_cov_uti"

# Joint-model config MUST match the Track C winner's config exactly.
ARCH = "a3"
ALPHA = 0.5
WINDOW = 120
STRIDE = 10
SAMPLES_PER_MIN = 15  # matches unet_segmentation.SAMPLES_PER_MIN


def _load_merged_amp_positive(seed: int, holdout_chip: str) -> dict:
    """Load merged COV+UTI cache, split by chip, return train/val/test arrays.

    Test = held-out UTI chip. Val = one other UTI chip, picked deterministically
    from `seed`. Train = ALL COV pixels + UTI pixels from every chip that is
    neither the held-out test chip nor the val chip.

    Uses `y_ttp_min` (per-well/per-plate TTP target) as the training target,
    matching the merged cache schema (Task 2/3).
    """
    npz = np.load(MERGED_CACHE, allow_pickle=False)
    X = npz["X"]                    # (N, 450)
    y = npz["y_ttp_min"]            # (N,)
    chip_id = npz["chip_id"]
    well = npz["well_id"]
    origin = npz["origin"]

    is_uti = origin == "uti"
    is_cov = origin == "cov"

    uti_chips = sorted(np.unique(chip_id[is_uti]).tolist())
    if holdout_chip not in uti_chips:
        raise SystemExit(f"holdout_chip={holdout_chip!r} not among UTI chips {uti_chips}")

    is_test = is_uti & (chip_id == holdout_chip)

    remaining_uti = [c for c in uti_chips if c != holdout_chip]
    if len(remaining_uti) < 1:
        raise SystemExit(f"need >=1 remaining UTI chip for val; got {remaining_uti}")
    rng = np.random.default_rng(seed)
    val_chip = remaining_uti[int(rng.integers(len(remaining_uti)))]  # deterministic from seed
    is_val = is_uti & (chip_id == val_chip)

    is_train = ~(is_test | is_val)

    return {
        "X_tr": X[is_train], "y_tr": y[is_train],
        "X_va": X[is_val], "y_va": y[is_val],
        "X_te": X[is_test], "y_te": y[is_test],
        "chip_te": chip_id[is_test], "well_te": well[is_test],
        "val_chip": val_chip,
        "n_train_cov_px": int((is_train & is_cov).sum()),
        "n_train_uti_px": int((is_train & is_uti).sum()),
    }


def _sliding_windows(X: np.ndarray, y: np.ndarray, window: int, stride: int,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chop each (450,) pixel trace into overlapping windows.

    Returns:
      Xw: (N * n_windows, window)  float32
      yw: (N * n_windows,)         float32  — same TTP for every window
      cls: (N * n_windows,)         float32  — 1 = pixel with a real TTP (amp+)
    """
    N, T = X.shape
    n_win = 1 + (T - window) // stride
    starts = np.arange(n_win) * stride
    Xw = np.stack([X[:, s:s + window] for s in starts], axis=1).reshape(-1, window)
    yw = np.repeat(y, n_win)
    cls = np.ones(len(yw), dtype=np.float32)
    return Xw.astype(np.float32), yw.astype(np.float32), cls


def _predict_ttp_per_pixel(model, X_te: np.ndarray, window: int, stride: int,
                            device: str) -> np.ndarray:
    """Predict TTP per pixel by averaging window-level quant outputs."""
    Xw, _, _ = _sliding_windows(X_te, np.zeros(len(X_te)), window, stride)
    n_win_per = 1 + (X_te.shape[1] - window) // stride
    with torch.no_grad():
        outs = []
        for i in range(0, len(Xw), 512):
            xb = torch.from_numpy(Xw[i:i + 512]).float().unsqueeze(1).to(device)
            _, quant = model(xb)
            outs.append(quant.cpu().numpy())
    q = np.concatenate(outs).reshape(-1, n_win_per)
    return q.mean(axis=1)  # (N,) minutes


def _load_pretrained_encoder(model, ckpt_path: Path, device: str) -> None:
    """Load an SSL-pretrained encoder state_dict into `model.backbone.unet`.

    Mirrors joint_cls_quant.py:351-392 — strips an "encoder." (or "unet.")
    prefix from key names, drops keys whose shape mismatches the target
    (e.g. an SSL classifier head with a different n_classes), then loads
    with strict=False so only the encoder body transfers.
    """
    ssl_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ssl_state, dict) and "model_state" in ssl_state:
        ssl_state = ssl_state["model_state"]

    stripped = {}
    for k, v in ssl_state.items():
        if k.startswith("encoder."):
            stripped[k[len("encoder."):]] = v
        elif k.startswith("unet."):
            stripped[k[len("unet."):]] = v
        else:
            stripped[k] = v

    target_shapes = {k: v.shape for k, v in model.backbone.unet.state_dict().items()}
    for k in list(stripped.keys()):
        if k in target_shapes and stripped[k].shape != target_shapes[k]:
            del stripped[k]

    missing, unexpected = model.backbone.unet.load_state_dict(stripped, strict=False)
    print(f"[pretrained-encoder] loaded from {ckpt_path.name}: "
          f"missing={len(missing)} unexpected={len(unexpected)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-chip", type=str, default="uti_260728_EC_SD")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pretrained-encoder", type=str, default=None,
                    help="Optional path to an SSL-pretrained encoder state_dict "
                         "(e.g. p4_ssl_contrastive_unet_cov_uti_ep100/pretrained.pt). "
                         "Loads via strict=False into model.backbone.unet. If not "
                         "passed, the model trains from scratch (fresh init).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    data = _load_merged_amp_positive(args.seed, args.holdout_chip)
    Xw_tr, yw_tr, cls_tr = _sliding_windows(data["X_tr"], data["y_tr"], WINDOW, STRIDE)
    Xw_va, yw_va, cls_va = _sliding_windows(data["X_va"], data["y_va"], WINDOW, STRIDE)

    print(f"seed={args.seed} holdout={args.holdout_chip}")
    print(f"  train windows: {len(Xw_tr):,}  from {len(data['X_tr']):,} px "
          f"(cov={data['n_train_cov_px']:,} uti={data['n_train_uti_px']:,})")
    print(f"  val windows:   {len(Xw_va):,}  from {len(data['X_va']):,} px  (chip={data['val_chip']})")
    print(f"  test pixels:   {len(data['X_te']):,}  (chip={args.holdout_chip})")

    # Build joint model + optionally load SSL-pretrained encoder body.
    model = build_joint_model(arch=ARCH, window=WINDOW, alpha=ALPHA).to(args.device)
    if args.pretrained_encoder is not None:
        ckpt_path = Path(args.pretrained_encoder)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No pretrained encoder at {ckpt_path}")
        _load_pretrained_encoder(model, ckpt_path, args.device)
    else:
        print("[pretrained-encoder] none passed — training from scratch (fresh init)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    quant_loss_fn = torch.nn.MSELoss()
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(
        torch.from_numpy(Xw_tr).unsqueeze(1),
        torch.from_numpy(yw_tr),
        torch.from_numpy(cls_tr),
    )
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    history = []
    for e in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb, clsb in tr_loader:
            xb, yb, clsb = xb.to(args.device), yb.to(args.device), clsb.to(args.device)
            cls_logit, quant = model(xb)
            loss = ALPHA * quant_loss_fn(quant, yb) + (1 - ALPHA) * cls_loss_fn(cls_logit, clsb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss) * len(xb); n += len(xb)
        train_loss = total_loss / max(1, n)

        # Val MAE at per-pixel level
        model.eval()
        va_pred = _predict_ttp_per_pixel(model, data["X_va"], WINDOW, STRIDE, args.device)
        va_mae = float(np.mean(np.abs(va_pred - data["y_va"])))
        print(f"  epoch {e+1}/{args.epochs}  train_loss={train_loss:.4f}  val_mae={va_mae:.3f} min")
        history.append({"epoch": e + 1, "train_loss": train_loss, "val_mae_min": va_mae})

    # Test inference
    test_pred = _predict_ttp_per_pixel(model, data["X_te"], WINDOW, STRIDE, args.device)
    per_well = {}
    for w in np.unique(data["well_te"]):
        m = data["well_te"] == w
        per_well[int(w)] = {
            "true": float(data["y_te"][m][0]),
            "pred_med": float(np.median(test_pred[m])),
            "n_px": int(m.sum()),
            "abs_err": float(abs(np.median(test_pred[m]) - data["y_te"][m][0])),
        }
    mae = float(np.mean([v["abs_err"] for v in per_well.values()]))
    print(f"\n=== Test on {args.holdout_chip}: per-well median MAE = {mae:.2f} min ===")
    for w, v in sorted(per_well.items()):
        print(f"  well {w}: true={v['true']:.2f} pred_med={v['pred_med']:.2f} |err|={v['abs_err']:.2f}")

    # Save
    out_dir = RESULTS_ROOT / f"seed{args.seed}_holdout_{args.holdout_chip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict()}, out_dir / "checkpoints" / "last.pt")
    np.savez_compressed(out_dir / "predictions.npz", ttp_pred_min=test_pred.astype(np.float32))
    np.savez_compressed(out_dir / "labels.npz",
                        ttp_true_min=data["y_te"].astype(np.float32),
                        chip_id=data["chip_te"], well_id=data["well_te"].astype(np.int32))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "arch": ARCH, "alpha": ALPHA, "window": WINDOW, "stride": STRIDE,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "seed": args.seed, "holdout_chip": args.holdout_chip,
        "val_chip": data["val_chip"],
        "pretrained_encoder": str(args.pretrained_encoder) if args.pretrained_encoder else None,
        "cache": str(MERGED_CACHE.relative_to(PROJECT_ROOT)),
        "n_train_px": int(len(data["X_tr"])),
        "n_train_cov_px": data["n_train_cov_px"],
        "n_train_uti_px": data["n_train_uti_px"],
        "n_val_px": int(len(data["X_va"])),
        "n_test_px": int(len(data["X_te"])),
        "test_per_well_mae_min": mae,
    }, indent=2))
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
