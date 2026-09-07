"""RQ1 · KP + CoV-NTC mix preprocessing ablation. [Cat A] Report §NewData KP RQ1.

RQ1 COV+UTI mixed training — leave-one-UTI-chip-out; all COV in train.

For each (model, seed, holdout_chip) combination:
    - Load merged_cov_uti_cls_cache.npz
    - Split: test = UTI pixels for holdout_chip; val = 1 of the remaining UTI
      chips (deterministic from seed); train = all COV pixels + remaining
      UTI pixels (excluding val and test chips).
    - Train the chosen architecture for 40 epochs.
    - Save under results/w18_rq1_cov_uti_mix/<model>_seed<S>_holdout_<chip>/

Usage:
    python -m lacewing.classification.experiments.w18_rq1_cov_uti_mix \\
        --model inception --seed 0 --holdout-chip uti_260728_EC_SD

PBS driver:
    jobs/w18_rq1_cov_uti_mix.pbs (60-task array over 5 models × 4 chips × 3 seeds)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lacewing.classification.models import build as build_model
from lacewing.classification.core.evaluate import pixel_metrics, well_metrics
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


PROJECT_ROOT = DATA_ROOT  # was: parents[3]
MERGED_CACHE = LACEWING_PKG_DIR / "quantification" / "methods" / "cache" / "merged_cov_uti_cls_cache.npz"
RESULTS_ROOT = LACEWING_PKG_DIR / "classification" / "results" / "w18_rq1_cov_uti_mix"


def _split_merged(seed: int, holdout: str) -> dict:
    d = np.load(MERGED_CACHE, allow_pickle=False)
    is_uti = d["origin"] == "uti"
    is_cov = d["origin"] == "cov"
    uti_chips = sorted(np.unique(d["chip_id"][is_uti]).tolist())
    if holdout not in uti_chips:
        raise SystemExit(f"holdout {holdout!r} not in UTI chips {uti_chips}")

    is_test = is_uti & (d["chip_id"] == holdout)
    remaining_uti_chips = [c for c in uti_chips if c != holdout]
    rng = np.random.default_rng(seed)
    val_chip = remaining_uti_chips[int(rng.integers(len(remaining_uti_chips)))]
    is_val = is_uti & (d["chip_id"] == val_chip)
    is_train = ~(is_test | is_val)  # ALL COV + remaining UTI

    return {
        "X_tr": d["X"][is_train], "y_tr": d["y"][is_train].astype(np.float32),
        "X_va": d["X"][is_val], "y_va": d["y"][is_val].astype(np.float32),
        "X_te": d["X"][is_test], "y_te": d["y"][is_test].astype(np.float32),
        "chip_te": d["chip_id"][is_test], "well_te": d["well_id"][is_test],
        "val_chip": val_chip,
    }


def _train(model: torch.nn.Module, X_tr, y_tr, X_va, y_va, args) -> list[dict]:
    device = args.device
    ds = TensorDataset(torch.from_numpy(X_tr).float().unsqueeze(1), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # BCEWithLogitsLoss with positive-class weighting (mirrors train.py behaviour)
    n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []
    for e in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit = model(xb)
            if logit.ndim == 2 and logit.shape[1] == 1:
                logit = logit[:, 0]
            loss = loss_fn(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(xb); n += len(xb)
        with torch.no_grad():
            model.eval()
            xv = torch.from_numpy(X_va).float().unsqueeze(1).to(device)
            score_v = torch.sigmoid(model(xv)).cpu().numpy()
            if score_v.ndim == 2 and score_v.shape[1] == 1:
                score_v = score_v[:, 0]
            m = pixel_metrics(y_va.astype(int), score_v)
            history.append({"epoch": e + 1, "train_loss": total / n,
                            "val_acc": m.accuracy, "val_auroc": m.auroc})
            print(f"  epoch {e+1}/{args.epochs}  loss={total/n:.4f}  val_acc={m.accuracy:.4f}", flush=True)
    return history


def _predict(model, X, device, batch=512):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().unsqueeze(1).to(device)
            logit = model(xb).cpu().numpy()
            if logit.ndim == 2 and logit.shape[1] == 1:
                logit = logit[:, 0]
            scores.append(logit)
    return 1.0 / (1.0 + np.exp(-np.concatenate(scores)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["ann", "cnn1d", "fcn", "resnet", "inception"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout-chip", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    data = _split_merged(args.seed, args.holdout_chip)
    print(f"[w18_rq1_cov_uti_mix] model={args.model} seed={args.seed} holdout={args.holdout_chip}")
    print(f"  train: COV + remaining UTI ({len(data['X_tr'])} px)")
    print(f"  val:   {data['val_chip']} ({len(data['X_va'])} px)")
    print(f"  test:  {args.holdout_chip} ({len(data['X_te'])} px)")

    model = build_model(args.model, data["X_tr"].shape[1:]).to(args.device)
    history = _train(model, data["X_tr"], data["y_tr"], data["X_va"], data["y_va"], args)

    scores = _predict(model, data["X_te"], args.device)
    pix = pixel_metrics(data["y_te"].astype(int), scores)
    well = well_metrics(data["y_te"].astype(int), scores, data["chip_te"], data["well_te"])

    out_dir = RESULTS_ROOT / f"{args.model}_seed{args.seed}_holdout_{args.holdout_chip}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict()}, out_dir / "checkpoints" / "last.pt")
    np.savez_compressed(out_dir / "predictions.npz", scores=scores.astype(np.float32),
                        chip_te=data["chip_te"], well_te=data["well_te"], y_te=data["y_te"])
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps({
        "pixel_acc": pix.accuracy, "pixel_auroc": pix.auroc, "pixel_f1": pix.f1,
        "well_acc": well["accuracy"], "well_f1": well["f1"],
    }, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "experiment": "w18_rq1_cov_uti_mix",
        "model": args.model, "seed": args.seed, "holdout_chip": args.holdout_chip,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "val_chip": data["val_chip"],
        "n_train_px": int(len(data["X_tr"])), "n_val_px": int(len(data["X_va"])),
        "n_test_px": int(len(data["X_te"])),
    }, indent=2))
    print(f"\n[ok] wrote {out_dir}")
    print(f"  pixel_acc={pix.accuracy:.4f}  well_acc={well['accuracy']:.4f}")


if __name__ == "__main__":
    main()
