"""P3-A2p Stage-1: pretrain a cls-only head on amp-positive + NTC windows.

Trains the *same* small conv cls net used by A1-A4's non-shared archs, but
on its own (no quant head, no joint loss). The resulting cls.pt is loaded
FROZEN in A2p's Stage-2 (see joint_cls_quant.py --pretrained-cls), where
its outputs serve as a stable, calibrated gate for the quant head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lacewing.quantification.methods.joint_cls_quant import (
    JointWindowsDataset,
    _load_ntc_pixels,
    CACHE_STEM,
)
from lacewing.quantification.regression_shared.data import dataset as reg_ds


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"


def build_cls_head() -> nn.Module:
    """Mirror the cls_head used by A1-A4 (non-shared) architectures."""
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage-1 cls-only pretrain for P3-A2p (Paper 19 protocol)."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=120)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from lacewing.classification.core.seeding import seed_everything
    seed_everything(args.seed)

    # Same data pool as joint_cls_quant: amp-positive + NTC pixels combined.
    arr = reg_ds.make_split(CACHE_STEM, seed=args.seed, normalise_y=False)
    X_pos = arr.X_tr
    y_pos = arr.y_tr
    cls_pos = np.ones(len(X_pos), dtype=np.float32)

    X_ntc, cls_ntc = _load_ntc_pixels()
    y_ntc = np.full(len(X_ntc), 30.0, dtype=np.float32)  # dummy TTPs; ignored

    X_all = np.concatenate([X_pos, X_ntc], axis=0)
    y_all = np.concatenate([y_pos, y_ntc], axis=0)
    cls_all = np.concatenate([cls_pos, cls_ntc], axis=0)

    # Reuse the joint dataset for windowing + cls labels — we just ignore the
    # quant labels it also produces.
    ds = JointWindowsDataset(X_all, y_all, cls_all, args.window, args.stride, arch="a2p")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = build_cls_head().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0; n = 0
        for x, _y_q, y_c in loader:
            x = x.to(args.device); y_c = y_c.to(args.device)
            cls_logit = model(x).squeeze(-1)
            loss = bce(cls_logit, y_c)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * x.size(0); n += x.size(0)
        avg = total / max(n, 1)
        history.append({"epoch": epoch, "tr_loss": avg})
        print(f"[cls-pretrain] epoch {epoch:02d}/{args.epochs}  tr={avg:.4f}")

    out_dir = RESULTS_ROOT / "p3_cls_pretrained" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "cls.pt")
    (out_dir / "config.json").write_text(json.dumps({
        "window": args.window, "stride": args.stride,
        "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "seed": args.seed,
    }, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[ok] wrote {out_dir}/cls.pt")


if __name__ == "__main__":
    main()
