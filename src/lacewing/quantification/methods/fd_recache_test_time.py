"""Re-run existing F-D checkpoints on the SD test set to cache per-timestep
predictions (argmax + softmax).  Feeds the extraction-rule sweep.

Idempotent: skips seeds where test_time_predictions.npz already exists.
Runs on CPU by default; use --device cuda for GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lacewing.quantification.methods import unet_segmentation as u
from lacewing.quantification.regression_shared.data import dataset as reg_ds


RESULTS_ROOT = Path(__file__).resolve().parents[3] / "Analysis/quantification/methods/results"

# (dir_name, ckpt_tag, needs_reg_ds)
TARGETS = [
    ("p1_fd_manual4_spatA3",   "last.pt"),
    ("p1_fd_manual4_spatA3_bb_gru", "last.pt"),
    ("p1_fd_manual4_spatA3_bb_cnn_gru_par", "last.pt"),
    ("p1_fd_unet_seg_ow30_spatA3", "last.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--force", action="store_true", help="Overwrite existing caches")
    args = p.parse_args()

    for dir_name, ckpt_tag in TARGETS:
        root = RESULTS_ROOT / dir_name
        if not root.exists():
            print(f"[skip] {root}: not found")
            continue
        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir():
                continue
            ckpt_path = seed_dir / "checkpoints" / ckpt_tag
            if not ckpt_path.exists():
                print(f"[skip] {seed_dir.name}: no {ckpt_tag}")
                continue
            out_path = seed_dir / "test_time_predictions.npz"
            if out_path.exists() and not args.force:
                print(f"[skip] {seed_dir.name}: cache exists ({out_path.stat().st_size // 1024} KB)")
                continue

            # Load seed number from dir name.
            seed_num = int(seed_dir.name.replace("seed", ""))
            arr = reg_ds.make_split(
                "regress_all_filt_abcd_ntcRaw_madk1p5_spatA3", seed=seed_num
            )

            model = u.build_model()
            ck = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            model.to(args.device)

            print(f"[run] {dir_name}/{seed_dir.name}: caching {arr.X_te.shape[0]} test pixels")
            u._save_test_time_argmax(model, arr.X_te, args.device, out_path)
            print(f"  wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
