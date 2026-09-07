"""Aggregate W15-W16 scoreboard, pick best config per workstream, print
report-ready table.

Winner rule per workstream: lowest median MAE. Break ties by (lowest
|Δslope|, then highest r²).
"""
from __future__ import annotations

from pathlib import Path
import re
import json

import pandas as pd
from lacewing._paths import LACEWING_PKG_DIR, DATA_ROOT


SB = LACEWING_PKG_DIR / "quantification" / "eval" / "scoreboard.csv"


def _strip_seed(mid: str) -> str:
    return re.sub(r"_seed\d+.*$", "", mid)


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["config"] = df.method_id.apply(_strip_seed)
    agg = df.groupby("config").agg(
        n_seeds=("method_id", "count"),
        med_mae=("per_well_mae_min", "median"),
        med_r2=("per_well_r2", "median"),
        med_dslope=("slope_proximity_min_per_decade", "median"),
        med_cov=("within_well_cov", "median"),
        n_pass=("passes_spearman_filter", "sum"),
    ).reset_index().sort_values(["med_mae", "med_dslope", "med_r2"],
                                 ascending=[True, True, False])
    return agg


def main() -> None:
    df = pd.read_csv(SB)

    entries = {}

    # 1-2. rule_ttp, rule_cy0
    for rule in ("rule_ttp", "rule_cy0"):
        r = df[df.method_id == rule]
        if len(r):
            rr = r.iloc[0]
            entries[rule] = {
                "label": rule,
                "med_mae": float(rr.per_well_mae_min),
                "n_seeds": 1,
                "n_pass": int(bool(rr.passes_spearman_filter)),
            }

    # 3. F-A U-Net curriculum σ (existing)
    fa_agg = _agg(df[df.method_id.str.startswith("p2_fa_curriculum_bb_unet_wmse_seed")])
    if len(fa_agg):
        r = fa_agg.iloc[0]
        entries["fa_unet_curriculum"] = {
            "label": "F-A U-Net curriculum σ",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # 4. F-B U-Net (existing)
    fb_agg = _agg(df[df.method_id.str.startswith("p1_fb_slide_cls_w120_stride10_spatA3_bb_unet_kthr0.7_seed")])
    if len(fb_agg):
        r = fb_agg.iloc[0]
        entries["fb_unet"] = {
            "label": "F-B U-Net",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # 5. F-D best: consider all extraction-rule × backbone combos on manual labels + qLAMP-rule
    fd_agg = _agg(df[df.method_id.str.startswith("p1_fd_")])
    if len(fd_agg):
        r = fd_agg.iloc[0]
        entries["fd_best"] = {
            "label": f"F-D best ({r.config})",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # 6. P3 best variant
    p3_agg = _agg(df[df.method_id.str.startswith("p3_")])
    if len(p3_agg):
        r = p3_agg.iloc[0]
        entries["p3_best"] = {
            "label": f"P3 best ({r.config})",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # 7a. P4 best on F-A framing
    p4_fa_agg = _agg(df[df.method_id.str.startswith("p4_") & df.method_id.str.contains("_fa_seed")])
    if len(p4_fa_agg):
        r = p4_fa_agg.iloc[0]
        entries["p4_best_fa"] = {
            "label": f"P4 best F-A ({r.config})",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # 7b. P4 best on F-B framing
    p4_fb_agg = _agg(df[df.method_id.str.startswith("p4_") & df.method_id.str.contains("_fb_seed")])
    if len(p4_fb_agg):
        r = p4_fb_agg.iloc[0]
        entries["p4_best_fb"] = {
            "label": f"P4 best F-B ({r.config})",
            "med_mae": float(r.med_mae),
            "n_seeds": int(r.n_seeds),
            "n_pass": int(r.n_pass),
        }

    # Print table
    print("W15-W16 report scoreboard:")
    print(f'{"key":25s}  {"label":50s}  {"MAE":>6}  {"seeds":>5}  {"pass":>5}')
    for k, e in entries.items():
        print(f'{k:25s}  {e["label"]:50s}  {e["med_mae"]:>6.2f}  '
              f'{e["n_seeds"]:>5d}  {e["n_pass"]:>5d}')

    # Save to json for figure builder
    out = Path("Meetings/Week_16_20260803/report_scoreboard_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
