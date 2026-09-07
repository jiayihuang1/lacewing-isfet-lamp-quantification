# Analysis/quantification/eval/w13_winner_selection.py
"""W13 winner selection: rank F-A and F-B configs by lexicographic rule.

Prints top-3 for each framing + the CLI to rerun the winner at seeds 3 + 4.
"""
from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
MAIN_SB = HERE / "scoreboard.csv"
AUX_SBS = [
    HERE / "fa_extraction_scoreboard.csv",
    HERE / "seed_ensemble_scoreboard.csv",
    HERE / "tta_scoreboard.csv",
]


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _to_float(x, default=float("inf")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float:
    if not values:
        return float("inf")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def rank_configs(rows: list[dict], framing_prefix: str) -> list[tuple[str, dict]]:
    """Group scoreboard rows by (framing_config sans seed suffix), aggregate seeds.

    Returns list of (config_key, agg) sorted lexicographically by
    (median MAE, median |Δslope|, -median r², median CoV).
    """
    # Group by "everything before _seed{N}".
    groups: dict[str, list[dict]] = {}
    for r in rows:
        m = r.get("method_id", "")
        if not m.startswith(framing_prefix):
            continue
        # Strip _seed{N} suffix.
        if "_seed" in m:
            key = m.rsplit("_seed", 1)[0]
        else:
            key = m
        groups.setdefault(key, []).append(r)

    aggs = []
    for key, rs in groups.items():
        mae = [_to_float(r.get("per_well_mae_min")) for r in rs]
        dsl = [_to_float(r.get("slope_proximity_min_per_decade")) for r in rs]
        r2 = [_to_float(r.get("per_well_r2"), default=-float("inf")) for r in rs]
        cov = [_to_float(r.get("within_well_cov")) for r in rs]
        n_pass = sum(1 for r in rs if str(r.get("passes_spearman_filter", "")).lower() == "true")
        aggs.append((key, {
            "n_seeds": len(rs),
            "med_mae": _median(mae),
            "med_dslope": _median(dsl),
            "med_r2": _median(r2),
            "med_cov": _median(cov),
            "n_pass": n_pass,
        }))
    # Lexicographic sort: MAE ↑, |Δslope| ↑, r² ↓, CoV ↑.
    aggs.sort(key=lambda t: (t[1]["med_mae"], t[1]["med_dslope"],
                              -t[1]["med_r2"], t[1]["med_cov"]))
    return aggs


def print_top(rows: list[dict], framing_prefix: str, label: str, top_n: int = 5) -> None:
    aggs = rank_configs(rows, framing_prefix)
    print(f"\n=== Top {top_n} {label} configs (framing_prefix={framing_prefix}) ===")
    if not aggs:
        print("  (none)")
        return
    for i, (key, a) in enumerate(aggs[:top_n], start=1):
        marker = " *" if i == 1 else "  "
        print(f"{marker}{i}. {key}")
        print(f"     MAE={a['med_mae']:.2f}  |Ds|={a['med_dslope']:.2f}  "
              f"r2={a['med_r2']:+.2f}  CoV={a['med_cov']:.3f}  "
              f"n_seeds={a['n_seeds']}  pass={a['n_pass']}/{a['n_seeds']}")


def main() -> None:
    all_rows = _load_rows(MAIN_SB)
    for aux in AUX_SBS:
        all_rows.extend(_load_rows(aux))
    print(f"[w13_winner] loaded {len(all_rows)} total rows (main + aux)")
    print_top(all_rows, framing_prefix="p1_fa_",  label="F-A")
    print_top(all_rows, framing_prefix="p1_fb_",  label="F-B")

    print("\n=== 5-seed rerun template ===")
    print("Once you've decided on the winning configs, run e.g.:")
    print("  qsub jobs/p1_winner_rerun.pbs   # after editing the METHODS list")


if __name__ == "__main__":
    main()
