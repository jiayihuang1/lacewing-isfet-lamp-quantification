"""Canonical predictions.npz and labels.npz schema for the RQ2 scoreboard.

Every quantification method emits these two files per run. The evaluation
harness (resolution_metrics.py + scoreboard.py) reads them and produces a
scoreboard row.

Shapes are per-pixel arrays, length n_pixels, in a fixed order determined
by the method (typically the order of the test-split rows in the cache).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PredictionArrays:
    ttp_pred_min: np.ndarray  # (n_pixels,) float32, predicted TTP in minutes
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. per-timestep density


@dataclass
class LabelArrays:
    ttp_true_min: np.ndarray            # (n_pixels,) float32, qLAMP TTP in minutes
    chip_id: np.ndarray                 # (n_pixels,) U-string
    well_id: np.ndarray                 # (n_pixels,) int32
    log10_concentration: np.ndarray     # (n_pixels,) float32, log10(copies/uL)
    split: np.ndarray                   # (n_pixels,) U-string, {"train","val","test"}


def write_predictions(
    path: Path,
    ttp_pred_min: np.ndarray,
    ttp_pred_extras: dict[str, np.ndarray] | None = None,
) -> None:
    ttp = np.ascontiguousarray(ttp_pred_min, dtype=np.float32)
    to_save: dict[str, np.ndarray] = {"ttp_pred_min": ttp}
    if ttp_pred_extras:
        for k, v in ttp_pred_extras.items():
            to_save[f"extra__{k}"] = np.ascontiguousarray(v)
    np.savez_compressed(path, **to_save)


def read_predictions(path: Path) -> PredictionArrays:
    d = np.load(path, allow_pickle=False)
    extras = {k[len("extra__"):]: d[k] for k in d.files if k.startswith("extra__")}
    return PredictionArrays(ttp_pred_min=d["ttp_pred_min"], extras=extras)


def write_labels(
    path: Path,
    ttp_true_min: np.ndarray,
    chip_id: np.ndarray,
    well_id: np.ndarray,
    log10_concentration: np.ndarray,
    split: np.ndarray,
) -> None:
    # Fixed U-widths: chip_id U160 accommodates the concatenated
    # dose-response+SD chip identifiers observed at ~76 chars (see
    # "Chip ID truncation risk" in docs/superpowers/plans/known-risks.md).
    # Assert on write to fail loudly rather than silently truncate.
    chip_arr = np.asarray(chip_id)
    max_chip_len = max((len(str(c)) for c in chip_arr), default=0)
    if max_chip_len > 160:
        raise ValueError(
            f"chip_id length {max_chip_len} exceeds U160 cap — "
            "widen dtype in schema.py or shorten chip identifiers"
        )
    split_arr = np.asarray(split)
    max_split_len = max((len(str(s)) for s in split_arr), default=0)
    if max_split_len > 16:
        raise ValueError(
            f"split label length {max_split_len} exceeds U16 cap"
        )
    np.savez_compressed(
        path,
        ttp_true_min=np.ascontiguousarray(ttp_true_min, dtype=np.float32),
        chip_id=chip_arr.astype("U160"),
        well_id=np.ascontiguousarray(well_id, dtype=np.int32),
        log10_concentration=np.ascontiguousarray(log10_concentration, dtype=np.float32),
        split=split_arr.astype("U16"),
    )


def read_labels(path: Path) -> LabelArrays:
    d = np.load(path, allow_pickle=False)
    return LabelArrays(
        ttp_true_min=d["ttp_true_min"],
        chip_id=d["chip_id"],
        well_id=d["well_id"],
        log10_concentration=d["log10_concentration"],
        split=d["split"],
    )


def validate_alignment(preds: PredictionArrays, labels: LabelArrays) -> None:
    n_p = len(preds.ttp_pred_min)
    n_l = len(labels.ttp_true_min)
    if n_p != n_l:
        raise ValueError(
            f"pixel count mismatch: predictions has {n_p}, labels has {n_l}"
        )
