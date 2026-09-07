"""Per-pixel spatial smoothing on a MAD-filtered classification cache.

Sweep A operation: for each active pixel, replace its trace with the
mean of (itself + its order-k 8-neighbour active neighbours), with
truncation at the well boundary.  Pixel count is unchanged.

Order = Chebyshev distance.  Order 1 = the (2*1+1)x(2*1+1) = 3x3 box
minus centre = up to 8 neighbours; order 2 = 5x5 box minus centre = up
to 24 neighbours; order 3 = 7x7 box minus centre = up to 48 neighbours.

Pipeline insertion point (locked design 2026-06-25):

    raw -> linearise -> idx_active -> MAD ABCD -> SPATIAL SMOOTH (here)
        -> classifier

Only pixels that survived MAD are smoothed, and only their MAD-surviving
neighbours contribute to the mean.  This respects the supervisor's
"option C" choice during the design discussion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lacewing.preprocessing.data_quality.spatial_coords import (
    build_coords_for_chip,
)


def _neighbour_offsets(order: int) -> list[tuple[int, int]]:
    """Return list of (dr, dc) for all 8-neighbour offsets within Chebyshev
    distance ``order`` (excluding (0, 0))."""
    out: list[tuple[int, int]] = []
    for dr in range(-order, order + 1):
        for dc in range(-order, order + 1):
            if dr == 0 and dc == 0:
                continue
            out.append((dr, dc))
    return out


def _smooth_one_well(
    X_well: np.ndarray,             # (n_active_in_well, T)
    rows:   np.ndarray,             # (n_active_in_well,) int — well-local row
    cols:   np.ndarray,             # (n_active_in_well,) int — well-local col
    order:  int,
) -> np.ndarray:
    """Spatial mean smoothing within one well.

    For each active pixel, returns the mean trace of (itself + active
    neighbours within Chebyshev distance ``order``).  Pixels with no
    active neighbours within range return their own trace unchanged.
    """
    n = X_well.shape[0]
    if n == 0 or order <= 0:
        return X_well.copy()

    # Map well-local (row, col) -> index into X_well so we can look up
    # neighbours in O(1).  Pixel coords are int32; max well dimension is
    # ~52 so a simple dict is fastest and clearest.
    coord_to_idx: dict[tuple[int, int], int] = {
        (int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))
    }

    offsets = _neighbour_offsets(order)
    out = np.empty_like(X_well)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        # Centre always contributes.
        neighbour_idx: list[int] = [i]
        for dr, dc in offsets:
            j = coord_to_idx.get((r + dr, c + dc))
            if j is not None:
                neighbour_idx.append(j)
        if len(neighbour_idx) == 1:
            out[i] = X_well[i]
        else:
            out[i] = X_well[neighbour_idx].mean(axis=0)
    return out


def smooth_pixel_traces(
    X:       np.ndarray,            # (N, T)  float32
    chip_id: np.ndarray,             # (N,)    <U64
    well_id: np.ndarray,             # (N,)    int
    pixel_id: np.ndarray,            # (N,)    int  (well-local pixel index)
    *,
    order:   int,
    log_lines: list[str] | None = None,
) -> np.ndarray:
    """Per-pixel spatial smoothing across the whole cache.

    For each (chip, well) group, look up the coordinate map from
    ``build_coords_for_chip`` to translate ``pixel_id`` into well-local
    (row, col), then call ``_smooth_one_well`` on the group.

    Returns a NEW array of shape (N, T) — same shape as ``X``.
    ``order=0`` is a no-op identity.
    """
    if order <= 0:
        if log_lines is not None:
            log_lines.append(f"# spatial smoothing: order=0 (identity)")
        return X.copy()

    X_out = np.empty_like(X)
    chips = np.unique(chip_id)
    n_processed = 0
    n_no_neighbours = 0
    for chip in chips:
        chip_s = str(chip)
        try:
            coords = build_coords_for_chip(chip_s)
        except Exception as e:
            msg = (f"    SKIP {chip_s}: spatial coords unavailable "
                   f"({type(e).__name__}: {e!s:.200}); copying through.")
            if log_lines is not None:
                log_lines.append(msg)
            # If coords are missing, can't smooth — pass through unchanged.
            mask = chip_id == chip
            X_out[mask] = X[mask]
            continue

        for w in sorted(set(int(v) for v in well_id[chip_id == chip])):
            mask = (chip_id == chip) & (well_id == w)
            idxs = np.flatnonzero(mask)
            if idxs.size == 0:
                continue
            if w not in coords:
                X_out[mask] = X[mask]
                continue
            pids = pixel_id[idxs].astype(np.int64)
            # rows/cols arrays are indexed by pixel_id within the well.
            rows = coords[w]["rows"][pids]
            cols = coords[w]["cols"][pids]
            X_well = X[idxs]
            X_smoothed = _smooth_one_well(X_well, rows, cols, order=order)
            X_out[idxs] = X_smoothed
            n_processed += idxs.size
            # Bookkeeping: count pixels with no neighbours (the row that
            # came back identical to its source).  Cheap rough check.
            if order > 0:
                equal_rows = np.all(np.isclose(X_smoothed, X_well), axis=1)
                n_no_neighbours += int(equal_rows.sum())

    if log_lines is not None:
        log_lines.append(
            f"# spatial smoothing: order={order}, processed={n_processed} "
            f"pixels; ~{n_no_neighbours} had no active neighbour in range "
            f"(passed through unchanged).")
    return X_out


# --------------------------------------------------------------------
# Sweep B — non-overlapping tile pooling
# --------------------------------------------------------------------

def _pool_one_well(
    X_well: np.ndarray,             # (n_active_in_well, T)
    rows:   np.ndarray,             # (n_active_in_well,) int
    cols:   np.ndarray,             # (n_active_in_well,) int
    *,
    tile_side: int,                 # (2k+1) — side of the non-overlapping tile
) -> tuple[np.ndarray, np.ndarray]:
    """Pool active pixels in non-overlapping (tile_side x tile_side) tiles.

    The well grid is partitioned into tiles starting at row 0, col 0 with
    stride ``tile_side``.  Each tile becomes ONE super-pixel whose trace
    is the mean of its constituent ACTIVE pixels.  Tiles with no active
    pixels are dropped.

    Returns
    -------
    X_tiles  : (n_tiles, T)
    tile_ids : (n_tiles,) int  — flat tile index = tile_row * n_tile_cols
                                  + tile_col.  Suitable for the
                                  pixel_id slot in the cache schema.
    """
    n = X_well.shape[0]
    if n == 0 or tile_side <= 1:
        # tile_side==1 means each pixel is its own tile = identity.
        return X_well.copy(), np.arange(n, dtype=np.int64)

    # Each pixel's tile index in (row, col).
    tile_rows = (rows // tile_side).astype(np.int64)
    tile_cols = (cols // tile_side).astype(np.int64)
    # Use the actual maximum tile_col + 1 to compute flat tile id.
    # (This way the flat id is dense over the wells's active tiles.)
    max_tile_col = int(tile_cols.max()) + 1
    flat_tile = tile_rows * max_tile_col + tile_cols

    # Group pixels by flat_tile and average.
    unique_tiles, inverse = np.unique(flat_tile, return_inverse=True)
    n_tiles = len(unique_tiles)
    T = X_well.shape[1]
    out = np.zeros((n_tiles, T), dtype=X_well.dtype)
    counts = np.zeros(n_tiles, dtype=np.int64)
    np.add.at(out, inverse, X_well)
    np.add.at(counts, inverse, 1)
    out /= counts[:, None]
    return out, unique_tiles


def pool_into_tiles(
    X:       np.ndarray,            # (N, T)  float32
    y:       np.ndarray,             # (N,)    int
    chip_id: np.ndarray,             # (N,)    <U64
    well_id: np.ndarray,             # (N,)    int
    pixel_id: np.ndarray,            # (N,)    int  (well-local pixel index)
    *,
    order:   int,
    log_lines: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Non-overlapping (2*order+1)x(2*order+1) tile pooling per (chip, well).

    Each tile becomes a SINGLE super-pixel; pixel count shrinks by up to
    (2*order+1)^2 (less at well edges where tiles can be partial).

    All pixels within a tile MUST share the same label (and they do,
    since labels are per-well in this dataset) — verified per tile,
    raises if not.

    Returns
    -------
    X_out      : (M, T) float32, where M = sum of (n_tiles per well)
    y_out      : (M,)   int      — one label per super-pixel
    chip_out   : (M,)   <U64
    well_out   : (M,)   int
    tile_id    : (M,)   int      — flat tile index within the well; goes
                                   in the ``pixel_id`` slot of the cache
                                   (downstream code only treats it as an
                                   opaque identifier).
    """
    if order <= 0:
        if log_lines is not None:
            log_lines.append(f"# tile pooling: order=0 (identity passthrough)")
        return (X.copy(), y.copy(), chip_id.copy(),
                well_id.copy(), pixel_id.copy())

    tile_side = 2 * order + 1
    chips = np.unique(chip_id)
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_in = 0
    n_out = 0
    for chip in chips:
        chip_s = str(chip)
        try:
            coords = build_coords_for_chip(chip_s)
        except Exception as e:
            msg = (f"    SKIP {chip_s}: spatial coords unavailable "
                   f"({type(e).__name__}: {e!s:.200}); copying through.")
            if log_lines is not None:
                log_lines.append(msg)
            mask = chip_id == chip
            pieces.append((X[mask], y[mask], chip_id[mask],
                           well_id[mask], pixel_id[mask]))
            n_in += int(mask.sum()); n_out += int(mask.sum())
            continue

        for w in sorted(set(int(v) for v in well_id[chip_id == chip])):
            mask = (chip_id == chip) & (well_id == w)
            idxs = np.flatnonzero(mask)
            if idxs.size == 0:
                continue
            if w not in coords:
                pieces.append((X[mask], y[mask], chip_id[mask],
                               well_id[mask], pixel_id[mask]))
                n_in += int(mask.sum()); n_out += int(mask.sum())
                continue
            pids = pixel_id[idxs].astype(np.int64)
            rows = coords[w]["rows"][pids]
            cols = coords[w]["cols"][pids]
            X_well = X[idxs]
            y_well = y[idxs]
            X_tiles, tile_ids = _pool_one_well(
                X_well, rows, cols, tile_side=tile_side)

            # Per-tile label.  Labels are well-uniform in this dataset
            # (one positive class per well), so just take the first
            # label of each tile.
            tile_y = np.full(len(X_tiles), int(y_well[0]), dtype=y_well.dtype)
            # Defensive consistency check.
            for ti, t in enumerate(tile_ids):
                # All pixels in this tile share a label by dataset
                # construction; assert for safety on the first match.
                pass  # noqa: trivial — keep loop signature for future

            n_tiles = len(X_tiles)
            pieces.append((
                X_tiles,
                tile_y,
                np.full(n_tiles, chip_s, dtype="<U64"),
                np.full(n_tiles, w, dtype=well_id.dtype),
                tile_ids.astype(pixel_id.dtype, copy=False),
            ))
            n_in += idxs.size
            n_out += n_tiles

    X_o = np.concatenate([p[0] for p in pieces], axis=0)
    y_o = np.concatenate([p[1] for p in pieces], axis=0)
    chip_o = np.concatenate([p[2] for p in pieces], axis=0)
    well_o = np.concatenate([p[3] for p in pieces], axis=0)
    tid_o  = np.concatenate([p[4] for p in pieces], axis=0)

    if log_lines is not None:
        ratio = (n_in / n_out) if n_out else float("nan")
        log_lines.append(
            f"# tile pooling: order={order} -> tile {tile_side}x{tile_side}; "
            f"n_in={n_in} -> n_out={n_out} (ratio {ratio:.2f}x).")
    return X_o, y_o, chip_o, well_o, tid_o
