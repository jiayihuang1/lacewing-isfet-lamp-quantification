"""Spatial-filter preprocessing layer for the per-pixel classification cache.

Two complementary operations on the MAD-filtered cache, both applied as
a post-MAD layer (before the classifier sees the data):

* ``smooth_pixel_traces``  — per-pixel spatial smoothing.  Each surviving
  active pixel's trace is replaced by the mean of itself + its
  Chebyshev-distance-k 8-neighbour active neighbours.  Pixel count is
  unchanged; per-pixel granularity is preserved.  See ``Sweep A``.

* ``pool_into_tiles``  — non-overlapping (2k+1)x(2k+1) tile pooling.
  Pixels are grouped into spatial tiles; each tile becomes a single
  "super-pixel" with a trace = mean of its constituents.  Pixel count
  shrinks by ~(2k+1)**2.  See ``Sweep B``.

Both operations:
  * use the well-local row/col coordinates from
    ``lacewing.preprocessing.data_quality.spatial_coords.build_coords_for_chip``
  * truncate at the well boundary (edge pixels average over fewer
    neighbours; no padding, no reflection)
  * only consider ACTIVE pixels (those that survived MAD ABCD)
"""
