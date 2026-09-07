"""Per-method amplification-curve landmark extractors.

One file per landmark.  Each extractor takes ``(time_min, signal)`` and
returns the landmark time in minutes.  All extractors share the
smoothing helper and search-window defaults from ``common.py``.
"""
