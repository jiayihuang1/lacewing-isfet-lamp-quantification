"""Analysis pipeline for the multi-Vref chip data (Data/Multi/).

This package targets the ``Matthew_Multi`` branch of the titan repo,
which is a SEPARATE clone at ``Code/titan-signal-processing-multi/``.
DO NOT import from ``lacewing.quantification`` or ``lacewing.classification``
in the same Python process — those modules wire in the ``main`` branch of
titan, and ``import titan`` is cached per-process.

If you need to compare old- and new-titan results, run the two pipelines
in separate Python processes (e.g. two notebook kernels).
"""
