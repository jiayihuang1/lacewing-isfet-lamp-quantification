"""Backward-compatibility re-exports.

Files moved into core/ subpackage during Week 7 reorganisation.  This
module preserves the older external import paths
(``from lacewing.classification import paths``) so existing scripts in
Meetings/, notebooks/, etc. continue to work without modification.
"""
from .core import paths  # noqa: F401
