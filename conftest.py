"""Root conftest — adds the project root to sys.path so that
``Analysis.*`` namespace packages are importable from any test."""
import sys
from pathlib import Path

# Ensure project root is on sys.path for namespace-package imports
# (e.g. ``from src.lacewing.quantification.eval.schema import ...``).
sys.path.insert(0, str(Path(__file__).parent))
