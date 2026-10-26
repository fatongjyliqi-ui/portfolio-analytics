"""Macht die Module im Projektordner fuer die Tests importierbar."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # keine Fenster waehrend der Tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
