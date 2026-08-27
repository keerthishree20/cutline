#!/usr/bin/env python
"""CSV -> parquet, once. Run after download_data.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import data  # noqa: E402

if __name__ == "__main__":
    data.build_parquet(force="--force" in sys.argv)
