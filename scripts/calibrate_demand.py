#!/usr/bin/env python3
"""CLI: RAS + least-squares demand calibrator → wants.yaml / buy_packages.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketsim.demand.calibrate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
