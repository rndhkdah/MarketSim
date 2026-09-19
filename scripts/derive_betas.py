#!/usr/bin/env python3
"""CLI: derive betas (pure functions) and write betas.md / betas.npz."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketsim.layer1.betas import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
