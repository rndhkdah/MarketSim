"""T7.15 — quickstart examples run against an in-process server."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples.{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quickstart_trader(config_dir: Path) -> None:
    summary = _load("quickstart_trader").run_trader(config_dir)
    assert summary["account"] == "AGENT:trader"
    assert summary["tick_after"] == summary["tick_before"] + 1
    assert summary["order_id"] >= 1
    assert "bonds" in summary["observe_keys"]
    assert len(summary["state_hash"]) == 64


def test_quickstart_operator(config_dir: Path) -> None:
    summary = _load("quickstart_operator").run_operator(config_dir)
    assert summary["firm_id"] == "acme"
    assert summary["operator"] == "operator"
    assert summary["live"] is True
    assert summary["vacancies"] == 2.0
    assert summary["tick"] == 1
    assert len(summary["state_hash"]) == 64
