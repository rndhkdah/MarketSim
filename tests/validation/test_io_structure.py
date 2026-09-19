from __future__ import annotations

import pytest

from marketsim.core.config import load_config
from marketsim.layer1.checks import assert_structure
from marketsim.layer1.io import load_io, resolve_io_path


@pytest.mark.validation
@pytest.mark.parametrize("source", ["seed"])
def test_structure_via_load_io(config_dir, source, monkeypatch) -> None:
    cfg = load_config(config_dir, overrides={"world.io_source": source})
    io = load_io(resolve_io_path(cfg))

    def boom(*_a, **_k):
        raise AssertionError("checks must not call build_A")

    monkeypatch.setattr("marketsim.layer1.build_io.build_A", boom)
    assert_structure(io, cfg)
