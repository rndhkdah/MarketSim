from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.errors import IOTableError
from marketsim.layer1.build_io import build_A
from marketsim.layer1.io import load_io, save_io


def test_roundtrip(io, tmp_path) -> None:
    dest = tmp_path / "io.json"
    save_io(io, dest)
    again = load_io(dest)
    assert again.codes == io.codes
    assert np.allclose(again.A, io.A)
    assert np.allclose(again.L, io.L)


def test_loaded_a_matches_build_a(io) -> None:
    assert np.allclose(io.A, build_A(), atol=1e-6)


def test_rho_ge_one_rejected(io, tmp_path) -> None:
    bad = io.to_dict()
    # force a non-productive table
    n = len(bad["codes"])
    bad["A"] = (np.ones((n, n)) * 0.2).tolist()
    bad["mu"] = [0.2 * n] * n
    dest = tmp_path / "bad.json"
    import json

    dest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(IOTableError, match="Hawkins"):
        load_io(dest)
