from __future__ import annotations

import numpy as np

from marketsim.layer1.build_io import build_A, mix_matrix, table_dict


def test_nnz_219_and_mix_columns() -> None:
    A = build_A()
    assert A.shape == (18, 18)
    assert int(np.count_nonzero(A > 0)) == 219
    mix = mix_matrix()
    assert np.allclose(mix.sum(0), 1.0, atol=1e-9)


def test_committed_json_matches_builder(io) -> None:
    built = np.asarray(table_dict()["A"], dtype=float)
    assert np.allclose(io.A, built, atol=1e-9)
