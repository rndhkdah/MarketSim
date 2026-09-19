from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.erlang import ErlangChain, ErlangSmoother


def test_impulse_mass_and_mean() -> None:
    mean_m = 6.0
    k = 3
    chain = ErlangChain(k, mean_m, (1,))
    out = []
    out.append(float(chain.push(np.array([1.0]))))
    for _ in range(400):
        out.append(float(chain.push(np.array([0.0]))))
    y = np.array(out)
    assert abs(y.sum() - 1.0) < 1e-6
    t = np.arange(len(y))
    assert abs(float((t * y).sum()) - mean_m) < 1e-6


def test_k1_is_geometric() -> None:
    chain = ErlangChain(1, 4.0, ())
    a = 1.0 / (4.0 + 1.0)
    assert chain.push(1.0) == pytest.approx(a)
    assert chain.push(0.0) == pytest.approx(a * (1 - a))


def test_seed_then_constant_inflow_is_fixed_point() -> None:
    chain = ErlangChain(3, 5.0, (2, 3))
    flow = np.linspace(1.0, 2.0, 6).reshape(2, 3)
    chain.seed(flow)
    assert np.allclose(chain.push(flow), flow, atol=1e-12)


def test_mass_conserved() -> None:
    chain = ErlangChain(2, 3.0, (4,))
    inflow = np.array([0.4, 0.0, 1.2, 0.3, 0.0])
    out = 0.0
    for x in inflow:
        out += float(chain.push(np.array([x])).sum())
    assert out + float(chain.content().sum()) == pytest.approx(float(inflow.sum()))


def test_mixed_means_and_shapes() -> None:
    means = np.array([[0.0, 4.0], [2.0, 8.0]])
    chain = ErlangChain(2, means, (2, 2))
    x = np.ones((2, 2))
    chain.seed(x)
    assert np.allclose(chain.push(x), x)


def test_zero_mean_passthrough() -> None:
    chain = ErlangChain(3, 0.0, (3,))
    x = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(chain.push(x), x)
    sm = ErlangSmoother(3, 0.0, np.zeros(2))
    u = np.array([4.0, 5.0])
    assert np.array_equal(sm.push(u), u)


def test_smoother_moves_toward_input() -> None:
    sm = ErlangSmoother(2, 4.0, 0.0)
    y = sm.push(1.0)
    assert 0.0 < float(y) < 1.0
