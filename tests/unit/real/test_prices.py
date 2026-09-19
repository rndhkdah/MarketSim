from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.prices import (
    PriceState,
    baseline_price_state,
    sector_pass_through,
    step_prices,
    tightness,
    unit_cost,
)
from marketsim.real.steady_state import compute_real_baseline


def _vecs(cfg, io):
    real = compute_real_baseline(io, cfg)
    pt, lag = sector_pass_through(cfg, real.codes)
    return real, pt, lag


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_price_fixed_point(cfg, io, pi_star: float) -> None:
    real, pt, lag = _vecs(cfg, io)
    dyn = cfg.dynamics
    p0 = np.ones(real.S)
    nuc = unit_cost(io.A, p0, 1.0, real.flat(real.ell), real.flat(real.m), 1.0, np.zeros(real.S))
    tight = tightness(
        real.flat(real.x0),
        real.flat(real.K0),
        np.array([cfg.sectors.params(c).util_target for c in real.codes]),
        real.flat(real.inv0),
        real.flat(real.s0),
        real.flat(real.cover),
        real.is_stock,
        dyn.prices.kappa_util,
        dyn.prices.gamma_cover,
        dyn.prices.cover_floor,
    )
    assert np.allclose(tight, 1.0, atol=1e-9)
    g = float(np.exp(pi_star / 12.0))
    st = baseline_price_state(real)
    st = step_prices(
        st,
        nuc=nuc,
        markup=real.flat(real.markup),
        tight=tight,
        z_cost=np.zeros(real.S),
        pi_e=pi_star,
        pt=pt,
        ptlag=lag,
        fast_mean_m=dyn.prices.fast_mean_m,
        step_max=dyn.prices.step_max_month,
        g=g,
    )
    assert np.allclose(st.p, g, atol=1e-12)
    assert st.p_imp == pytest.approx(g, abs=1e-12)


def test_step_bound_no_level_cap() -> None:
    target = np.log(5.0)
    st = PriceState(p=np.array([1.0]), pf=np.zeros(1), ps=np.zeros(1), p_imp=1.0)
    logs = []
    for _ in range(20):
        prev = np.log(st.p[0])
        st = step_prices(
            st,
            nuc=np.array([1.0]),
            markup=np.array([1.0]),
            tight=np.array([1.0]),
            z_cost=np.array([target]),
            pi_e=0.0,
            pt=np.array([1.0]),
            ptlag=np.array([0.5]),
            fast_mean_m=1.0,
            step_max=0.15,
            g=1.0,
        )
        dlp = np.log(st.p[0]) - prev
        logs.append(dlp)
    assert logs[0] == pytest.approx(0.15, abs=1e-12)
    assert all(abs(d - 0.15) < 1e-12 for d in logs[:8])
    assert st.p[0] > 4.0
    assert st.p[0] < 5.0 or abs(np.log(st.p[0]) - target) < 0.15


def test_utilities_half_life_exceeds_transport(cfg, io) -> None:
    real, pt, lag = _vecs(cfg, io)
    dyn = cfg.dynamics
    z = np.full(real.S, np.log(1.2))
    st = baseline_price_state(real)
    path = []
    for _ in range(80):
        nuc = unit_cost(io.A, st.p, 1.0, real.flat(real.ell), real.flat(real.m), 1.0, np.zeros(real.S))
        st = step_prices(
            st,
            nuc=nuc,
            markup=real.flat(real.markup),
            tight=np.ones(real.S),
            z_cost=z,
            pi_e=0.0,
            pt=pt,
            ptlag=lag,
            fast_mean_m=dyn.prices.fast_mean_m,
            step_max=dyn.prices.step_max_month,
            g=1.0,
        )
        path.append(st.p.copy())
    final = path[-1]
    idx = {c: i for i, c in enumerate(real.codes)}

    def half_life(code: str) -> int:
        i = idx[code]
        target = np.log(final[i])
        for t, p in enumerate(path, start=1):
            if np.log(p[i]) >= 0.5 * target:
                return t
        return 80

    assert half_life("UTILITIES") > half_life("TRANSPORT")
