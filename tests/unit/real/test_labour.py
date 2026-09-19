from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.labour import adjust_employment, step_labour, target_employment, wage_growth
from marketsim.real.steady_state import compute_real_baseline


def _fc_ustar(cfg, codes):
    fc = np.array([cfg.sectors.params(c).fixed_cost for c in codes])
    ustar = np.array([cfg.sectors.params(c).util_target for c in codes])
    return fc, ustar


def test_labour_fixed_point(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fc, ustar = _fc_ustar(cfg, real.codes)
    n, w, u = step_labour(
        real.flat(real.n0),
        1.0,
        ell=real.flat(real.ell),
        fc=fc,
        k=real.flat(real.K0),
        ustar=ustar,
        plan=real.flat(real.x0),
        z_sup=np.zeros(real.S),
        lf=real.LF,
        cfg=cfg,
        pi_e=0.0,
    )
    assert np.allclose(n, real.flat(real.n0), atol=1e-9)
    assert w == pytest.approx(1.0, abs=1e-12)
    assert u == pytest.approx(cfg.dynamics.labour.u_star, abs=1e-12)


def test_hiring_faster_than_firing(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    n0 = real.flat(real.n0)
    up = adjust_employment(n0, 1.02 * n0, 3.0, 6.0, real.LF, 0.995)
    down = adjust_employment(n0, 0.98 * n0, 3.0, 6.0, real.LF, 0.995)
    assert float((up - n0).sum()) == pytest.approx(2.0 * float((n0 - down).sum()), rel=1e-9)


def test_wage_falls_four_times_slower() -> None:
    up = wage_growth(0.0, 0.04, 0.05, 0.40, 4.0)
    down = wage_growth(0.0, 0.06, 0.05, 0.40, 4.0)
    assert up > 0
    assert down < 0
    assert abs(up / down) == pytest.approx(4.0, abs=1e-12)


def test_operating_leverage_long_run(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fc, ustar = _fc_ustar(cfg, real.codes)
    plan = 0.9 * real.flat(real.x0)
    n_star = target_employment(real.flat(real.ell), fc, real.flat(real.K0), ustar, plan, np.zeros(real.S))
    n0 = real.flat(real.n0)
    drop = (n0 - n_star) / n0
    expect = (1.0 - fc) * 0.10
    assert np.allclose(drop, expect, atol=1e-9)
