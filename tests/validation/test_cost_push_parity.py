from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.prices import baseline_price_state, sector_pass_through, step_prices, unit_cost
from marketsim.real.steady_state import compute_real_baseline


@pytest.mark.validation
def test_energy_cost_shock_ranks_like_g(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    pt, lag = sector_pass_through(cfg, real.codes)
    dyn = cfg.dynamics
    idx = {c: i for i, c in enumerate(real.codes)}
    z = np.zeros(real.S)
    z[idx["ENERGY"]] = np.log(1.3)
    st = baseline_price_state(real)
    for _ in range(240):
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
    dp = st.p - 1.0
    ranked = [c for c, _ in sorted(zip(real.codes, dp, strict=True), key=lambda t: -t[1])]
    non_energy = [c for c in ranked if c != "ENERGY"]
    assert set(non_energy[:3]) == {"UTILITIES", "MATERIALS", "TRANSPORT"}
    assert set(non_energy[-2:]).issubset({"SOFTWARE", "BANKS", "HEALTH", "INSURANCE"})
    g_rank = [c for c, _ in sorted(zip(real.codes, io.G[:, idx["ENERGY"]], strict=True), key=lambda t: -t[1])]
    g_non = [c for c in g_rank if c != "ENERGY"]
    assert g_non[:3] == non_energy[:3]
