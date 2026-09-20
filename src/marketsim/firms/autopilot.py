"""Firm-scale Phase-2 rules (T5.07). Calls the real-layer functions; no copies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.real.capex import cost_of_capital_gap, start_rate
from marketsim.real.labour import step_labour
from marketsim.real.orders import order_matrix
from marketsim.real.prices import PriceState, step_prices, tightness, unit_cost
from marketsim.real.production import expected_sales, plan_output


def _v(*xs: float) -> np.ndarray:
    return np.asarray(xs, dtype=float)


@dataclass
class Autopilot:
    """One-cell wrappers around R4, R5, R7, R8, §2.6, R9."""

    cfg: Config

    def plan(
        self,
        se: float,
        inv: float,
        cover: float,
        leak: float,
        tau_inv: float,
        backlog: float,
        tau_b: float,
        is_stock: bool,
        k_eff: float,
    ) -> float:
        """R4 plan (units / month)."""
        return float(
            plan_output(
                _v(se),
                _v(inv),
                _v(cover),
                _v(leak),
                _v(tau_inv),
                _v(backlog),
                tau_b,
                np.array([is_stock]),
                _v(k_eff),
            )[0]
        )

    def expect_sales(self, se: float, sales: float) -> float:
        assert self.cfg.dynamics is not None
        return float(expected_sales(_v(se), _v(sales), self.cfg.dynamics.expectations.tau_sales_m)[0])

    def orders(self, a_col: np.ndarray, plan: float, s_in: np.ndarray, stor_in: np.ndarray, n_in: float, tau_in: float) -> np.ndarray:
        """R5 input orders for one buyer (cr / month)."""
        return order_matrix(a_col[:, None] if a_col.ndim == 1 else a_col, _v(plan), s_in, stor_in, n_in, tau_in)

    def price(
        self,
        state: PriceState,
        *,
        a: np.ndarray,
        p: np.ndarray,
        w: float,
        ell: float,
        m: float,
        p_imp: float,
        z_sup: float,
        markup: float,
        x: float,
        k: float,
        ustar: float,
        inv: float,
        se: float,
        cover: float,
        is_stock: bool,
        z_cost: float,
        pi_e: float,
        pt: float,
        ptlag: float,
        g: float,
    ) -> PriceState:
        """R7 posted price for one cell."""
        assert self.cfg.dynamics is not None
        dyn = self.cfg.dynamics
        nuc = unit_cost(a, p, w, _v(ell), _v(m), p_imp, _v(z_sup))
        tight = tightness(
            _v(x), _v(k), _v(ustar), _v(inv), _v(se), _v(cover), np.array([is_stock]),
            dyn.prices.kappa_util, dyn.prices.gamma_cover, dyn.prices.cover_floor,
        )
        return step_prices(
            state,
            nuc=nuc if nuc.shape == state.p.shape else _v(float(np.asarray(nuc).reshape(-1)[0])),
            markup=_v(markup),
            tight=tight if tight.shape == state.p.shape else _v(float(np.asarray(tight).reshape(-1)[0])),
            z_cost=_v(z_cost),
            pi_e=pi_e,
            pt=_v(pt),
            ptlag=_v(ptlag),
            fast_mean_m=dyn.prices.fast_mean_m,
            step_max=dyn.prices.step_max_month,
            g=g,
        )

    def labour(self, n: float, w: float, *, ell: float, fc: float, k: float, ustar: float, plan: float, z_sup: float, lf: float, pi_e: float) -> tuple[float, float, float]:
        """R8 hire/fire + wage. Returns ``(n, w, U)``."""
        n_new, w_new, u = step_labour(
            _v(n),
            w,
            ell=_v(ell),
            fc=_v(fc),
            k=_v(k),
            ustar=_v(ustar),
            plan=_v(plan),
            z_sup=_v(z_sup),
            lf=lf,
            cfg=self.cfg,
            pi_e=pi_e,
        )
        return float(n_new[0]), float(w_new), float(u)

    def starts(
        self,
        *,
        k: float,
        u: float,
        ustar: float,
        g_e: float,
        r: float,
        pi_e: float,
        r_n: float,
        nd: float,
    ) -> float:
        """§2.6 capacity starts (units / month)."""
        assert self.cfg.dynamics is not None
        dyn = self.cfg.dynamics
        cap = self.cfg.edges.capex
        cc = cost_of_capital_gap(r, pi_e, r_n, 0.0, 0.0, _v(nd), dyn.credit.pricing)
        sl = np.zeros(1)
        rate = start_rate(
            delta=dyn.capex.delta_annual,
            unit_scale=cap.unit_scale,
            phi=cap.coefficients.phi_accelerator,
            psi=cap.coefficients.psi_utilisation,
            chi=cap.coefficients.chi_cost_of_capital,
            q_tobin_coef=cap.coefficients.q_tobin,
            q_scale=cap.q_scale,
            q_clip=dyn.capex.q_clip,
            g_e=_v(g_e),
            anchor_growth=dyn.expectations.anchor_growth,
            u=_v(u),
            ustar=_v(ustar),
            sl=sl,
            cc_gap=cc,
            ln_q=np.zeros(1),
            cap_mult=dyn.capex.start_rate_cap_mult,
        )
        return float(k * rate[0] / 12.0)

    def payout(self, profit: float, debt: float, debt_target: float, payout_frac: float) -> float:
        """R9 slow leverage-norm payout (cr / month)."""
        assert self.cfg.dynamics is not None
        gap = debt - debt_target
        return float(max(payout_frac * max(profit, 0.0) - self.cfg.dynamics.firms.kappa_leverage / 12.0 * gap, 0.0))
