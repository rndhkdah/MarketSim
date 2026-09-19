"""Central-bank skeleton: CPI histories, anchored π_e, quarterly Taylor rule (§2.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marketsim.core.clock import EventQueue
from marketsim.core.config import Config


def seed_price_history(g: float, n: int = 13) -> list[float]:
    """``n`` CPI levels on the π* path, oldest first: ``G^(k−n+1)``."""
    return [float(g ** (k - (n - 1))) for k in range(n)]


def infl_n(hist: list[float], months: int) -> float:
    """Annualised log change over ``months`` (π12 = 1×, π3 = 4×)."""
    return float((12.0 / months) * np.log(hist[-1] / hist[-1 - months]))


def policy_inflation(pi3: float, pi12: float, core3: float, core12: float, core_weight: float) -> float:
    headline = 0.5 * pi3 + 0.5 * pi12
    core = 0.5 * core3 + 0.5 * core12
    return float((1.0 - core_weight) * headline + core_weight * core)


def taylor_target(r_n: float, pi_star: float, phi_pi: float, phi_y: float, pi_pol: float, gap: float) -> float:
    """§2.5: ``r_n + π* + φ_π(π_pol − π*) + φ_y·gap``."""
    return r_n + pi_star + phi_pi * (pi_pol - pi_star) + phi_y * gap


def is_quarter_end_month(month_index: int) -> bool:
    """``month_index`` is 0-based completed months; meeting after months 3,6,9,12,…"""
    return (month_index + 1) % 3 == 0


@dataclass
class CentralBank:
    """Quarterly Taylor rule with smoothing and an ELB. Units: annual decimals."""

    r_n: float
    pi_star: float
    phi_pi: float
    phi_y: float
    rho: float
    elb: float
    core_weight: float
    infl_anchor: float
    tau_infl_m: float
    r_rule: float = 0.0
    r: float = 0.0
    pi_e: float = 0.0
    cpi_hist: list[float] = field(default_factory=list)
    core_hist: list[float] = field(default_factory=list)
    month: int = 0

    @classmethod
    def from_config(cls, cfg: Config, pi_star: float) -> CentralBank:
        assert cfg.dynamics is not None
        tay = cfg.edges.policy.taylor
        g = float(np.exp(pi_star / 12.0))
        r0 = tay.r_neutral + pi_star
        hist = seed_price_history(g)
        return cls(
            r_n=tay.r_neutral,
            pi_star=pi_star,
            phi_pi=tay.phi_inflation,
            phi_y=tay.phi_output_gap,
            rho=tay.smoothing,
            elb=cfg.dynamics.policy.elb,
            core_weight=cfg.dynamics.policy.core_weight,
            infl_anchor=cfg.dynamics.expectations.infl_anchor,
            tau_infl_m=cfg.dynamics.expectations.tau_infl_m,
            r_rule=r0,
            r=r0,
            pi_e=pi_star,
            cpi_hist=list(hist),
            core_hist=list(hist),
        )

    def pi12(self) -> float:
        return infl_n(self.cpi_hist, 12)

    def pi3(self) -> float:
        return infl_n(self.cpi_hist, 3)

    def observe_prices(self, cpi: float, core: float) -> None:
        self.cpi_hist.append(float(cpi))
        self.cpi_hist.pop(0)
        self.core_hist.append(float(core))
        self.core_hist.pop(0)
        self.pi_e += (
            (self.infl_anchor * self.pi_star + (1.0 - self.infl_anchor) * self.pi12() - self.pi_e)
            / self.tau_infl_m
        )

    def maybe_meet(
        self,
        gap: float,
        z_mon: float = 0.0,
        *,
        rate_override: float | None = None,
        pi_star: float | None = None,
        phi_pi: float | None = None,
        phi_y: float | None = None,
        smoothing: float | None = None,
    ) -> bool:
        """If this month is a quarter-end, update ``r_rule`` and ``r``. Returns True on a meeting."""
        met = is_quarter_end_month(self.month)
        if met:
            if pi_star is not None:
                self.pi_star = float(pi_star)
            if phi_pi is not None:
                self.phi_pi = float(phi_pi)
            if phi_y is not None:
                self.phi_y = float(phi_y)
            if smoothing is not None:
                self.rho = float(smoothing)
            if rate_override is not None:
                self.r_rule = float(rate_override)
                self.r = max(self.elb, float(rate_override) + z_mon)
            else:
                pi_pol = policy_inflation(
                    self.pi3(),
                    self.pi12(),
                    infl_n(self.core_hist, 3),
                    infl_n(self.core_hist, 12),
                    self.core_weight,
                )
                target = taylor_target(self.r_n, self.pi_star, self.phi_pi, self.phi_y, pi_pol, gap)
                self.r_rule = self.rho * self.r_rule + (1.0 - self.rho) * target
                self.r = max(self.elb, self.r_rule + z_mon)
        else:
            self.r = max(self.elb, self.r_rule + z_mon)
        self.month += 1
        return met

    def schedule_meetings(self, queue: EventQueue, horizon_months: int, days_per_month: int = 21) -> None:
        """Queue a ``("cb_meeting", month)`` payload on each quarter-end tick."""
        for m in range(horizon_months):
            if is_quarter_end_month(m):
                tick = (m + 1) * days_per_month - 1
                queue.schedule(tick, {"kind": "cb_meeting", "month": m}, priority=0)
