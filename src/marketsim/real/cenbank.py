"""Central-bank skeleton: CPI histories, anchored π_e, quarterly Taylor rule (§2.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from marketsim.core.clock import EventQueue
from marketsim.core.config import Config
from marketsim.real.policy.monetary_rule import MonetaryRule, elb_toolkit


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
    framework: MonetaryRule | None = None
    u_star: float = 0.05
    last_payload: dict | None = None

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

    def pi_pol(self) -> float:
        w = self.framework.core_weight if self.framework is not None else self.core_weight
        return policy_inflation(
            self.pi3(),
            self.pi12(),
            infl_n(self.core_hist, 3),
            infl_n(self.core_hist, 12),
            w,
        )

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
        u: float | None = None,
        g_obs: float = 0.0,
        makeup: float | None = None,
        fci: float | None = None,
        risk_off: float | None = None,
        spread: float = 0.0,
        s0: float = 0.0,
        gate: float = 1.0,
        guidance: float | None = None,
        credit_gap: float = 0.0,
    ) -> bool:
        """If this month is a meeting, update ``r_rule`` and ``r``. Returns True on a meeting."""
        u_now = self.u_star if u is None else float(u)
        if self.framework is not None:
            info = self.framework.information_set(
                cpi_hist=self.cpi_hist,
                core_hist=self.core_hist,
                u=u_now,
                gap=gap,
                core_weight=self.framework.core_weight,
            )
            self.framework.update_rstar(g_obs)
            self.framework.update_strategy(
                pi_pol=info.pi_pol,
                pi_star=self.pi_star,
                u=info.u,
                spread=spread,
                s0=s0,
                gate=gate,
            )
            met = self.framework.is_meeting(self.month)
        else:
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
                if self.framework is not None:
                    self.framework.rho_q = float(smoothing)
            if phi_pi is not None and self.framework is not None:
                self.framework.phi_pi = float(phi_pi)
            if phi_y is not None and self.framework is not None:
                self.framework.phi_y = float(phi_y)
            if rate_override is not None:
                self.r_rule = float(rate_override)
                self.r = max(self.elb, float(rate_override) + z_mon)
                if self.framework is not None:
                    self.framework.r_ann = float(rate_override)
            elif self.framework is not None:
                self._framework_meet(
                    gap,
                    z_mon,
                    u=u_now,
                    makeup=makeup,
                    fci=fci,
                    risk_off=risk_off,
                    guidance=guidance,
                    credit_gap=credit_gap,
                )
            else:
                target = taylor_target(self.r_n, self.pi_star, self.phi_pi, self.phi_y, self.pi_pol(), gap)
                self.r_rule = self.rho * self.r_rule + (1.0 - self.rho) * target
                self.r = max(self.elb, self.r_rule + z_mon)
        else:
            announced = self.framework.r_ann if self.framework is not None else self.r_rule
            if self.framework is not None:
                self._apply_elb(announced, z_mon, gap, guidance)
            else:
                self.r = max(self.elb, announced + z_mon)
        self.month += 1
        return met

    def _apply_elb(self, shadow: float, z_mon: float, gap: float, guidance: float | None) -> None:
        """Record the shadow rate; at the ELB engage guidance then QE (LOLR is a lever)."""
        fw = self.framework
        if fw is None:
            self.r = max(self.elb, shadow + z_mon)
            return
        engaged, r_elb, qe = elb_toolkit(
            shadow=shadow,
            elb=self.elb,
            gap=gap,
            guidance=guidance,
            credibility=fw.guidance_cred,
            qe_per_gap=fw.qe_per_gap,
        )
        fw.shadow_rate = float(shadow)
        fw.qe_intent = qe
        fw.last_elb_tools = engaged
        self.r = max(self.elb, r_elb + z_mon)

    def _framework_meet(
        self,
        gap: float,
        z_mon: float,
        *,
        u: float,
        makeup: float | None,
        fci: float | None,
        risk_off: float | None,
        guidance: float | None,
        credit_gap: float,
    ) -> None:
        assert self.framework is not None
        fw = self.framework
        info = fw.last_info
        r_rule, r_ann, payload = fw.decide(
            r_n=self.r_n,
            pi_star=self.pi_star,
            pi_pol=float(info.get("pi_pol", self.pi_pol())),
            u=float(info.get("u", u)),
            u_star=self.u_star,
            gap=float(info.get("gap", gap)),
            r_rule=self.r_rule,
            makeup=fw.last_makeup if makeup is None else makeup,
            fci_tighten=fw.last_fci if fci is None else fci,
            risk_off=fw.last_risk_off if risk_off is None else risk_off,
            credit_gap=credit_gap,
        )
        self.r_rule = r_rule
        self._apply_elb(r_ann, z_mon, gap, guidance)
        payload = dict(payload)
        payload["shadow"] = fw.shadow_rate
        payload["elb_tools"] = list(fw.last_elb_tools)
        payload["qe"] = fw.qe_intent
        self.last_payload = payload

    def schedule_meetings(self, queue: EventQueue, horizon_months: int, days_per_month: int = 21) -> None:
        """Queue a ``("cb_meeting", month)`` payload on each meeting tick."""
        for m in range(horizon_months):
            meet = self.framework.is_meeting(m) if self.framework is not None else is_quarter_end_month(m)
            if meet:
                tick = (m + 1) * days_per_month - 1
                queue.schedule(tick, {"kind": "cb_meeting", "month": m}, priority=0)
