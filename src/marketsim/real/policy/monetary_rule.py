"""Committee calendar and dual-mandate reaction function (D15 / §2.14.1–§2.14.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.core.config import Config


def meeting_months(mpy: int) -> tuple[int, ...]:
    """0-based month-of-year indices that host a meeting. ``mpy=4`` → 2,5,8,11."""
    if mpy <= 0:
        return ()
    months = [int(round(i * 12 / mpy) - 1) % 12 for i in range(1, mpy + 1)]
    return tuple(sorted(set(months)))


def is_meeting_month(month_index: int, mpy: int) -> bool:
    return (month_index % 12) in meeting_months(mpy)


def rho_per_meeting(rho_quarterly: float, mpy: int) -> float:
    """``ρ^(4/mpy)`` so gradualism is per-meeting, not per-quarter."""
    if mpy <= 0:
        return float(rho_quarterly)
    return float(rho_quarterly ** (4.0 / mpy))


def grid_rate(r: float, step: float) -> float:
    """Snap to the ``step`` grid. ``step<=0`` is continuous."""
    if step <= 0:
        return float(r)
    return float(round(float(r) / step) * step)


def announce(r_rule: float, r_ann: float, step: float, deadband: float) -> float:
    """New announced rate, or the previous one if the grid move is inside the deadband."""
    if step <= 0:
        return float(r_rule)
    snapped = grid_rate(r_rule, step)
    if abs(snapped - r_ann) >= deadband:
        return snapped
    return float(r_ann)


def reaction_target(
    *,
    r_n: float,
    pi_star: float,
    phi_pi: float,
    phi_u: float,
    phi_y: float,
    pi_pol: float,
    u: float,
    u_star: float,
    gap: float,
    g_trend: float = 0.0,
    kappa_rstar: float = 0.0,
    makeup: float = 0.0,
    fci_tighten: float = 0.0,
    risk_off: float = 0.0,
    phi_credit: float = 0.0,
    credit_gap: float = 0.0,
) -> float:
    """§2.14.2 target (annual decimal). Strategy terms default off."""
    r_star = r_n + kappa_rstar * g_trend
    return (
        r_star
        + pi_star
        + phi_pi * (pi_pol - pi_star)
        + phi_u * (u_star - u)
        + phi_y * gap
        + makeup
        - fci_tighten
        - risk_off
        + phi_credit * credit_gap
    )


def statement_tone(pi_pol: float, pi_star: float, u: float, u_star: float) -> int:
    """Ordinal −1 / 0 / +1 from the rule's inflation and unemployment gaps."""
    infl = pi_pol - pi_star
    slack = u - u_star
    if infl > 0.002 and slack <= 0.002:
        return 1
    if infl < -0.002 or slack > 0.002:
        return -1
    return 0


def step_plevel_gap(plevel: float, pi_pol: float, pi_star: float, decay: float, clip: float) -> float:
    """Leaky cumulative price-level gap, clipped to ``±clip``. Units: log-points."""
    nxt = float(decay) * float(plevel) + (float(pi_pol) - float(pi_star)) / 12.0
    lim = abs(float(clip))
    return max(-lim, min(lim, nxt))


def sahm_risk_off(
    u_hist: list[float],
    *,
    threshold: float,
    cut: float,
    decay_m: int,
    residual: float,
) -> tuple[float, float]:
    """Return ``(risk_off, new_residual)``. Trigger: 3m U − 12m min ≥ threshold."""
    if residual > 0:
        nxt = max(0.0, residual - cut / max(decay_m, 1))
        return residual, nxt
    if len(u_hist) < 15:
        return 0.0, 0.0
    u3 = sum(u_hist[-3:]) / 3.0
    u_min = min(u_hist[-15:-3])
    if u3 - u_min >= threshold:
        return float(cut), float(cut) * (1.0 - 1.0 / max(decay_m, 1))
    return 0.0, 0.0


def fci_tighten(
    *,
    spread: float,
    s0: float,
    gate: float,
    phi_fci: float,
    w_spread: float,
    w_gate: float,
    equity_drawdown: float = 0.0,
    term_premium: float = 0.0,
    w_eq: float = 0.0,
    w_tp: float = 0.0,
) -> float:
    """Weighted FCI. Equity term is 0 until Phase 6."""
    del equity_drawdown, term_premium
    cs = (float(spread) - float(s0)) / 0.01
    return float(phi_fci) * (w_spread * cs + w_gate * (1.0 - float(gate)) + w_eq * 0.0 + w_tp * 0.0)


def elb_toolkit(
    *,
    shadow: float,
    elb: float,
    gap: float,
    guidance: float | None,
    credibility: float,
    qe_per_gap: float,
) -> tuple[tuple[str, ...], float, float]:
    """Order at the ELB: guidance → QE → (LOLR is a separate lever). Returns engaged, r, qe/GDP."""
    engaged: list[str] = []
    r = float(shadow)
    qe = 0.0
    if r >= elb:
        return (), r, 0.0
    if guidance is not None:
        r = float(credibility) * float(guidance) + (1.0 - float(credibility)) * r
        engaged.append("guidance")
    r = max(elb, r)
    if r <= elb + 1e-15 and gap < 0:
        qe = float(qe_per_gap) * (-float(gap))
        engaged.append("qe")
    return tuple(engaged), r, qe


def infl_from_hist(hist: list[float], months: int) -> float:
    """Annualised log change over ``months``. Units: annual decimal."""
    return float((12.0 / months) * np.log(hist[-1] / hist[-1 - months]))


def slice_lag(hist: list[float], lag_m: int, *, min_len: int = 13) -> list[float]:
    """Drop the newest ``lag_m`` unpublished months; left-pad so π12 is defined."""
    lag = max(0, int(lag_m))
    if lag == 0:
        out = list(hist)
    elif lag >= len(hist):
        out = [float(hist[0])] if hist else [1.0]
    else:
        out = list(hist)[:-lag]
    while len(out) < min_len:
        out.insert(0, out[0] if out else 1.0)
    return out


def vintage_pi_pol(
    cpi_hist: list[float],
    core_hist: list[float],
    *,
    lag_m: int,
    core_weight: float,
) -> float:
    """§2.14.2 policy inflation on the published CPI vintage (lag in months)."""
    cpi = slice_lag(cpi_hist, lag_m)
    core = slice_lag(core_hist, lag_m)
    headline = 0.5 * infl_from_hist(cpi, 3) + 0.5 * infl_from_hist(cpi, 12)
    core_pi = 0.5 * infl_from_hist(core, 3) + 0.5 * infl_from_hist(core, 12)
    w = float(core_weight)
    return float((1.0 - w) * headline + w * core_pi)


@dataclass
class InformationSet:
    """What the committee can see this meeting. Units: annual decimal / share."""

    pi_pol: float
    u: float
    gap: float
    source: str


def committee_draw(seed: int, meeting_n: int, dispersion_bp: float) -> float:
    """Deterministic-per-seed draw in annual-decimal units."""
    if dispersion_bp == 0.0:
        return 0.0
    h = (int(seed) * 0x9E3779B9 + int(meeting_n) * 0x85EBCA6B) & 0xFFFFFFFF
    unit = (h / 0xFFFFFFFF) * 2.0 - 1.0
    return (dispersion_bp / 10_000.0) * unit


def projection_path(
    r_rule: float,
    target: float,
    rho_meet: float,
    step: float,
    deadband: float,
    r_ann: float,
    n: int = 8,
    noise_bp: float = 0.0,
    seed: int = 0,
    meeting_n: int = 0,
) -> list[float]:
    """Dot-plot: the rule iterated under frozen gaps."""
    path: list[float] = []
    rr, ra = float(r_rule), float(r_ann)
    for i in range(n):
        rr = rho_meet * rr + (1.0 - rho_meet) * target
        ra = announce(rr, ra, step, deadband)
        noise = committee_draw(seed + 17, meeting_n + i + 1, noise_bp)
        path.append(ra + noise)
    return path


@dataclass
class MonetaryRule:
    """Autopilot of CENBANK: calendar, dual mandate, grid, communication."""

    mpy: int = 8
    rate_step: float = 0.0025
    deadband: float = 0.0010
    blackout_days: int = 10
    minutes_lag_days: int = 21
    phi_pi: float = 1.50
    phi_u: float = 1.0
    phi_y: float = 0.0
    core_weight: float = 0.5
    rho_q: float = 0.80
    kappa_rstar: float = 1.0
    rstar_tau_m: int = 60
    dispersion_bp: float = 0.0
    projection_noise_bp: float = 25.0
    seed: int = 0
    g_trend: float = 0.0
    r_ann: float = 0.0
    meeting_n: int = 0
    last_decision: dict[str, Any] = field(default_factory=dict)
    pending_minutes: dict[str, Any] | None = None
    makeup: float = 0.0
    makeup_decay: float = 0.98
    makeup_clip: float = 0.02
    plevel: float = 0.0
    sahm_on: bool = False
    sahm_threshold: float = 0.005
    sahm_cut: float = 0.005
    sahm_decay_m: int = 12
    sahm_residual: float = 0.0
    u_hist: list[float] = field(default_factory=list)
    phi_fci: float = 0.0
    w_spread: float = 0.4
    w_gate: float = 0.3
    phi_credit: float = 0.0
    guidance_cred: float = 0.7
    qe_per_gap: float = 0.02
    last_makeup: float = 0.0
    last_fci: float = 0.0
    last_risk_off: float = 0.0
    shadow_rate: float = 0.0
    qe_intent: float = 0.0
    last_elb_tools: tuple[str, ...] = ()
    cpi_lag_m: int = 1
    unemployment_lag_m: int = 1
    gdp_lag_q: int = 1
    use_published_vintages: bool = True
    _true_u: list[float] = field(default_factory=list)
    _true_gap: list[float] = field(default_factory=list)
    last_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Config, *, r0: float, seed: int = 0) -> MonetaryRule:
        assert cfg.policy is not None
        mon = cfg.policy.monetary
        tay = cfg.edges.policy.taylor
        phi_y = tay.phi_output_gap * mon.rule.phi_y_mult
        return cls(
            mpy=mon.calendar.meetings_per_year,
            rate_step=mon.calendar.rate_step,
            deadband=mon.calendar.deadband,
            blackout_days=mon.calendar.blackout_days,
            minutes_lag_days=mon.calendar.minutes_lag_days,
            phi_pi=mon.rule.phi_inflation,
            phi_u=mon.rule.phi_u,
            phi_y=phi_y,
            core_weight=mon.rule.core_weight,
            rho_q=mon.rule.smoothing,
            kappa_rstar=mon.rule.kappa_rstar,
            rstar_tau_m=mon.rule.rstar_tau_m,
            dispersion_bp=mon.committee.dispersion_bp,
            projection_noise_bp=mon.committee.projection_noise_bp,
            seed=int(seed),
            r_ann=float(r0),
            makeup=mon.strategy.makeup,
            makeup_decay=mon.strategy.makeup_decay,
            makeup_clip=mon.strategy.makeup_clip,
            sahm_on=mon.risk_management.enabled,
            sahm_threshold=mon.risk_management.sahm_threshold,
            sahm_cut=mon.risk_management.sahm_cut,
            sahm_decay_m=mon.risk_management.decay_m,
            phi_fci=mon.financial_conditions.phi_fci,
            w_spread=mon.financial_conditions.weights.get("corp_spread", 0.4),
            w_gate=mon.financial_conditions.weights.get("capital_gate", 0.3),
            phi_credit=mon.credit.phi_credit,
            guidance_cred=mon.elb.guidance_credibility,
            qe_per_gap=mon.elb.qe_per_gap_point,
            cpi_lag_m=mon.data.cpi_lag_m,
            unemployment_lag_m=mon.data.unemployment_lag_m,
            gdp_lag_q=mon.data.gdp_lag_q,
            use_published_vintages=mon.data.use_published_vintages,
        )

    @property
    def rho_meet(self) -> float:
        return rho_per_meeting(self.rho_q, self.mpy)

    def is_meeting(self, month_index: int) -> bool:
        return is_meeting_month(month_index, self.mpy)

    def information_set(
        self,
        *,
        cpi_hist: list[float],
        core_hist: list[float],
        u: float,
        gap: float,
        core_weight: float | None = None,
    ) -> InformationSet:
        """CPI 1m, unemployment 1m, GDP 1q when ``use_published_vintages``; else the oracle."""
        w = self.core_weight if core_weight is None else float(core_weight)
        self._true_u.append(float(u))
        self._true_gap.append(float(gap))
        if len(self._true_u) > 36:
            self._true_u = self._true_u[-36:]
            self._true_gap = self._true_gap[-36:]
        if not self.use_published_vintages:
            pi = vintage_pi_pol(cpi_hist, core_hist, lag_m=0, core_weight=w)
            info = InformationSet(pi, float(u), float(gap), "oracle")
            self.last_info = {"pi_pol": info.pi_pol, "u": info.u, "gap": info.gap, "source": info.source}
            return info
        pi = vintage_pi_pol(cpi_hist, core_hist, lag_m=self.cpi_lag_m, core_weight=w)
        lu = int(self.unemployment_lag_m)
        u_see = self._true_u[-1 - lu] if len(self._true_u) > lu else float(u)
        lg = 3 * int(self.gdp_lag_q)
        gap_see = self._true_gap[-1 - lg] if len(self._true_gap) > lg else 0.0
        info = InformationSet(pi, float(u_see), float(gap_see), "vintage")
        self.last_info = {"pi_pol": info.pi_pol, "u": info.u, "gap": info.gap, "source": info.source}
        return info

    def in_blackout(self, month_index: int) -> bool:
        """True if a meeting is within ``blackout_days`` (converted to months)."""
        horizon = max(1, (self.blackout_days + 20) // 21)
        return any(self.is_meeting(month_index + k) for k in range(1, horizon + 1))

    def update_rstar(self, g_obs: float) -> float:
        self.g_trend += (float(g_obs) - self.g_trend) / max(self.rstar_tau_m, 1)
        return self.g_trend

    def update_strategy(
        self,
        *,
        pi_pol: float,
        pi_star: float,
        u: float,
        spread: float,
        s0: float,
        gate: float,
    ) -> tuple[float, float, float]:
        """Monthly leak of the makeup gap, Sahm residual and FCI. Returns the three terms."""
        if self.makeup > 0:
            self.plevel = step_plevel_gap(
                self.plevel, pi_pol, pi_star, self.makeup_decay, self.makeup_clip
            )
            self.last_makeup = self.makeup * self.plevel
        else:
            self.last_makeup = 0.0
        self.u_hist.append(float(u))
        if len(self.u_hist) > 24:
            self.u_hist = self.u_hist[-24:]
        if self.sahm_on:
            self.last_risk_off, self.sahm_residual = sahm_risk_off(
                self.u_hist,
                threshold=self.sahm_threshold,
                cut=self.sahm_cut,
                decay_m=self.sahm_decay_m,
                residual=self.sahm_residual,
            )
        else:
            self.last_risk_off = 0.0
        self.last_fci = fci_tighten(
            spread=spread,
            s0=s0,
            gate=gate,
            phi_fci=self.phi_fci,
            w_spread=self.w_spread,
            w_gate=self.w_gate,
        )
        return self.last_makeup, self.last_fci, self.last_risk_off

    def decide(
        self,
        *,
        r_n: float,
        pi_star: float,
        pi_pol: float,
        u: float,
        u_star: float,
        gap: float,
        r_rule: float,
        makeup: float = 0.0,
        fci_tighten: float = 0.0,
        risk_off: float = 0.0,
        credit_gap: float = 0.0,
    ) -> tuple[float, float, dict[str, Any]]:
        """One meeting. Returns ``(r_rule, r_ann, payload)``."""
        target = reaction_target(
            r_n=r_n,
            pi_star=pi_star,
            phi_pi=self.phi_pi,
            phi_u=self.phi_u,
            phi_y=self.phi_y,
            pi_pol=pi_pol,
            u=u,
            u_star=u_star,
            gap=gap,
            g_trend=self.g_trend,
            kappa_rstar=self.kappa_rstar,
            makeup=makeup,
            fci_tighten=fci_tighten,
            risk_off=risk_off,
            phi_credit=self.phi_credit,
            credit_gap=credit_gap,
        )
        rho = self.rho_meet
        r_rule = rho * r_rule + (1.0 - rho) * target
        r_rule += committee_draw(self.seed, self.meeting_n, self.dispersion_bp)
        r_ann = announce(r_rule, self.r_ann, self.rate_step, self.deadband)
        self.r_ann = r_ann
        self.meeting_n += 1
        dots = projection_path(
            r_rule,
            target,
            rho,
            self.rate_step,
            self.deadband,
            r_ann,
            noise_bp=self.projection_noise_bp,
            seed=self.seed,
            meeting_n=self.meeting_n,
        )
        payload = {
            "rate": r_ann,
            "r_rule": r_rule,
            "target": target,
            "tone": statement_tone(pi_pol, pi_star, u, u_star),
            "dots": dots,
            "meeting_n": self.meeting_n,
        }
        self.last_decision = payload
        return r_rule, r_ann, payload

    def to_state(self) -> dict[str, Any]:
        return {
            "g_trend": self.g_trend,
            "r_ann": self.r_ann,
            "meeting_n": self.meeting_n,
            "last_decision": dict(self.last_decision),
            "pending_minutes": dict(self.pending_minutes) if self.pending_minutes else None,
            "plevel": self.plevel,
            "sahm_residual": self.sahm_residual,
            "u_hist": list(self.u_hist),
            "last_makeup": self.last_makeup,
            "last_fci": self.last_fci,
            "last_risk_off": self.last_risk_off,
            "shadow_rate": self.shadow_rate,
            "qe_intent": self.qe_intent,
            "last_elb_tools": list(self.last_elb_tools),
            "true_u": list(self._true_u),
            "true_gap": list(self._true_gap),
            "last_info": dict(self.last_info),
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.g_trend = float(state.get("g_trend", 0.0))
        self.r_ann = float(state.get("r_ann", self.r_ann))
        self.meeting_n = int(state.get("meeting_n", 0))
        self.last_decision = dict(state.get("last_decision") or {})
        pend = state.get("pending_minutes")
        self.pending_minutes = dict(pend) if pend else None
        self.plevel = float(state.get("plevel", 0.0))
        self.sahm_residual = float(state.get("sahm_residual", 0.0))
        self.u_hist = [float(x) for x in (state.get("u_hist") or [])]
        self.last_makeup = float(state.get("last_makeup", 0.0))
        self.last_fci = float(state.get("last_fci", 0.0))
        self.last_risk_off = float(state.get("last_risk_off", 0.0))
        self.shadow_rate = float(state.get("shadow_rate", 0.0))
        self.qe_intent = float(state.get("qe_intent", 0.0))
        self.last_elb_tools = tuple(state.get("last_elb_tools") or ())
        self._true_u = [float(x) for x in (state.get("true_u") or [])]
        self._true_gap = [float(x) for x in (state.get("true_gap") or [])]
        self.last_info = dict(state.get("last_info") or {})
