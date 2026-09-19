"""Committee calendar and dual-mandate reaction function (D15 / §2.14.1–§2.14.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        )

    @property
    def rho_meet(self) -> float:
        return rho_per_meeting(self.rho_q, self.mpy)

    def is_meeting(self, month_index: int) -> bool:
        return is_meeting_month(month_index, self.mpy)

    def in_blackout(self, month_index: int) -> bool:
        """True if a meeting is within ``blackout_days`` (converted to months)."""
        horizon = max(1, (self.blackout_days + 20) // 21)
        return any(self.is_meeting(month_index + k) for k in range(1, horizon + 1))

    def update_rstar(self, g_obs: float) -> float:
        self.g_trend += (float(g_obs) - self.g_trend) / max(self.rstar_tau_m, 1)
        return self.g_trend

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
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.g_trend = float(state.get("g_trend", 0.0))
        self.r_ann = float(state.get("r_ann", self.r_ann))
        self.meeting_n = int(state.get("meeting_n", 0))
        self.last_decision = dict(state.get("last_decision") or {})
        pend = state.get("pending_minutes")
        self.pending_minutes = dict(pend) if pend else None
