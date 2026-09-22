"""T8.05 — historic replay vs stylised published paths (seed economy).

ADR-015: the concordance / ``z`` mapping is **proposed, not applied**. Historic
*levels* overflow SFC when raw price moves land on ``z_cost`` (QUESTIONS T4.11;
``claude/plan/reports/event-calibration.md``). This file asserts **direction +
ordering + rough sign** on the SFC-finite prefix. YAML seed distributions stay
as shipped (sign / order of magnitude). Level match is T8.04.

Published bands (calibration *targets*, not ShockBus values):

- 1973 oil — EIA / BP Statistical Review: Arabian Light ≈ $2.90 → $11.65
  (≈×4), Oct 1973–Jan 1974.
- 2008 GFC — FRED CSUSHPINSA peak–trough ≈ −27 % (2006–2012); FRED SP500
  Oct-2007–Mar-2009 ≈ −57 %. Seed: collateral −25…30 %, bank_equity −30…50 %,
  risk +400 bp. Quantity rationing, not a firm-specific spread (D14).
- 2020 COVID — BEA NIPA Q2-2020 real GDP −31.2 % SAAR (≈−9 % q/q); BLS CPS
  U-3 14.7 % Apr 2020. Seed: labour −10…15 %; want shifts; world_demand −10 %.
- 2022 energy — BLS CPI-U 9.1 % y/y Jun 2022; FOMC +525 bp in 16 months.
  Seed: cost_push ENERGY +0.5…0.9, AGRIFOOD +0.2.

Oil / energy: **12-month prefix only** (SFC / IEEE overflow by month 17–18).
GFC 24-month is marked ``slow``; the unmarked path is 12 months plus a
composition-only check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from marketsim.events.compose import AppliedShock, apply_composition
from marketsim.events.effects import ExtraEffects
from marketsim.events.schema import DistSpec, EventSpec, load_catalog
from marketsim.real.economy import RealEconomy
from marketsim.scenarios.irf import make_economy

TIERS = {"dynamics.households.demand_mode": "tiers_wants"}
CREDIT = {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True}


def _typical(dist: DistSpec) -> float:
    sign = float(dist.sign)
    if dist.dist == "fixed":
        return sign * float(dist.value)
    if dist.dist == "lognormal":
        return sign * float(dist.median)
    if dist.min is not None and dist.max is not None:
        return sign * 0.5 * (float(dist.min) + float(dist.max))
    return sign * float(dist.mean or dist.mode or 0.0)


def _fixed_spec(spec: EventSpec) -> EventSpec:
    raw = spec.model_dump()
    for shock in raw["composition"]:
        mag = _typical(DistSpec.model_validate(shock["magnitude"]))
        shock["magnitude"] = {"dist": "fixed", "value": abs(mag), "sign": 1.0 if mag >= 0 else -1.0}
    for extra in raw["effects_extra"]:
        m = extra["magnitude"]
        if isinstance(m, dict):
            mag = _typical(DistSpec.model_validate(m))
            extra["magnitude"] = float(mag)
    return EventSpec.model_validate(raw)


@dataclass
class ReplayPath:
    gap: np.ndarray
    cpi: np.ndarray
    x: np.ndarray
    x0: np.ndarray
    p: np.ndarray
    codes: tuple[str, ...]
    bank_equity: np.ndarray
    bank_equity0: float
    lam_coll: np.ndarray
    risk: np.ndarray
    spread: np.ndarray
    v_re: np.ndarray
    v_re0: float
    collateral_mult: float
    labour_mult: dict[str, float]
    want_shift: dict[str, float]

    def sx(self, code: str) -> np.ndarray:
        i = self.codes.index(code)
        return self.x[:, i] / self.x0[i] - 1.0

    def rel_p(self, code: str) -> np.ndarray:
        i = self.codes.index(code)
        basket = np.maximum(self.p.mean(axis=1), 1e-12)
        return self.p[:, i] / basket


def _script(
    eco: RealEconomy, ids: list[str], extras: ExtraEffects, rng: np.random.Generator
) -> list[AppliedShock]:
    cat = load_catalog("config/events")
    applied: list[AppliedShock] = []
    for eid in ids:
        spec = _fixed_spec(cat[eid])
        applied.extend(apply_composition(spec, eco.bus, rng, day=0, economy=eco, tick=1))
        if spec.effects_extra:
            extras.start(spec.effects_extra, rng, tick=0, economy=eco)
    return applied


def _run(config_dir: Path, ids: list[str], months: int = 12, *, overrides: dict | None = None) -> ReplayPath:
    eco: RealEconomy = make_economy(config_dir, pi_star=0.0, check_sfc=True, overrides=overrides or {})
    rng = np.random.default_rng(0)
    extras = ExtraEffects()
    x0 = np.asarray(eco.x, dtype=float).copy()
    eq0 = float(eco.bank_equity)
    v0 = float(eco.credit.v_re)
    _script(eco, ids, extras, rng)
    labour_mult = dict(extras.labour_mult)
    want_shift = dict(extras.want_shift)
    collateral_mult = float(extras.collateral_mult)
    gaps = np.zeros(months)
    cpi = np.zeros(months)
    xs = np.zeros((months, eco.real.S))
    ps = np.zeros((months, eco.real.S))
    eq = np.zeros(months)
    lam_coll = np.zeros(months)
    risk = np.zeros(months)
    spread = np.zeros((months, eco.real.S))
    v_re = np.zeros(months)
    for t in range(months):
        extras.apply(t * 21, eco)
        rec = eco.step_month()
        gaps[t] = rec["gdp"] / eco.fin.gdp0 - 1.0
        cpi[t] = rec["cpi"]
        xs[t] = rec["x"]
        ps[t] = np.asarray(eco.p, dtype=float)
        eq[t] = float(eco.bank_equity)
        lam_coll[t] = float(eco.credit.lam_coll)
        risk[t] = float(eco.bus.states["risk"])
        spread[t] = np.asarray(eco.spread, dtype=float)
        v_re[t] = float(eco.credit.v_re)
    return ReplayPath(
        gaps,
        cpi,
        xs,
        x0,
        ps,
        eco.codes,
        eq,
        eq0,
        lam_coll,
        risk,
        spread,
        v_re,
        v0,
        collateral_mult,
        labour_mult,
        want_shift,
    )


@pytest.mark.validation
def test_oil_1973_energy_rel_price_autos_realestate(config_dir: Path) -> None:
    # EIA/BP ×4 crude is a *target*. Seed z_cost ≈ 1.25 is not 24m SFC-finite
    # (QUESTIONS T4.11). Direction + ordering on the 12-month prefix only.
    path = _run(config_dir, ["oil_embargo_1973"], months=12)
    energy_rel = path.rel_p("ENERGY")
    assert float(energy_rel.mean()) > 1.0
    assert float(energy_rel.mean()) > float(path.rel_p("AUTOS").mean())
    assert float(energy_rel.mean()) > float(path.rel_p("REALESTATE").mean())
    assert float(path.sx("AUTOS").min()) < 0.0
    assert float(path.sx("REALESTATE").min()) < 0.0


@pytest.mark.validation
def test_gfc_2008_risk_collateral_composition(config_dir: Path) -> None:
    # FRED CSUSHPINSA / SP500 levels are not asserted (ADR-015). Seed composition
    # only: risk-off primitive, collateral haircut, bank-equity write-off.
    # D14: one corporate spread — do not invent a firm-specific rate.
    eco = make_economy(config_dir, pi_star=0.0, check_sfc=True, overrides=CREDIT)
    assert eco.cfg.dynamics is not None
    assert eco.cfg.dynamics.credit.pricing == "uniform"
    rng = np.random.default_rng(0)
    extras = ExtraEffects()
    eq0 = float(eco.bank_equity)
    applied = _script(eco, ["gfc_2008"], extras, rng)
    risk = [a for a in applied if a.kind == "risk_appetite"]
    assert risk and float(risk[0].magnitude) > 0.0
    # Seed midpoints: collateral −27.5 % → 0.725; write-off 40 % → 0.60 residual.
    assert 0.70 <= float(extras.collateral_mult) <= 0.75
    assert float(eco.bank_equity) < eq0
    assert 0.50 <= float(eco.bank_equity) / eq0 <= 0.70
    assert float(eco.credit.event_coll_mult) == float(extras.collateral_mult)


@pytest.mark.validation
def test_gfc_2008_gap_realestate_banks(config_dir: Path) -> None:
    # Output gap negative; REALESTATE and BANKS hit. Live loan rates stay
    # uniform (D14) — collateral / capital gate ration quantity, not price.
    path = _run(config_dir, ["gfc_2008"], months=12, overrides=CREDIT)
    assert float(path.gap.min()) < 0.0
    assert float(path.sx("REALESTATE").min()) < 0.0
    assert float(path.sx("BANKS").min()) < 0.0
    assert float(path.v_re.min()) < path.v_re0
    assert float(path.bank_equity.min()) < path.bank_equity0
    assert float(path.lam_coll.min()) < 1.0
    assert float(path.risk[0]) > 0.0
    assert float(np.ptp(path.spread, axis=1).max()) < 1e-12


@pytest.mark.validation
@pytest.mark.slow
def test_gfc_2008_24m_gap_realestate_banks(config_dir: Path) -> None:
    path = _run(config_dir, ["gfc_2008"], months=24, overrides=CREDIT)
    assert float(path.gap.min()) < 0.0
    assert float(path.sx("REALESTATE").min()) < 0.0
    assert float(path.sx("BANKS").min()) < 0.0
    assert float(np.ptp(path.spread, axis=1).max()) < 1e-12


@pytest.mark.validation
def test_covid_2020_health_vs_discret_want_window(config_dir: Path) -> None:
    # BEA NIPA / BLS CPS levels are T8.04. T4.11: labour cap dominates so HEALTH
    # never rises vs baseline; the want-shift window is HEALTH not below DISCRET
    # on the months 1–6 *mean* (slice [1:6]). Seed labour −10…15 % + want shifts.
    path = _run(config_dir, ["covid_2020"], months=12, overrides=TIERS)
    assert any(float(v) < 1.0 for v in path.labour_mult.values())
    assert float(path.want_shift.get("HEALTH", 0.0)) > 0.0
    assert float(path.want_shift.get("EATING_OUT_LEISURE", 0.0)) < 0.0
    assert float(path.sx("HEALTH")[1:6].mean()) > float(path.sx("DISCRET")[1:6].mean())


@pytest.mark.validation
def test_energy_inflation_2022_cpi_and_energy_price(config_dir: Path) -> None:
    # BLS CPI-U 9.1 % is the published *target*. Seed ENERGY z_cost 0.5–0.9
    # overflows after ~12m (QUESTIONS T4.11) — 12-month prefix only. Sign:
    # CPI up, ENERGY price up. No level match (ADR-015).
    path = _run(config_dir, ["energy_inflation_2022"], months=12)
    assert float(path.cpi.max()) > float(path.cpi[0])
    i = path.codes.index("ENERGY")
    assert float(path.p[:, i].max()) > float(path.p[0, i])
    assert float(path.rel_p("ENERGY").mean()) > 1.0
