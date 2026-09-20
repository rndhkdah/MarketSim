"""T4.11 — historic templates reproduce §4.5 directions within 24 months."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from marketsim.events.compose import apply_composition
from marketsim.events.effects import ExtraEffects
from marketsim.events.schema import DistSpec, EventSpec, load_catalog
from marketsim.real.economy import RealEconomy
from marketsim.scenarios.irf import make_economy

TIERS = {"dynamics.households.demand_mode": "tiers_wants"}


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
class Path:
    gap: np.ndarray
    cpi: np.ndarray
    x: np.ndarray
    x0: np.ndarray
    u: np.ndarray
    r: np.ndarray
    p: np.ndarray
    saving: np.ndarray
    ebitda: np.ndarray
    util: np.ndarray
    starts_proxy: np.ndarray
    codes: tuple[str, ...]

    def sx(self, code: str) -> np.ndarray:
        i = self.codes.index(code)
        return self.x[:, i] / self.x0[i] - 1.0

    def rel_p(self, code: str) -> np.ndarray:
        i = self.codes.index(code)
        basket = np.maximum(self.p.mean(axis=1), 1e-12)
        return self.p[:, i] / basket


def _run(config_dir, ids: list[str], months: int = 24, *, overrides: dict | None = None) -> Path:
    cat = load_catalog("config/events")
    eco: RealEconomy = make_economy(config_dir, pi_star=0.0, check_sfc=True, overrides=overrides or {})
    rng = np.random.default_rng(0)
    extras = ExtraEffects()
    x0 = np.asarray(eco.x, dtype=float).copy()
    for eid in ids:
        spec = _fixed_spec(cat[eid])
        apply_composition(spec, eco.bus, rng, day=0, economy=eco, tick=1)
        if spec.effects_extra:
            extras.start(spec.effects_extra, rng, tick=0, economy=eco)
    gaps = np.zeros(months)
    cpi = np.zeros(months)
    xs = np.zeros((months, eco.real.S))
    u = np.zeros(months)
    r = np.zeros(months)
    ps = np.zeros((months, eco.real.S))
    saving = np.zeros(months)
    ebitda = np.zeros((months, eco.real.S))
    util = np.zeros((months, eco.real.S))
    kpath = np.zeros((months, eco.real.S))
    for t in range(months):
        extras.apply(t * 21, eco)
        rec = eco.step_month()
        gaps[t] = rec["gdp"] / eco.fin.gdp0 - 1.0
        cpi[t] = rec["cpi"]
        xs[t] = rec["x"]
        u[t] = rec["U"]
        r[t] = rec["r"]
        ps[t] = np.asarray(eco.p, dtype=float)
        saving[t] = float(eco.last_agg.saving_rate) if eco.last_agg is not None else 0.0
        ebitda[t] = np.asarray(eco.eb_s, dtype=float)
        util[t] = np.asarray(eco.x, dtype=float) / np.maximum(np.asarray(eco.k, dtype=float), 1e-12)
        kpath[t] = np.asarray(eco.k, dtype=float)
    return Path(gaps, cpi, xs, x0, u, r, ps, saving, ebitda, util, kpath, eco.codes)


@pytest.mark.validation
def test_oil_embargo_directions(config_dir) -> None:
    # Seed z_cost ≈ 1.25 is not SFC-finite at 24m (QUESTIONS T4.11). Directions in 12m.
    on = _run(config_dir, ["oil_embargo_1973", "oil_monetary_tightening"], months=12)
    off = _run(config_dir, ["oil_embargo_1973"], months=12)
    i = on.codes.index("ENERGY")
    for p in (on, off):
        assert p.cpi.max() > p.cpi[0]
        assert float(p.gap.min()) < 0
        rev = p.p[:, i] * p.x[:, i]
        assert float(rev.max()) > float(rev[0])
        assert float(p.sx("AUTOS").min()) < 0
        assert float(p.sx("TRANSPORT").min()) < 0
    assert float(on.r.max()) >= on.r[0]


@pytest.mark.validation
def test_gfc_directions(config_dir) -> None:
    ov = {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True}
    on = _run(config_dir, ["gfc_2008", "gfc_fiscal_stimulus"], overrides=ov)
    off = _run(config_dir, ["gfc_2008"], overrides=ov)
    for p in (on, off):
        assert float(p.sx("CONSTRUCT")[0:24].min()) < float(p.sx("STAPLES")[0:24].min()) + 1e-9
        assert float(p.sx("AUTOS")[0:24].min()) < float(p.sx("STAPLES")[0:24].mean())
        assert float(p.u[0:24].max()) > p.u[0]
    assert float(on.r[6:24].min()) <= on.r[0] + 1e-9


@pytest.mark.validation
def test_covid_directions(config_dir) -> None:
    on = _run(config_dir, ["covid_2020", "covid_fiscal_stimulus"], overrides=TIERS)
    off = _run(config_dir, ["covid_2020"], overrides=TIERS)
    for p in (on, off):
        assert float(p.sx("DISCRET")[0:12].min()) < -0.15
        # Want-shift window (labour extra also cuts HEALTH; relative sign is the composition).
        assert float(p.sx("HEALTH")[1:6].mean()) > float(p.sx("DISCRET")[1:6].mean())
        assert float(p.saving[1:6].mean()) > float(p.saving[0])
    assert float(on.cpi.max()) > float(on.cpi[0])


@pytest.mark.validation
def test_chip_shortage_directions(config_dir) -> None:
    on = _run(config_dir, ["chip_shortage_2020", "auto_production_cuts"])
    off = _run(config_dir, ["chip_shortage_2020"])
    assert float(on.sx("AUTOS")[3:24].min()) < 0
    assert float(on.rel_p("SEMIS")[0:12].mean()) > float(on.rel_p("SOFTWARE")[0:12].mean())
    assert abs(float(on.sx("SOFTWARE")[0:12].mean())) < abs(float(on.sx("SEMIS")[0:12].mean())) + 0.05
    assert float(off.sx("SEMIS")[0:12].min()) < 0


@pytest.mark.validation
def test_energy_inflation_directions(config_dir) -> None:
    on = _run(config_dir, ["energy_inflation_2022", "energy_monetary_tightening"], months=12)
    off = _run(config_dir, ["energy_inflation_2022"], months=12)
    for p in (on, off):
        assert p.cpi.max() > p.cpi[0]
    assert float(on.r.max()) >= on.r[0]


@pytest.mark.validation
def test_tohoku_directions(config_dir) -> None:
    p = _run(config_dir, ["tohoku_2011", "auto_parts_shortage"])
    assert float(p.sx("AUTOS")[0:6].min()) < 0
    assert float(p.sx("CONSTRUCT").max()) > 0


@pytest.mark.validation
def test_ai_boom_directions(config_dir) -> None:
    p = _run(config_dir, ["ai_boom_2023"])
    semis = p.codes.index("SEMIS")
    util = p.codes.index("UTILITIES")
    assert float(p.util[12:24, semis].mean()) >= float(p.util[0:3, semis].mean()) - 1e-9
    assert float(p.starts_proxy[-1, semis]) >= float(p.starts_proxy[0, semis]) - 1e-9
    assert float(p.util[12:24, util].mean()) >= float(p.util[0:3, util].mean()) - 1e-9


@pytest.mark.validation
@pytest.mark.parametrize(
    "eid",
    ["dotcom_2000", "asian_crisis_1997", "suez_2021", "thailand_floods_2011"],
)
def test_other_templates_move_gdp_or_target(config_dir, eid: str) -> None:
    p = _run(config_dir, [eid])
    moved = abs(float(p.gap[0:24].mean())) > 1e-6 or abs(float(p.cpi[-1] - p.cpi[0])) > 1e-6
    if eid == "thailand_floods_2011":
        assert float(p.sx("SEMIS")[0:12].min()) < 0
    elif eid == "dotcom_2000":
        assert float(p.sx("SOFTWARE")[0:24].min()) < 0 or moved
    else:
        assert moved
