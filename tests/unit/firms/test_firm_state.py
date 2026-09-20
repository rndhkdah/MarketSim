"""T5.01 — firms.yaml, Firm / Plant state, registry."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config, load_config
from marketsim.core.errors import ConfigError
from marketsim.firms.firm import (
    LEVERS,
    Firm,
    FirmRegistry,
    FirmsFile,
    Plant,
    cells_from_lists,
)


def _cells() -> frozenset[tuple[str, str]]:
    return cells_from_lists(("CAPITAL", "INDUSTRIAL"), ("AUTOS", "SOFTWARE"))


def _firm(fid: str = "acme", *, region: str = "INDUSTRIAL", sector: str = "AUTOS") -> Firm:
    return Firm(
        id=fid,
        operator="alice",
        plants=(Plant(region=region, sector=sector, capacity=10.0, vintage=4),),
        employees=3.0,
        wage_offer=1.2,
        cash=50.0,
        posted_price={(region, sector): 1.05},
        autopilot={"pricing": False},
    )


def test_shipped_firms_yaml_loads(cfg: Config) -> None:
    assert cfg.firms is not None
    f = cfg.firms
    assert f.npc_initial_share == 1.0
    assert f.goods_market.epsilon_s == 4.0
    assert f.goods_market.tau_loyalty_m == 6.0
    assert f.pricing.step_max_month == 0.15
    assert f.pricing.floor_of_pref == 0.01
    assert f.production.overtime_cap == 1.10
    assert f.labour.hire_share_of_unemployed == 0.10
    assert f.labour.firing_cost_months == 2.0
    assert f.procurement.premium_max == 0.5
    assert f.capex.new_cell_setup_premium == 0.10
    assert f.capex.capacity_growth_cap_year == 0.50
    assert f.financing.nd_ebitda_soft == 4.0
    assert f.financing.nd_ebitda_hard == 6.0
    assert f.financing.rating_multipliers["CCC"] == 0.25
    assert f.bankruptcy.plant_recovery == 0.70
    assert f.reports.monthly_lag_days == 10
    assert set(f.financing.rating_multipliers) == {
        "AAA",
        "AA",
        "A",
        "BBB",
        "BB",
        "B",
        "CCC",
    }
    # Phase-2 aggregate block is unchanged.
    assert cfg.dynamics is not None
    assert cfg.dynamics.firms.kappa_leverage == 0.03


def test_firms_file_rejects_hard_cap_below_soft() -> None:
    with pytest.raises(Exception, match="nd_ebitda_hard"):
        FirmsFile.model_validate({"financing": {"nd_ebitda_soft": 6.0, "nd_ebitda_hard": 4.0}})


def test_override_npc_share(config_dir) -> None:
    cfg = load_config(config_dir, {"firms.npc_initial_share": 0.6})
    assert cfg.firms is not None
    assert cfg.firms.npc_initial_share == pytest.approx(0.6)


def test_firm_state_roundtrip() -> None:
    firm = _firm()
    other = Firm.from_state(firm.to_state())
    assert other.to_state() == firm.to_state()
    assert other.plants[0].cell == ("INDUSTRIAL", "AUTOS")
    assert other.autopilot["pricing"] is False
    assert all(other.autopilot[k] is True for k in LEVERS if k != "pricing")


def test_registry_roundtrip_and_sorted_iteration() -> None:
    cells = _cells()
    reg = FirmRegistry([_firm("zeta"), _firm("alpha")], cells=cells)
    assert reg.ids == ("alpha", "zeta")
    assert [f.id for f in reg] == ["alpha", "zeta"]
    restored = FirmRegistry.from_state(reg.to_state())
    assert restored.to_state() == reg.to_state()
    assert [f.id for f in restored] == ["alpha", "zeta"]


def test_unknown_plant_cell_rejected() -> None:
    cells = _cells()
    with pytest.raises(ConfigError, match="unknown cell"):
        FirmRegistry([_firm(sector="MOON")], cells=cells)
    with pytest.raises(ConfigError, match="unknown cell"):
        FirmRegistry([_firm(region="MARS")], cells=cells)


def test_duplicate_id_rejected() -> None:
    cells = _cells()
    with pytest.raises(ConfigError, match="duplicate"):
        FirmRegistry([_firm("acme"), _firm("acme")], cells=cells)


def test_add_keeps_sorted_ids() -> None:
    reg = FirmRegistry([_firm("m")], cells=_cells())
    reg.add(_firm("a"))
    assert reg.ids == ("a", "m")
