"""T6.01 — markets.yaml, instrument registry, ADV rule."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config, load_config
from marketsim.core.errors import ConfigError
from marketsim.layer1.build_io import CODES
from marketsim.market.instruments import (
    FIRM_PREFIX,
    NPC_PREFIX,
    Instrument,
    InstrumentRegistry,
    MarketsFile,
    average_daily_volume,
    firm_symbol,
    npc_symbol,
)


def _registry(cfg: Config) -> InstrumentRegistry:
    assert cfg.markets is not None
    return InstrumentRegistry.from_markets(cfg.markets, cfg.codes)


def test_shipped_markets_yaml_loads(cfg: Config) -> None:
    assert cfg.markets is not None
    m = cfg.markets
    assert m.turnover == pytest.approx(0.004)
    assert m.pricing.mode == "structural"
    assert m.pricing.lambda_m == pytest.approx(0.978)
    assert m.pricing.horizon_m == 120
    assert m.mm.participation_cap == pytest.approx(0.25)
    assert m.impact.delta == pytest.approx(0.5)
    assert m.impact.half_lives_d == (1.0, 5.0, 20.0, 60.0, 250.0)
    assert m.clob.halt_band == pytest.approx(0.20)
    assert m.fees.transaction_tax == pytest.approx(0.0)
    assert m.margin.equity_initial == pytest.approx(0.50)
    assert m.margin.equity_maintenance == pytest.approx(0.25)
    assert {s.symbol for s in m.static} >= {
        "GB_BILL",
        "GB_NOTE",
        "GB_BOND",
        "CORP_POOL",
        "CASH",
        "OIL",
        "METALS",
        "GRAINS",
    }


def test_registry_lists_npc_bonds_cash_commodities_and_indices(cfg: Config) -> None:
    reg = _registry(cfg)
    assert tuple(cfg.codes) == CODES
    npc = [npc_symbol(c) for c in CODES]
    assert all(s in reg for s in npc)
    assert len(npc) == 18
    for code in CODES:
        inst = reg.get(npc_symbol(code))
        assert inst.kind == "eq_npc"
        assert inst.venue == "engine_mm"
        assert inst.tradable is True
        assert inst.sector == code
        idx = reg.get(f"IDX:{code}")
        assert idx.kind == "index"
        assert idx.tradable is False
    for name in ("GB_BILL", "GB_NOTE", "GB_BOND", "CORP_POOL"):
        assert reg.get(name).tradable is True
        assert reg.get(name).venue == "auction_mm"
    assert reg.get("CASH").tradable is False
    assert reg.get("OIL").sector == "ENERGY"
    assert reg.get("METALS").sector == "MATERIALS"
    assert reg.get("GRAINS").sector == "AGRIFOOD"
    assert reg.get("IDX:MARKET").tradable is False
    assert reg.get("IDX:BOND").tradable is False
    assert list(reg) == [reg.get(s) for s in reg.symbols()]
    assert list(reg.symbols()) == sorted(reg.symbols())


def test_adv_rule(cfg: Config) -> None:
    reg = _registry(cfg)
    cap = 250.0
    assert average_daily_volume(cap, 0.004) == pytest.approx(1.0)
    assert reg.adv(cap) == pytest.approx(0.004 * cap)
    assert reg.adv(cap, symbol="EQ:NPC:AUTOS") == pytest.approx(1.0)
    with pytest.raises(ConfigError, match="unknown"):
        reg.adv(cap, symbol="EQ:FIRM:missing")


def test_dynamic_firm_listing(cfg: Config) -> None:
    reg = _registry(cfg)
    assert "EQ:FIRM:acme" not in reg
    listed = reg.list_firm("acme")
    assert listed.symbol == firm_symbol("acme") == f"{FIRM_PREFIX}acme"
    assert listed.kind == "eq_firm"
    assert listed.venue == "clob"
    assert listed.tradable is True
    again = reg.list_firm("acme")
    assert again.symbol == listed.symbol
    assert len([i for i in reg if i.symbol.startswith(FIRM_PREFIX)]) == 1
    with pytest.raises(ConfigError, match="invalid firm id"):
        reg.list_firm("bad:id")
    with pytest.raises(ConfigError, match="invalid firm id"):
        reg.list_firm("")


def test_registry_roundtrip(cfg: Config) -> None:
    reg = _registry(cfg)
    reg.list_firm("zeta")
    reg.list_firm("alpha")
    restored = InstrumentRegistry.from_state(reg.to_state())
    assert restored.to_state() == reg.to_state()
    assert restored.turnover == pytest.approx(0.004)
    assert restored.symbols() == reg.symbols()
    assert [i.symbol for i in restored] == list(reg.symbols())
    assert restored.get("EQ:FIRM:alpha").venue == "clob"


def test_duplicate_symbol_rejected() -> None:
    inst = Instrument("EQ:NPC:AUTOS", "eq_npc", "engine_mm", True, "AUTOS")
    with pytest.raises(ConfigError, match="duplicate"):
        InstrumentRegistry([inst, inst], turnover=0.004)


def test_override_turnover(config_dir) -> None:
    cfg = load_config(config_dir, {"markets.turnover": 0.01})
    assert cfg.markets is not None
    assert cfg.markets.turnover == pytest.approx(0.01)
    reg = InstrumentRegistry.from_markets(cfg.markets, cfg.codes)
    assert reg.adv(100.0) == pytest.approx(1.0)


def test_markets_file_rejects_bad_margin() -> None:
    with pytest.raises(Exception, match="equity_maintenance"):
        MarketsFile.model_validate({"margin": {"equity_initial": 0.2, "equity_maintenance": 0.5}})


def test_npc_prefix_constant() -> None:
    assert NPC_PREFIX == "EQ:NPC:"
    assert npc_symbol("SOFTWARE") == "EQ:NPC:SOFTWARE"
