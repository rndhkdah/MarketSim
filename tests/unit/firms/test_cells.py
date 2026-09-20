"""T5.04 — cell aggregates over NPC + firms, including zero-NPC cells."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.firms.cells import (
    aggregate_cell,
    aggregates_for_registry,
    economy_cell_aggregates,
    step_zero_npc_cell,
)
from marketsim.firms.firm import Firm, FirmRegistry, Plant, cells_from_lists
from marketsim.firms.founding import CellBook, NpcCell
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy


def test_aggregates_equal_sums() -> None:
    firms = FirmRegistry(
        [
            Firm(
                id="a",
                operator="x",
                plants=(Plant("INDUSTRIAL", "AUTOS", 10.0),),
                employees=4.0,
                posted_price={("INDUSTRIAL", "AUTOS"): 1.1},
                customer_share={("INDUSTRIAL", "AUTOS"): 0.5},
            ),
            Firm(
                id="b",
                operator="y",
                plants=(Plant("INDUSTRIAL", "AUTOS", 6.0),),
                employees=2.0,
                posted_price={("INDUSTRIAL", "AUTOS"): 0.9},
                customer_share={("INDUSTRIAL", "AUTOS"): 0.5},
            ),
        ],
        cells=cells_from_lists(("INDUSTRIAL",), ("AUTOS",)),
    )
    book = CellBook({("INDUSTRIAL", "AUTOS"): NpcCell("INDUSTRIAL", "AUTOS", 20.0, 40.0)})
    [agg] = aggregates_for_registry(
        firms,
        book,
        npc_output={("INDUSTRIAL", "AUTOS"): 18.0},
        npc_employment={("INDUSTRIAL", "AUTOS"): 8.0},
        npc_price={("INDUSTRIAL", "AUTOS"): 1.0},
        npc_sales={("INDUSTRIAL", "AUTOS"): 18.0},
    )
    assert agg.capacity == pytest.approx(36.0)
    assert agg.output == pytest.approx(18.0 + 16.0)
    assert agg.employment == pytest.approx(14.0)
    assert agg.sales == pytest.approx(18.0 + 5.0 + 3.0)
    assert np.isfinite(agg.as_vector()).all()


def test_zero_npc_pref_from_firms() -> None:
    agg = aggregate_cell(
        region="INDUSTRIAL",
        sector="AUTOS",
        npc_capacity=0.0,
        npc_output=0.0,
        npc_employment=0.0,
        npc_price=9.9,
        npc_sales=0.0,
        firm_capacity=np.array([4.0, 6.0]),
        firm_output=np.array([4.0, 6.0]),
        firm_employment=np.array([1.0, 1.0]),
        firm_prices=np.array([1.0, 2.0]),
        firm_sales=np.array([4.0, 6.0]),
    )
    assert agg.p_ref == pytest.approx(1.6)
    assert agg.utilisation == pytest.approx(1.0)


def test_zero_npc_cell_runs_120_months() -> None:
    reg = FirmRegistry(
        [
            Firm(
                id="solo",
                operator="z",
                plants=(Plant("INDUSTRIAL", "AUTOS", 5.0),),
                employees=1.0,
                posted_price={("INDUSTRIAL", "AUTOS"): 1.05},
            )
        ],
        cells=cells_from_lists(("INDUSTRIAL",), ("AUTOS",)),
    )
    hist = step_zero_npc_cell(reg, region="INDUSTRIAL", sector="AUTOS", demand=4.0, months=120)
    assert len(hist) == 120
    for agg in hist:
        assert np.isfinite(agg.as_vector()).all()
        assert agg.npc_capacity == 0.0
        assert agg.capacity == pytest.approx(5.0)
        assert agg.p_ref == pytest.approx(1.05)


def test_economy_aggregates_match_npc_when_no_firms(cfg: Config) -> None:
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, check_sfc=True)
    rows = eco.cell_aggregates()
    assert len(rows) == len(cfg.codes)
    autos = next(r for r in rows if r.sector == "AUTOS")
    si = eco.codes.index("AUTOS")
    assert autos.npc_capacity == pytest.approx(float(np.asarray(eco.k).reshape(-1, 18)[0, si]))
    assert autos.firm_capacity == 0.0
    assert autos.output == pytest.approx(float(np.asarray(eco.x).reshape(-1, 18)[0, si]))
    # Same helper, same numbers.
    again = economy_cell_aggregates(eco)
    assert again[0].capacity == rows[0].capacity
