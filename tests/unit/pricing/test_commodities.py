"""T6.18 — OIL tracks ENERGY + carry; a 3× cost-push is transmitted without a level cap."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.layer1.build_io import CODES
from marketsim.market.instruments import COMMODITY_SECTORS
from marketsim.pricing.commodities import (
    CARRY,
    DT_DAY,
    DT_MONTH,
    commodity_price,
    commodity_prices,
    national_reference_price,
)


def test_oil_tracks_energy_reference_plus_carry() -> None:
    # §6.3 storage / convenience: annual decimal, applied per month here.
    carry = 0.12
    p_energy = 1.40
    factor = 1.0 + carry * DT_MONTH
    oil = commodity_price(p_energy, carry=carry, dt=DT_MONTH)
    assert oil == pytest.approx(p_energy * factor)

    path = np.array([1.0, 1.2, 0.85, 2.0])
    oils = np.array([commodity_price(p, carry=carry, dt=DT_MONTH) for p in path])
    assert oils == pytest.approx(path * factor)

    priced = commodity_prices(
        {"ENERGY": p_energy, "MATERIALS": 0.5, "AGRIFOOD": 2.0},
        carry=carry,
        dt=DT_MONTH,
    )
    assert COMMODITY_SECTORS["OIL"] == "ENERGY"
    assert priced["OIL"] == pytest.approx(oil)
    assert priced["METALS"] == pytest.approx(0.5 * factor)
    assert priced["GRAINS"] == pytest.approx(2.0 * factor)

    assert CARRY == 0.0
    assert commodity_price(p_energy) == pytest.approx(p_energy * (1.0 + CARRY * DT_DAY))


def test_threefold_cost_push_transmitted_without_cap() -> None:
    # R7 adds z_cost to log price. A 3× event is z = ln 3; no commodity level cap.
    p0 = 1.0
    z_cost = float(np.log(3.0))
    p_energy = p0 * np.exp(z_cost)
    carry = 0.06
    oil = commodity_price(p_energy, carry=carry, dt=DT_MONTH)
    assert oil == pytest.approx(3.0 * p0 * (1.0 + carry * DT_MONTH))

    # AGENTS rule 7: shocks run 3–5×; the level is never clipped.
    assert commodity_price(5.0 * p0, carry=0.0) == pytest.approx(5.0)

    p = np.ones((2, len(CODES)))
    p[:, CODES.index("ENERGY")] = 3.0
    ref = national_reference_price(p, "ENERGY", codes=CODES)
    assert ref == pytest.approx(3.0)
    assert commodity_price(ref, carry=0.0) == pytest.approx(3.0)
