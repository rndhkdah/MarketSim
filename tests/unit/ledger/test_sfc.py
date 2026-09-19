from __future__ import annotations

import pytest

from marketsim.core.errors import SFCError
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _four_entity() -> Ledger:
    """HH, FIRM, BANKSYS, GOVT with opening deposits, consumption and wages."""
    led = Ledger.empty()
    for name in ("HH:0", "NPC:0:FIRM", "BANKSYS", "GOVT"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            [
                Entry("HH:0", "DEP", 80.0),
                Entry("GOVT", "DEP", 20.0),
                Entry("BANKSYS", "DEP", -100.0),
            ],
        )
    )
    # HH buys 10 from the firm (payment through bank deposits).
    led.post(
        Tx(
            1,
            "consumption",
            [Entry("HH:0", "DEP", -10.0), Entry("NPC:0:FIRM", "DEP", 10.0)],
        )
    )
    # Firm pays 4 wages.
    led.post(
        Tx(
            1,
            "wages",
            [Entry("NPC:0:FIRM", "DEP", -4.0), Entry("HH:0", "DEP", 4.0)],
        )
    )
    return led


def test_hand_built_four_entity_passes() -> None:
    led = _four_entity()
    assert_consistent(led, 1)
    assert led.position("HH:0", "DEP") == pytest.approx(74.0)
    assert led.position("NPC:0:FIRM", "DEP") == pytest.approx(6.0)


def test_one_sided_posting_named() -> None:
    led = _four_entity()
    # Bypass post() and append a one-sided journal line (and the matching stock).
    ei = led.entities.id("HH:0")
    ii = led.instruments.id("DEP")
    led.pos[ei, ii] += 5.0
    led._ticks.append(2)
    led._tags.append("consumption")
    led._ent.append(ei)
    led._inst.append(ii)
    led._amt.append(5.0)
    with pytest.raises(SFCError, match="one-sided|unbalanced"):
        assert_consistent(led, 1)


def test_missing_counter_entry_named() -> None:
    led = _four_entity()
    ei = led.entities.id("HH:0")
    ii = led.instruments.id("DEP")
    led.pos[ei, ii] += 3.0
    with pytest.raises(SFCError, match="missing counter-entry"):
        assert_consistent(led, 1)


def test_stock_changed_without_flow_named() -> None:
    led = _four_entity()
    # Keep instrument sums at 0 but skip the journal: HH +5, BANKSYS −5.
    led.pos[led.entities.id("HH:0"), led.instruments.id("DEP")] += 5.0
    led.pos[led.entities.id("BANKSYS"), led.instruments.id("DEP")] -= 5.0
    with pytest.raises(SFCError, match="stock changed without a flow"):
        assert_consistent(led, 1)
