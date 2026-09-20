"""T8.02 — supply contracts: shortage priority and ledger cash flows."""

from __future__ import annotations

import pytest

from marketsim.firms.accounts import firm_entity
from marketsim.firms.contracts import (
    SETTLE_TAG,
    ContractBook,
    ForwardContract,
    open_forward,
)
from marketsim.firms.founding import npc_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent

_REGION = "0"
_SECTOR = "ENERGY"
_WANTED = 10.0  # units; both buyers order the same qty
_BASE_AVAIL = 20.0  # units before the supply event
_CUT_AVAIL = 10.0  # units after the supply cut
_PRICE = 1.25  # index, posted as cr/unit
_OPENING_CASH = 100.0  # cr


def _seller() -> str:
    return npc_entity(_REGION, _SECTOR)


def _book_one_contract(buyer: str, seller: str, *, months: int = 3) -> ContractBook:
    book = ContractBook()
    book.add(
        open_forward(
            buyer_id=buyer,
            seller_id=seller,
            region=_REGION,
            sector=_SECTOR,
            qty=_WANTED,
            price=_PRICE,
            months=months,
        )
    )
    return book


def test_contracted_buyer_served_first_during_supply_event() -> None:
    """A supply cut that leaves only one order's worth of goods: contract ≥ spot fill."""
    seller = _seller()
    contracted = firm_entity("steel")
    spot = firm_entity("spotco")
    spot_demand = {seller: {spot: _WANTED}}

    full = _book_one_contract(contracted, seller)
    full_res = full.settle_month({seller: _BASE_AVAIL}, spot_demand)
    assert full_res.fill_rate(contracted, seller, contracted=True) == pytest.approx(1.0)
    assert full_res.fill_rate(spot, seller, contracted=False) == pytest.approx(1.0)

    cut = _book_one_contract(contracted, seller)
    cut_res = cut.settle_month({seller: _CUT_AVAIL}, spot_demand)
    c_rate = cut_res.fill_rate(contracted, seller, contracted=True)
    s_rate = cut_res.fill_rate(spot, seller, contracted=False)
    assert c_rate >= s_rate
    assert cut_res.contract_fill[(contracted, seller)] == pytest.approx(_CUT_AVAIL)
    assert cut_res.spot_fill[(spot, seller)] == pytest.approx(0.0)


def test_leftover_avail_goes_to_spot_prorata() -> None:
    seller = _seller()
    contracted = firm_entity("steel")
    a = firm_entity("a")
    b = firm_entity("b")
    leftover_avail = 16.0  # units; 10 contracted + 6 leftover
    book = _book_one_contract(contracted, seller)
    res = book.settle_month(
        {seller: leftover_avail},
        {seller: {a: _WANTED, b: _WANTED}},
    )
    assert res.contract_fill[(contracted, seller)] == pytest.approx(_WANTED)
    assert res.spot_fill[(a, seller)] == pytest.approx(3.0)
    assert res.spot_fill[(b, seller)] == pytest.approx(3.0)
    assert res.fill_rate(contracted, seller, contracted=True) >= res.fill_rate(a, seller, contracted=False)


def test_contract_cash_flows_on_ledger() -> None:
    seller = _seller()
    buyer = firm_entity("acme")
    led = Ledger.empty()
    led.register_entity(buyer)
    led.register_entity(seller)
    led.register_entity("BANKSYS")
    led.post(
        Tx(0, "opening", (Entry(buyer, "DEP", _OPENING_CASH), Entry("BANKSYS", "DEP", -_OPENING_CASH)))
    )
    book = _book_one_contract(buyer, seller, months=1)
    res = book.settle_month({seller: _WANTED}, {}, ledger=led, tick=1)
    assert_consistent(led)
    cash = _PRICE * _WANTED
    assert res.cash_cr == pytest.approx(cash)
    assert led.position(buyer, "DEP") == pytest.approx(_OPENING_CASH - cash)
    assert led.position(seller, "DEP") == pytest.approx(cash)
    assert SETTLE_TAG in led.flow_tags
    assert SETTLE_TAG in led._tags


def test_remaining_qty_and_tenor_decay() -> None:
    seller = _seller()
    buyer = firm_entity("acme")
    book = _book_one_contract(buyer, seller, months=3)
    book.settle_month({seller: _WANTED}, {})
    assert len(book) == 1
    left = book.contracts[0]
    assert left.remaining_months == 2
    assert left.remaining_qty == pytest.approx(_WANTED * 2)
    book.settle_month({seller: _WANTED}, {})
    book.settle_month({seller: _WANTED}, {})
    assert len(book) == 0


def test_state_roundtrip() -> None:
    seller = _seller()
    c = ForwardContract(
        buyer_id=firm_entity("acme"),
        seller_id=seller,
        region=_REGION,
        sector=_SECTOR,
        qty=_WANTED,
        price=_PRICE,
        remaining_qty=_WANTED * 2,
        remaining_months=2,
    )
    book = ContractBook()
    book.add(c)
    restored = ContractBook.from_state(book.to_state())
    assert restored.to_state() == book.to_state()
