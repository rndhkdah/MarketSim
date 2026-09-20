"""T6.13 — margin arithmetic, borrow fees, loss waterfall, gate 6."""

from __future__ import annotations

import math

import pytest

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FIRM_PREFIX, MarginCfg
from marketsim.market.margin import (
    BROKER_SPREAD,
    DAYS_PER_YEAR,
    MarginBook,
    account_equity,
    haircut,
    required_margin,
    requirement,
)
from marketsim.market.mm import MM_ACCOUNT
from marketsim.market.settlement import settle_fill
from marketsim.market.shorting import (
    FEE0,
    KAPPA,
    StockLoanBook,
    borrow_fee_annual,
    is_lendable,
    utilisation,
)
from marketsim.market.venue import Fill, Side

SYM = "EQ:NPC:AUTOS"
OIL = "OIL"
FIRM = f"{FIRM_PREFIX}acme"
P0 = 10.0


def _ledger(*entities: str) -> Ledger:
    led = Ledger.empty(debug_journal=True)
    for name in entities:
        led.register_entity(name)
    return led


def _fill(taker: str, qty: int, price: float, *, tick: int = 1, oid: int = 1) -> Fill:
    return Fill(
        symbol=SYM,
        price=price,
        qty=qty,
        maker=MM_ACCOUNT,
        taker=taker,
        maker_order_id=0,
        taker_order_id=oid,
        notional=price * qty,
        tick=tick,
    )


def _open_long(
    led: Ledger,
    book: MarginBook,
    account: str,
    qty: int,
    *,
    price: float = P0,
    cash: float,
    loan: float,
    tick: int = 0,
    oid: int = 1,
) -> None:
    led.post(Tx(tick, "opening", (Entry(account, "DEP", cash), Entry("BANKSYS", "DEP", -cash))))
    if loan > 0.0:
        book.draw(led, account, loan, tick)
    settle_fill(led, _fill(account, qty, price, tick=tick, oid=oid), Side.BUY)
    book.set_cost(account, SYM, price)
    book.track(account)


def test_margin_arithmetic(cfg: Config) -> None:
    assert cfg.markets is not None
    mcfg = cfg.markets.margin
    assert mcfg.equity_initial == pytest.approx(0.50)
    assert mcfg.equity_maintenance == pytest.approx(0.25)
    assert mcfg.commodity_initial == pytest.approx(0.10)
    assert mcfg.commodity_maintenance == pytest.approx(0.07)
    assert haircut(SYM, mcfg, initial=True) == pytest.approx(0.50)
    assert haircut(SYM, mcfg, initial=False) == pytest.approx(0.25)
    assert haircut(OIL, mcfg, initial=True) == pytest.approx(0.10)
    assert haircut(OIL, mcfg, initial=False) == pytest.approx(0.07)

    qty, px = 100.0, P0
    init = required_margin(qty, px, mcfg.equity_initial)
    maint = required_margin(qty, px, mcfg.equity_maintenance)
    assert init == pytest.approx(500.0)
    assert maint == pytest.approx(250.0)

    led = _ledger("alice", MM_ACCOUNT, "BANKSYS")
    book = MarginBook(mcfg)
    _open_long(led, book, "alice", 100, cash=500.0, loan=500.0)
    assert_consistent(led)
    assert led.position("alice", "LOAN") == pytest.approx(-500.0)
    assert led.position("BANKSYS", "LOAN") == pytest.approx(500.0)
    assert led.position(MM_ACCOUNT, "LOAN") == pytest.approx(0.0)
    assert book.debit["alice"] == pytest.approx(500.0)

    prices = {SYM: P0}
    eq = account_equity(led, "alice", prices)
    assert eq == pytest.approx(500.0)
    assert requirement(led, "alice", prices, mcfg, initial=True) == pytest.approx(500.0)
    assert requirement(led, "alice", prices, mcfg, initial=False) == pytest.approx(250.0)

    # p = 6: MV 600, equity 100, maintenance 150 → breach. Restore initial: sell 400 cr.
    shocked = {SYM: 6.0}
    eq6 = account_equity(led, "alice", shocked)
    maint6 = requirement(led, "alice", shocked, mcfg, initial=False)
    init6 = requirement(led, "alice", shocked, mcfg, initial=True)
    assert eq6 == pytest.approx(100.0)
    assert maint6 == pytest.approx(150.0)
    assert eq6 < maint6
    # S ≥ G − E/λ = 600 − 100 / 0.50 = 400 cr → 400 / 6 units.
    need_cr = init6 - eq6
    assert need_cr == pytest.approx(200.0)
    assert need_cr / mcfg.equity_initial == pytest.approx(400.0)

    snaps = book.check(led, shocked)
    alice = next(s for s in snaps if s.account == "alice")
    assert alice.breached
    assert book.pending
    assert book.pending[0].side is Side.SELL
    assert book.pending[0].qty == math.ceil((400.0 / 6.0) - 1e-12)

    # Check queues only — debit / shares unchanged until the next tick.
    assert book.debit["alice"] == pytest.approx(500.0)
    assert led.position("alice", SYM) == pytest.approx(100.0)
    taken = book.take_forced_orders()
    assert taken and not book.pending


def test_borrow_fee_accrues() -> None:
    assert borrow_fee_annual(0.8) > borrow_fee_annual(0.1)
    assert borrow_fee_annual(0.0) == pytest.approx(FEE0)
    assert is_lendable(SYM)
    assert is_lendable(OIL)
    assert not is_lendable(FIRM)

    led = _ledger("alice", "bob", "NPC:0:AUTOS", MM_ACCOUNT, "BANKSYS")
    led.register_instrument(SYM, financial=True)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("bob", SYM, 90.0),
                Entry(MM_ACCOUNT, SYM, 20.0),
                Entry("alice", SYM, -10.0),
                Entry("NPC:0:AUTOS", SYM, -100.0),
                Entry("alice", "DEP", 50.0),
                Entry("BANKSYS", "DEP", -50.0),
            ),
        )
    )
    assert_consistent(led)
    loans = StockLoanBook()
    assert not loans.locate(FIRM, 1.0, led)
    assert loans.locate(SYM, 5.0, led)
    u = utilisation(led, SYM)
    assert u == pytest.approx(10.0 / 110.0)

    px = {SYM: P0}
    paid = 0.0
    for day in range(3):
        before = led.position("alice", "DEP")
        txs = loans.accrue(led, px, tick=day + 1)
        assert txs
        assert_consistent(led)
        after = led.position("alice", "DEP")
        assert after < before
        paid += before - after
    daily = 10.0 * P0 * borrow_fee_annual(u, fee0=FEE0, kappa=KAPPA) / DAYS_PER_YEAR
    assert paid == pytest.approx(3.0 * daily)
    assert led.position(MM_ACCOUNT, "DEP") == pytest.approx(paid)

    # Tighter utilisation → larger daily fee (same short, fewer longs).
    led2 = _ledger("alice", MM_ACCOUNT)
    led2.register_instrument(SYM, financial=True)
    led2.post(
        Tx(
            0,
            "opening",
            (
                Entry(MM_ACCOUNT, SYM, 20.0),
                Entry("alice", SYM, -20.0),
                Entry("alice", "DEP", 50.0),
                Entry(MM_ACCOUNT, "DEP", -50.0),
            ),
        )
    )
    u_hi = utilisation(led2, SYM)
    assert u_hi > u
    StockLoanBook().accrue(led2, px, tick=1)
    fee_hi = 50.0 - led2.position("alice", "DEP")
    assert fee_hi > daily


def test_loss_waterfall_reaches_bank_capital() -> None:
    led = _ledger("dave", MM_ACCOUNT, "BANKSYS")
    book = MarginBook(MarginCfg(), broker_cap=40.0)
    # 10 units at 100 cr: notional 1000, 100 cash + 900 loan (equity 100 cr).
    _open_long(led, book, "dave", 10, cash=100.0, loan=900.0, price=100.0)
    assert_consistent(led)
    nfa0 = float(net_financial_assets(led)[led.entities.id("BANKSYS")])

    # Gap down to 20 cr/unit: MV 200, equity −700. Flatten at the mark.
    settle_fill(led, _fill("dave", 10, 20.0, tick=2, oid=2), Side.SELL)
    book.repay(led, "dave", 200.0, tick=2)
    assert led.position("dave", SYM) == pytest.approx(0.0)
    assert book.debit["dave"] == pytest.approx(700.0)
    eq = account_equity(led, "dave", {SYM: 20.0})
    assert eq == pytest.approx(-700.0)

    res = book.waterfall(led, "dave", tick=3)
    assert res.broker_loss == pytest.approx(40.0)
    assert res.bank_writeoff == pytest.approx(660.0)
    assert book.debit["dave"] == pytest.approx(0.0)
    assert led.position("dave", "LOAN") == pytest.approx(0.0)
    assert_consistent(led)

    nfa1 = float(net_financial_assets(led)[led.entities.id("BANKSYS")])
    assert nfa1 == pytest.approx(nfa0 - 660.0)
    # MM sold 10 @ 10 (+1000), bought 10 @ 20 (−200), gifted the 40 cr cap.
    assert led.position(MM_ACCOUNT, "DEP") == pytest.approx(1000.0 - 200.0 - 40.0)


def test_gate6_liquidation_cascade(cfg: Config) -> None:
    """Gate 6: bounded monotone deleveraging, SFC, losses follow the waterfall."""
    assert cfg.markets is not None
    mcfg = cfg.markets.margin
    led = _ledger("alice", "bob", "carol", "dave", MM_ACCOUNT, "BANKSYS")
    book = MarginBook(mcfg, broker_spread=BROKER_SPREAD, broker_cap=40.0)
    _open_long(led, book, "alice", 100, cash=500.0, loan=500.0, oid=1)
    _open_long(led, book, "bob", 80, cash=400.0, loan=400.0, oid=2)
    _open_long(led, book, "carol", 60, cash=300.0, loan=300.0, oid=3)
    # Thin equity: already a large loser after the gap, residual hits the bank.
    _open_long(led, book, "dave", 50, cash=20.0, loan=480.0, oid=4)
    assert_consistent(led)
    nfa0 = float(net_financial_assets(led)[led.entities.id("BANKSYS")])

    prices: dict[str, float] = {SYM: 6.0}
    kernel = ImpactKernel()
    adv = {SYM: 100.0}
    sigma = {SYM: 0.05}
    agents = ("alice", "bob", "carol", "dave")

    def gross() -> float:
        return sum(max(led.position(a, SYM), 0.0) for a in agents)

    def leverage() -> float:
        return sum(book.debit.values())

    g_prev, d_prev = gross(), leverage()
    g0, d0 = g_prev, d_prev
    waves = 0
    writeoffs = 0.0
    for wave in range(8):
        book.accrue_interest(led, policy_rate=0.0, tick=10 + wave)
        snaps = book.check(led, prices)
        assert_consistent(led)
        if not book.pending:
            break
        assert any(s.breached for s in snaps)
        # Largest loser first: dave's mark-to-market is the most negative.
        assert book.pending[0].account == "dave" or wave > 0
        book.execute_forced(
            led,
            prices,
            {SYM: kernel},
            adv=adv,
            sigma=sigma,
            spread=0.0,
            tick=11 + wave,
        )
        resolved = book.resolve_deficits(led, prices, tick=11 + wave)
        writeoffs += sum(r.bank_writeoff for r in resolved)
        assert_consistent(led)
        g, d = gross(), leverage()
        assert g <= g_prev + 1e-9
        assert d <= d_prev + 1e-9
        assert math.isfinite(prices[SYM]) and prices[SYM] > 0.0
        g_prev, d_prev = g, d
        waves += 1

    assert waves >= 1
    assert g_prev < g0
    assert d_prev < d0
    assert prices[SYM] < 6.0  # forced sales executed with impact
    nfa1 = float(net_financial_assets(led)[led.entities.id("BANKSYS")])
    assert writeoffs > 0.0
    assert nfa1 == pytest.approx(nfa0 - writeoffs, abs=1e-8)
    assert book.debit["dave"] == pytest.approx(0.0)
    assert_consistent(led)
