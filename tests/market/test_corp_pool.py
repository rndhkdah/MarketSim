"""T6.29 — corporate bond pool, D14 menu, gate 16."""

from __future__ import annotations

import math

import pytest

from marketsim.firms.financing import (
    borrowing_rate,
    credit_limit,
    funding_menu,
    funding_split,
    pool_funding_rate,
)
from marketsim.firms.firm import FirmsFile
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.auction import Bid, clear_uniform
from marketsim.market.corp_pool import (
    BOND_DURATION_Y,
    ISSUER,
    LAMBDA_S,
    NOTE_DURATION_Y,
    POOL_DURATION_Y,
    POOL_INSTRUMENT,
    XI_MEAN_DAYS,
    CorpPool,
    cloan_face,
    holder_value,
    nav_per_unit,
    pool_units,
    settle_tap,
    weekly_tap,
    write_down_cloan,
    y_match,
)


def _entities(*names: str) -> Ledger:
    led = Ledger.empty()
    for name in names:
        led.register_entity(name)
    return led


def _open_pool(
    *,
    cloan: dict[str, float] | None = None,
    holders: dict[str, float] | None = None,
) -> Ledger:
    cloan = cloan or {"FIRM:aaa": 60.0, "FIRM:ccc": 40.0}
    holders = holders or {"MM": 70.0, "INSURANCE": 30.0}
    names = {ISSUER, "BANKSYS", *cloan, *holders}
    led = _entities(*sorted(names))
    entries: list[Entry] = []
    for firm, face in cloan.items():
        entries.extend((Entry(ISSUER, "CLOAN", face), Entry(firm, "CLOAN", -face)))
    for holder, face in holders.items():
        entries.extend((Entry(holder, POOL_INSTRUMENT, face), Entry(ISSUER, POOL_INSTRUMENT, -face)))
    led.post(Tx(0, "opening", tuple(entries), memo="pool opening"))
    return led


def test_y_match_interpolates_note_bond_duration() -> None:
    assert y_match(0.04, 0.04) == pytest.approx(0.04)
    weight = (POOL_DURATION_Y - NOTE_DURATION_Y) / (BOND_DURATION_Y - NOTE_DURATION_Y)
    assert y_match(0.03, 0.05) == pytest.approx(0.03 + weight * 0.02)
    assert NOTE_DURATION_Y < POOL_DURATION_Y < BOND_DURATION_Y


def test_identical_terms_different_ratings() -> None:
    """Two firms, AAA vs CCC, same tick → identical bank and pool rates (D14)."""
    ym = y_match(0.040, 0.050)
    s_t = 0.015
    menu = funding_menu(policy_rate=0.042, y_match=ym, spread=s_t)
    again = funding_menu(policy_rate=0.042, y_match=ym, spread=s_t)
    assert menu == again
    assert menu.bank_rate == pytest.approx(borrowing_rate(0.042, s_t))
    assert menu.pool_rate == pytest.approx(pool_funding_rate(ym, s_t))
    assert menu.bank_rate == pytest.approx(0.042 + s_t)
    assert menu.pool_rate == pytest.approx(ym + s_t)
    cfg = FirmsFile()
    aaa_cap = credit_limit(ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="AAA", cfg=cfg)
    ccc_cap = credit_limit(ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="CCC", cfg=cfg)
    assert aaa_cap != ccc_cap
    pool_face, bank_face = funding_split(80.0, 0.5)
    assert pool_face == pytest.approx(40.0)
    assert bank_face == pytest.approx(40.0)


def test_nav_accounting_through_default() -> None:
    led = _open_pool()
    assert nav_per_unit(led) == pytest.approx(1.0)
    assert pool_units(led) == pytest.approx(100.0)
    assert cloan_face(led) == pytest.approx(100.0)
    write_down_cloan(led, "FIRM:aaa", 25.0, tick=1)
    assert nav_per_unit(led) == pytest.approx(0.75)
    assert holder_value(led, "MM") == pytest.approx(70.0 * 0.75)
    assert holder_value(led, "INSURANCE") == pytest.approx(30.0 * 0.75)
    assert holder_value(led, "MM") + holder_value(led, "INSURANCE") == pytest.approx(cloan_face(led))
    assert led.position("FIRM:aaa", "CLOAN") == pytest.approx(-35.0)
    assert led.position(ISSUER, "CLOAN") == pytest.approx(75.0)
    assert led.position(ISSUER, POOL_INSTRUMENT) == pytest.approx(-100.0)
    assert_consistent(led)
    assert led.position("MM", POOL_INSTRUMENT) + led.position("INSURANCE", POOL_INSTRUMENT) + led.position(
        ISSUER, POOL_INSTRUMENT
    ) == pytest.approx(0.0)
    assert led.position(ISSUER, "CLOAN") + led.position("FIRM:aaa", "CLOAN") + led.position(
        "FIRM:ccc", "CLOAN"
    ) == pytest.approx(0.0)


def test_gate16_default_raises_all_firm_rates_equally() -> None:
    led = _open_pool()
    ym = y_match(0.042, 0.042)
    s_t = 0.015
    pool = CorpPool()
    before = pool.terms(policy_rate=0.042, y_match=ym, s_t=s_t)
    write_down_cloan(led, "FIRM:aaa", 20.0, tick=2)
    nav = nav_per_unit(led)
    assert nav < 1.0
    for _ in range(XI_MEAN_DAYS):
        pool.mark_day(xi=0.0, nav=nav, y_match=ym, s_t=s_t)
    after_aaa = pool.terms(policy_rate=0.042, y_match=ym, s_t=s_t)
    after_ccc = pool.terms(policy_rate=0.042, y_match=ym, s_t=s_t)
    assert after_aaa == after_ccc
    assert after_aaa.bank_rate > before.bank_rate
    assert after_aaa.pool_rate > before.pool_rate
    assert after_aaa.bank_rate - before.bank_rate == pytest.approx(after_aaa.pool_rate - before.pool_rate)
    assert after_aaa.spread == pytest.approx(after_ccc.spread)


def test_tap_issuance_settles_sfc() -> None:
    led = _entities(ISSUER, "MM", "BANKSYS", "alice", "FIRM:a", "FIRM:b")
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("alice", "DEP", 50.0),
                Entry("MM", "DEP", 200.0),
                Entry("BANKSYS", "DEP", -250.0),
            ),
        )
    )
    size = 30.0
    result = weekly_tap(
        led,
        size,
        (Bid("alice", 0.044, 10.0),),
        y_fair=0.042,
        q0=20.0,
        tick=5,
        borrowers={"FIRM:a": 10.0, "FIRM:b": 20.0},
    )
    assert result.filled == pytest.approx(size)
    assert result.agent_fill["alice"] == pytest.approx(10.0)
    assert led.position("alice", POOL_INSTRUMENT) == pytest.approx(10.0)
    assert led.position("MM", POOL_INSTRUMENT) == pytest.approx(result.npc_fill)
    assert led.position(ISSUER, POOL_INSTRUMENT) == pytest.approx(-size)
    assert led.position("FIRM:a", "CLOAN") == pytest.approx(-10.0)
    assert led.position("FIRM:b", "CLOAN") == pytest.approx(-20.0)
    assert led.position(ISSUER, "CLOAN") == pytest.approx(size)
    assert led.position(ISSUER, "DEP") == pytest.approx(0.0)
    assert_consistent(led)
    assert led.position("alice", POOL_INSTRUMENT) + led.position("MM", POOL_INSTRUMENT) + led.position(
        ISSUER, POOL_INSTRUMENT
    ) == pytest.approx(0.0)
    assert led.position(ISSUER, "CLOAN") + led.position("FIRM:a", "CLOAN") + led.position(
        "FIRM:b", "CLOAN"
    ) == pytest.approx(0.0)


def test_tap_uses_clear_uniform() -> None:
    size, y_fair, q0 = 12.0, 0.042, 8.0
    bids = (Bid("alice", 0.043, 4.0),)
    direct = clear_uniform(size, bids, y_fair=y_fair, q0=q0)
    led = _entities(ISSUER, "MM", "BANKSYS", "alice")
    led.post(
        Tx(
            0,
            "opening",
            (Entry("alice", "DEP", 20.0), Entry("MM", "DEP", 20.0), Entry("BANKSYS", "DEP", -40.0)),
        )
    )
    result = clear_uniform(size, bids, y_fair=y_fair, q0=q0)
    settle_tap(led, result, tick=1)
    assert result.stop_out == pytest.approx(direct.stop_out)
    assert result.npc_fill == pytest.approx(direct.npc_fill)
    assert_consistent(led)


def test_selling_pressure_raises_every_firm_rate_equally() -> None:
    ym = y_match(0.041, 0.048)
    s_t = 0.015
    z_risk = 0.02
    calm = CorpPool()
    dump = CorpPool()
    base = calm.terms(policy_rate=0.04, y_match=ym, s_t=s_t, z_risk=z_risk)
    for _ in range(XI_MEAN_DAYS):
        dump.mark_day(xi=-0.05, nav=1.0, y_match=ym, s_t=s_t, z_risk=z_risk)
    aaa = dump.terms(policy_rate=0.04, y_match=ym, s_t=s_t, z_risk=z_risk)
    ccc = dump.terms(policy_rate=0.04, y_match=ym, s_t=s_t, z_risk=z_risk)
    assert aaa == ccc
    assert aaa.bank_rate > base.bank_rate
    assert aaa.pool_rate > base.pool_rate
    assert aaa.bank_rate - base.bank_rate == pytest.approx(aaa.pool_rate - base.pool_rate)
    assert aaa.spread - base.spread == pytest.approx(aaa.bank_rate - base.bank_rate)
    assert dump.borrowing_spread(s_t, z_risk) > s_t + LAMBDA_S * z_risk
    assert math.exp(-0.05) < 1.0
