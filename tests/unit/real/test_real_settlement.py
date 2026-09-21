from __future__ import annotations

import pytest

from marketsim.core.errors import SFCError
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.economy import RealEconomy
from marketsim.real.settlement import clip_firm_loans_bug


def test_sfc_120_months_shocked(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    for t in range(120):
        eco.sh["dem"] = 0.02 * (0.9 ** t) if t < 36 else 0.0
        eco.step_month()


def test_saving_identity_each_month(cfg, io) -> None:
    from marketsim.real.settlement import nfa_map

    eco = RealEconomy(cfg, io, pi_star=0.02, check_sfc=True)
    for t in range(24):
        eco.sh["dem"] = 0.01 if t == 3 else eco.sh["dem"] * 0.9
        before = nfa_map(eco.ledger)
        eco.step_month()
        after = nfa_map(eco.ledger)
        hh = after["HH:0"] - before["HH:0"]
        govt = after["GOVT"] - before["GOVT"]
        row = after["ROW"] - before["ROW"]
        firms = sum(after[k] - before[k] for k in after if k.startswith("NPC:0:"))
        assert hh == pytest.approx(-govt - firms - row, abs=1e-9)


def test_clip_debt_breaks_sfc(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    eco.step_month()
    clip_firm_loans_bug(eco.ledger, eco.real)
    with pytest.raises(SFCError):
        assert_consistent(eco.ledger, 1)
