"""Opening balance sheet for `banks.mode: passthrough` (§2.2)."""

from __future__ import annotations

from pathlib import Path

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.steady_state import FinancialBaseline, RealBaseline

GOVT_MIX = (("GB_BILL", 0.20), ("GB_NOTE", 0.40), ("GB_BOND", 0.40))


def build_phase2_ledger(cfg: Config, real: RealBaseline, n_regions: int = 1) -> Ledger:
    path = Path(cfg.config_dir) / "ledger.yaml"
    return Ledger.from_yaml(path, codes=real.codes, n_regions=n_regions)


def post_opening_passthrough(
    ledger: Ledger,
    real: RealBaseline,
    fin: FinancialBaseline,
    *,
    region: int = 0,
) -> None:
    """HH holds firm loans and all government bonds; firms and GOVT are the counterparties."""
    hh = f"HH:{region}"
    entries: list[Entry] = []
    for i, code in enumerate(real.codes):
        firm = f"NPC:{region}:{code}"
        amt = float(fin.debt[i])
        if abs(amt) < 1e-15:
            continue
        entries.append(Entry(hh, "LOAN", amt))
        entries.append(Entry(firm, "LOAN", -amt))
    for inst, share in GOVT_MIX:
        amt = float(fin.B * share)
        entries.append(Entry(hh, inst, amt))
        entries.append(Entry("GOVT", inst, -amt))
    ledger.post(Tx(0, "opening", entries, memo="passthrough opening"))


def opening_wealth(ledger: Ledger, hh: str = "HH:0") -> float:
    """Consumption-relevant W = DEP + government bonds (+ passthrough LOAN)."""
    w = ledger.position(hh, "DEP")
    for inst in ("GB_BILL", "GB_NOTE", "GB_BOND", "LOAN"):
        if inst in ledger.instruments:
            w += ledger.position(hh, inst)
    return float(w)


def open_passthrough_books(cfg: Config, real: RealBaseline, fin: FinancialBaseline) -> Ledger:
    led = build_phase2_ledger(cfg, real)
    post_opening_passthrough(led, real, fin)
    assert_consistent(led, 0)
    return led
