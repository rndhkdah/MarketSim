"""Stock-flow-consistency assertions (§2.2)."""

from __future__ import annotations

import numpy as np

from marketsim.core.errors import SFCError
from marketsim.ledger.journal import Ledger
from marketsim.ledger.matrices import transaction_flow_matrix


def _financial_mask(ledger: Ledger) -> np.ndarray:
    return np.array([ledger.instruments.is_financial_id(i) for i in range(len(ledger.instruments))])


def net_financial_assets(ledger: Ledger, pos: np.ndarray | None = None) -> np.ndarray:
    """NFA per entity = sum of financial instrument positions."""
    p = ledger.pos if pos is None else pos
    return p[:, _financial_mask(ledger)].sum(axis=1)


def assert_consistent(ledger: Ledger, period: int | None = None, *, atol: float = 1e-8) -> None:
    """Checks (a)–(d). `period` is accepted for the card signature; the journal is the period."""
    del period
    _assert_tx_balanced(ledger, atol=atol)
    _assert_instrument_sums(ledger, atol=atol)
    tags, ents, tfm = transaction_flow_matrix(ledger)
    _assert_tfm_rows(tags, ents, tfm, atol=atol)
    _assert_nfa_identity(ledger, ents, tfm, atol=atol)


def _assert_tx_balanced(ledger: Ledger, *, atol: float) -> None:
    """(a) each Tx balanced — regroup columnar journal by (tick, tag) run."""
    n = len(ledger._ticks)
    if n == 0:
        return
    start = 0
    while start < n:
        tick = ledger._ticks[start]
        tag = ledger._tags[start]
        end = start + 1
        while end < n and ledger._ticks[end] == tick and ledger._tags[end] == tag:
            end += 1
        by_inst: dict[int, float] = {}
        for i in range(start, end):
            inst = ledger._inst[i]
            if ledger.instruments.is_financial_id(inst):
                by_inst[inst] = by_inst.get(inst, 0.0) + ledger._amt[i]
        for inst, total in by_inst.items():
            if abs(total) > atol:
                raise SFCError(
                    f"one-sided or unbalanced posting tag={tag!r} instrument={ledger.instruments.name(inst)} sum={total}",
                    tag=tag,
                    amount=total,
                )
        start = end


def _assert_instrument_sums(ledger: Ledger, *, atol: float) -> None:
    """(b) Σ_entities pos[:, k] = 0 for every financial instrument."""
    mask = _financial_mask(ledger)
    col = ledger.pos[:, mask].sum(axis=0)
    names = [n for n, f in zip(ledger.instruments.names, mask, strict=True) if f]
    for name, total in zip(names, col, strict=True):
        if abs(float(total)) > atol:
            raise SFCError(
                f"missing counter-entry: instrument {name} sums to {total} across entities",
                amount=float(total),
            )


def _assert_tfm_rows(
    tags: tuple[str, ...],
    ents: tuple[str, ...],
    tfm: np.ndarray,
    *,
    atol: float,
) -> None:
    """(c) each TFM row sums to zero across entities."""
    del ents
    row_sum = tfm.sum(axis=1)
    for tag, total in zip(tags, row_sum, strict=True):
        if abs(float(total)) > atol:
            raise SFCError(f"TFM row {tag!r} sums to {total}", tag=tag, amount=float(total))


def _assert_nfa_identity(
    ledger: Ledger,
    ents: tuple[str, ...],
    tfm: np.ndarray,
    *,
    atol: float,
) -> None:
    """(d) current + capital account (TFM column) = Δ NFA.

    Opening postings are included in the journal, so the TFM column equals
    the *level* of NFA when the book starts at zero — which is Δ NFA from 0.
    If `pos_open` is set, compare against the increment since that snapshot.
    """
    nfa = net_financial_assets(ledger)
    account = tfm.sum(axis=0)
    open_nfa = getattr(ledger, "pos_open", None)
    if open_nfa is not None:
        expected = nfa - net_financial_assets(ledger, open_nfa)
    else:
        expected = nfa
    for entity, got, want in zip(ents, account, expected, strict=True):
        if abs(float(got - want)) > atol:
            raise SFCError(
                f"stock changed without a flow: entity={entity} TFM={got} ΔNFA={want}",
                entity=entity,
                amount=float(got - want),
            )
