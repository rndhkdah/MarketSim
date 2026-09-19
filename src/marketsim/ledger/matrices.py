"""Balance-sheet and transaction-flow matrices."""

from __future__ import annotations

import numpy as np

from marketsim.ledger.journal import Ledger


def balance_sheet_matrix(ledger: Ledger) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Return (entities, instruments, pos). pos[e, k] is +asset / −liability."""
    return ledger.entities.names, ledger.instruments.names, ledger.pos.copy()


def transaction_flow_matrix(
    ledger: Ledger,
    *,
    period_slice: slice | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """TFM[tag, entity] = sum of financial-instrument amounts posted with that tag.

    Each row sums to zero when every Tx was balanced.
    """
    tags = ledger.flow_tags
    ents = ledger.entities.names
    tag_index = {t: i for i, t in enumerate(tags)}
    tfm = np.zeros((len(tags), len(ents)))
    ticks = ledger._ticks
    if period_slice is None:
        indices = range(len(ticks))
    else:
        indices = range(*period_slice.indices(len(ticks)))
    for i in indices:
        tag = ledger._tags[i]
        inst = ledger._inst[i]
        if not ledger.instruments.is_financial_id(inst):
            continue
        ti = tag_index.get(tag)
        if ti is None:
            continue
        tfm[ti, ledger._ent[i]] += ledger._amt[i]
    return tags, ents, tfm
