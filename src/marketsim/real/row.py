"""Rest of world: exports, non-competing imports, trade-balance postings (R2, R9)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.real.steady_state import RealBaseline


def exports(
    x0: np.ndarray,
    p: np.ndarray,
    p_imp: float,
    eps_x: float,
    z_row: float = 0.0,
) -> np.ndarray:
    """Real export demand (cr/month): ``X0 · exp(z_row) · (p/p_imp)^{−ε_x}``."""
    rel = np.asarray(p, dtype=float) / p_imp
    return np.asarray(x0, dtype=float) * np.exp(z_row) * np.power(rel, -eps_x)


def import_bill(p_imp: float, m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Nominal import bill by sector (cr/month)."""
    return p_imp * np.asarray(m, dtype=float) * np.asarray(x, dtype=float)


def trade_balance(p: np.ndarray, ex_real: np.ndarray, imp_nom: np.ndarray) -> float:
    """Exports minus imports, nominal cr/month."""
    return float((np.asarray(p) * ex_real).sum() - imp_nom.sum())


def post_trade(
    ledger: Ledger,
    real: RealBaseline,
    *,
    p: np.ndarray,
    ex_real: np.ndarray,
    imp_nom: np.ndarray,
    tick: int,
    region: int = 0,
) -> float:
    """Post export and import tags. Returns the nominal trade balance."""
    for i, code in enumerate(real.codes):
        firm = f"NPC:{region}:{code}"
        ex_n = float(p[i] * ex_real[i])
        if abs(ex_n) > 1e-15:
            ledger.post(
                Tx(
                    tick,
                    "exports",
                    (Entry("ROW", "DEP", -ex_n), Entry(firm, "DEP", ex_n)),
                )
            )
        im_n = float(imp_nom[i])
        if abs(im_n) > 1e-15:
            ledger.post(
                Tx(
                    tick,
                    "imports",
                    (Entry(firm, "DEP", -im_n), Entry("ROW", "DEP", im_n)),
                )
            )
    return trade_balance(p, ex_real, imp_nom)


def row_nfa(ledger: Ledger) -> float:
    """ROW net financial assets (sum of financial columns)."""
    from marketsim.ledger.sfc import net_financial_assets

    return float(net_financial_assets(ledger)[ledger.entities.id("ROW")])


def row_from_config(cfg: Config, real: RealBaseline) -> tuple[np.ndarray, float]:
    """Baseline real exports and ε_x."""
    assert cfg.dynamics is not None
    return real.flat(real.X0), cfg.dynamics.row.export_price_elasticity
