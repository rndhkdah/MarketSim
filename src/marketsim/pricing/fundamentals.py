"""Fundamental value ``ln V = ln E^e + ln PE0 − D·Δρ + D·Δg_lr`` (T6.04 / §6.3)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from marketsim.core.config import Config

MTM_BOOK_WEIGHT = 0.25  # derive_betas: −disc_pass × 0.25 × bond_mtm_duration on β_r; applied to Δy10 here


def log_fundamental(
    ee: np.ndarray,
    pe0: np.ndarray,
    duration: np.ndarray,
    delta_rho: float | np.ndarray,
    delta_g_lr: float | np.ndarray = 0.0,
) -> np.ndarray:
    """``ln V_j`` (V in cr). ``ee`` is cr/year in the provider's convention."""
    return (
        np.log(np.asarray(ee, dtype=float))
        + np.log(np.asarray(pe0, dtype=float))
        - np.asarray(duration, dtype=float) * np.asarray(delta_rho, dtype=float)
        + np.asarray(duration, dtype=float) * np.asarray(delta_g_lr, dtype=float)
    )


def fundamental_values(
    ee: np.ndarray,
    pe0: np.ndarray,
    duration: np.ndarray,
    delta_rho: float | np.ndarray,
    delta_g_lr: float | np.ndarray = 0.0,
) -> np.ndarray:
    """``V_j`` (cr)."""
    return np.exp(log_fundamental(ee, pe0, duration, delta_rho, delta_g_lr))


def financials_dlnv(
    codes: Sequence[str],
    cfg: Config,
    *,
    r: float,
    r0: float,
    y10_t: float,
    y10_0: float,
) -> np.ndarray:
    """Ledger-not-modelled financials add-on to ``ln V`` (§6.3 / derive_betas).

    ``nim_rate_beta`` applies only when ``banks.mode: passthrough`` (no double
    count with Phase-2 full-bank NIM already in earnings).

    With ``bonds.pricing: market``, INSURANCE / BANKS mark-to-market emerges
    from holdings (§6.11): ``float_rate_beta`` and ``bond_mtm_duration`` are
    switched off. ``curve_beta`` and the passthrough NIM term stay.
    """
    out = np.zeros(len(codes), dtype=float)
    dr = float(r) - float(r0)
    d_curve = float(y10_t) - float(r) - (float(y10_0) - float(r0))
    d_y10 = float(y10_t) - float(y10_0)
    passthrough = cfg.dynamics is not None and cfg.dynamics.banks.mode == "passthrough"
    # §6.11: market marks replace the §6.3 float / duration overlays.
    market_marks = cfg.bonds is not None and cfg.bonds.pricing == "market"
    idx = {c: i for i, c in enumerate(codes)}
    for code, fin in cfg.sectors.financials.items():
        i = idx[code]
        nim = float(fin.nim_rate_beta) if passthrough else 0.0
        float_beta = 0.0 if market_marks else float(fin.float_rate_beta)
        mtm_dur = 0.0 if market_marks else float(fin.bond_mtm_duration)
        out[i] = (
            nim * dr
            + float_beta * dr
            + float(fin.curve_beta) * d_curve
            - MTM_BOOK_WEIGHT * mtm_dur * d_y10
        )
    return out
