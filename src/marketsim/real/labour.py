"""Labour demand and the Phillips wage rule (R8)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config


def target_employment(
    ell: np.ndarray,
    fc: np.ndarray,
    k: np.ndarray,
    ustar: np.ndarray,
    plan: np.ndarray,
    z_sup: np.ndarray,
) -> np.ndarray:
    """``n*`` in wage-units of labour (cr/month at w=1)."""
    return ell * np.exp(-z_sup) * (fc * k * ustar + (1.0 - fc) * plan)


def adjust_employment(
    n: np.ndarray,
    n_star: np.ndarray,
    tau_hire: float,
    tau_fire: float,
    lf: float,
    lf_cap: float,
) -> np.ndarray:
    """Asymmetric partial adjustment, then ``Σ n ≤ lf_cap · LF``.

    ``n`` may be ``(S,)`` (national) or ``(R, S)`` with ``lf`` of shape ``(R,)``.
    The 1-d path is the Phase-2 formula and stays bit-identical.
    """
    tau = np.where(n_star > n, tau_hire, tau_fire)
    n_new = n + (n_star - n) / tau
    if n_new.ndim == 1:
        total = float(n_new.sum())
        cap = lf_cap * lf
        if total > cap:
            n_new = n_new * (cap / total)
        return n_new
    lf_r = np.asarray(lf, dtype=float)
    total = n_new.sum(axis=-1)
    cap = lf_cap * lf_r
    scale = np.ones_like(total)
    over = total > cap
    scale[over] = cap[over] / total[over]
    return n_new * scale[..., None]


def unemployment(n: np.ndarray, lf: float | np.ndarray) -> float | np.ndarray:
    """Unemployment rate (share of LF). ``(R, S)`` / ``(R,)`` returns ``(R,)``."""
    n_arr = np.asarray(n, dtype=float)
    if n_arr.ndim == 1:
        return 1.0 - float(n_arr.sum()) / float(lf)
    lf_r = np.asarray(lf, dtype=float)
    return 1.0 - n_arr.sum(axis=-1) / lf_r


def wage_growth(
    pi_e: float,
    u: float,
    u_star: float,
    phillips_slope: float,
    stickiness_q: float,
) -> float:
    """Annual wage growth (decimal). Downward moves are divided by ``stickiness_q``."""
    dw = pi_e + phillips_slope * (u_star - u)
    if dw < 0.0:
        dw /= stickiness_q
    return float(dw)


def step_labour(
    n: np.ndarray,
    w: float,
    *,
    ell: np.ndarray,
    fc: np.ndarray,
    k: np.ndarray,
    ustar: np.ndarray,
    plan: np.ndarray,
    z_sup: np.ndarray,
    lf: float,
    cfg: Config,
    pi_e: float,
) -> tuple[np.ndarray, float, float]:
    """One labour/wage step. Returns ``(n, w, U)``."""
    assert cfg.dynamics is not None
    lab = cfg.dynamics.labour
    n_star = target_employment(ell, fc, k, ustar, plan, z_sup)
    n_new = adjust_employment(n, n_star, lab.tau_hire_m, lab.tau_fire_m, lf, lab.lf_cap)
    u = unemployment(n_new, lf)
    dw = wage_growth(pi_e, u, lab.u_star, cfg.edges.labour.phillips_slope, cfg.edges.labour.wage_stickiness_q)
    w_new = w * float(np.exp(dw / 12.0))
    return n_new, w_new, u
