"""Closed-form real-side steady state (§2.3). Shape is (R, S) with R = 1 in Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.layer1.io import IOTable


@dataclass
class RealBaseline:
    """Real-side seed. Sector arrays are (R, S); S_in0 is (R, S, S) [buyer-region, i, j]."""

    codes: tuple[str, ...]
    R: int
    S: int
    scale: float
    x0: np.ndarray
    s0: np.ndarray
    m: np.ndarray
    va: np.ndarray
    ell: np.ndarray
    gos: np.ndarray
    markup: np.ndarray
    K0: np.ndarray
    v: np.ndarray
    rho_K: float
    route_bus: np.ndarray
    res0: float
    I_bus0: float
    I0: float
    inv0: np.ndarray
    backlog0: np.ndarray
    S_in0: np.ndarray
    n0: np.ndarray
    LF: float
    cover: np.ndarray
    leak: np.ndarray
    tau_ob: np.ndarray
    tau_inv: np.ndarray
    is_stock: np.ndarray
    is_order: np.ndarray
    is_flow: np.ndarray
    d0: np.ndarray
    C0: np.ndarray
    I_fd: np.ndarray
    G0: np.ndarray
    X0: np.ndarray
    wages_pre_import: float
    wages_post_import: float
    va_pre_import: float
    va_post_import: float
    stor_in: np.ndarray
    crit: np.ndarray

    def flat(self, arr: np.ndarray) -> np.ndarray:
        """R=1 helper: drop the region axis."""
        return np.asarray(arr)[0] if arr.ndim >= 2 and arr.shape[0] == 1 else arr


def _vec(cfg: Config, key: str) -> np.ndarray:
    return np.array([getattr(cfg.sectors.params(c), key) for c in cfg.codes], dtype=float)


def mode_masks(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = cfg.dynamics.production.mode  # type: ignore[union-attr]
    is_stock = np.array([modes[c] == "stock" for c in cfg.codes])
    is_order = np.array([modes[c] == "order" for c in cfg.codes])
    is_flow = np.array([modes[c] == "flow" for c in cfg.codes])
    return is_stock, is_order, is_flow


def leak_cover_tau(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert cfg.dynamics is not None
    prod = cfg.dynamics.production
    is_stock, is_order, _ = mode_masks(cfg)
    invq = _vec(cfg, "inv_lag_q")
    cover = np.where(is_stock, prod.cover_scale * 3.0 * np.maximum(invq, 1.0), 0.0)
    leak = np.array([prod.leak_per_month.get(c, 0.0) for c in cfg.codes], dtype=float) * is_stock
    tau_inv = np.maximum(2.0, prod.tau_inv_mult * cover)
    tau_ob = np.where(is_order, prod.order_book_scale * 3.0 * np.maximum(invq, 1.0), 1.0)
    return leak, cover, tau_inv, tau_ob


def compute_real_baseline(
    io: IOTable,
    cfg: Config,
    *,
    scale: float | None = None,
    leak_override: np.ndarray | None = None,
) -> RealBaseline:
    """Closed-form real seed. Units: cr/month at baseline prices; R = 1."""
    assert cfg.dynamics is not None
    dyn = cfg.dynamics
    scale = float(cfg.world.scale * 100.0 if scale is None else scale)
    codes = io.codes
    s = io.n
    r_dim = 1
    A = io.A
    mu = io.mu
    is_stock, is_order, is_flow = mode_masks(cfg)
    leak, cover, tau_inv, tau_ob = leak_cover_tau(cfg)
    if leak_override is not None:
        leak = np.asarray(leak_override, dtype=float) * is_stock

    fd = io.final_demand
    wts = io.fd_weights
    C0 = wts["HOUSEHOLD"] * scale * fd["HOUSEHOLD"]
    I_fd = wts["INVESTMENT"] * scale * fd["INVESTMENT"]
    G0 = wts["GOVT"] * scale * fd["GOVT"]
    X0 = wts["EXPORTS"] * scale * fd["EXPORTS"]
    d0 = C0 + I_fd + G0 + X0

    lam = np.diag(1.0 + leak * cover)
    x0 = np.linalg.solve(np.eye(s) - lam @ A, lam @ d0)
    s0 = x0 / (1.0 + leak * cover)

    prior = dyn.row.import_prior
    m = np.array([prior.get(c, prior.get("default", 0.01)) for c in codes], dtype=float)
    mx = float((m * x0).sum())
    if mx <= 0:
        raise ValueError("import prior produced zero import bill")
    m = m * (float(X0.sum()) / mx)

    ws = np.array([cfg.edges.labour.wage_share[c] for c in codes], dtype=float)
    va_pre = 1.0 - mu
    wages_pre = float((ws * va_pre * x0).sum())
    va_pre_tot = float((va_pre * x0).sum())
    va = 1.0 - mu - m
    ell = ws * va
    gos = (1.0 - ws) * va
    wages_post = float((ws * va * x0).sum())
    va_post_tot = float((va * x0).sum())

    markup = 1.0 / (A.sum(axis=0) + ell + m)
    ustar = _vec(cfg, "util_target")
    k0 = x0 / ustar
    i0 = float(I_fd.sum())
    res0 = dyn.residential.share_of_investment * i0
    i_bus0 = (1.0 - dyn.residential.share_of_investment) * i0
    rho_k = dyn.capex.delta_annual * float((gos * x0).sum()) / i_bus0
    v = 12.0 * gos * ustar / rho_k

    bus = I_fd.copy()
    construct = codes.index("CONSTRUCT")
    bus[construct] -= res0
    route_bus = bus / bus.sum()

    inv0 = cover * s0
    backlog0 = np.where(is_order, (tau_ob - 1.0) * x0, 0.0)
    stor_in = is_stock[:, None] & (A > 0)
    s_in0 = np.where(stor_in, dyn.production.input_cover_m * A * x0[None, :], 0.0)
    n0 = ell * x0
    lf = float(n0.sum()) / (1.0 - dyn.labour.u_star)

    share = np.divide(A, mu[None, :], out=np.zeros_like(A), where=mu[None, :] > 1e-12)
    crit_sup = np.array([c in set(dyn.production.critical_input.suppliers) for c in codes])
    crit = (share >= dyn.production.critical_input.min_share) & crit_sup[:, None]

    def r1(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float)[None, ...]

    return RealBaseline(
        codes=codes,
        R=r_dim,
        S=s,
        scale=scale,
        x0=r1(x0),
        s0=r1(s0),
        m=r1(m),
        va=r1(va),
        ell=r1(ell),
        gos=r1(gos),
        markup=r1(markup),
        K0=r1(k0),
        v=r1(v),
        rho_K=float(rho_k),
        route_bus=r1(route_bus),
        res0=float(res0),
        I_bus0=float(i_bus0),
        I0=float(i0),
        inv0=r1(inv0),
        backlog0=r1(backlog0),
        S_in0=s_in0[None, ...],
        n0=r1(n0),
        LF=float(lf),
        cover=r1(cover),
        leak=r1(leak),
        tau_ob=r1(tau_ob),
        tau_inv=r1(tau_inv),
        is_stock=is_stock,
        is_order=is_order,
        is_flow=is_flow,
        d0=r1(d0),
        C0=r1(C0),
        I_fd=r1(I_fd),
        G0=r1(G0),
        X0=r1(X0),
        wages_pre_import=wages_pre,
        wages_post_import=wages_post,
        va_pre_import=va_pre_tot,
        va_post_import=va_post_tot,
        stor_in=stor_in,
        crit=crit,
    )
