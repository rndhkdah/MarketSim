"""Closed-form real-side steady state (§2.3). Shape is (R, S) with R = 1 in Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    mu: np.ndarray

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
    # Accounting intensity is the live column sum; io.mu can differ by reconstruction tol (~2e-6).
    mu = A.sum(axis=0)
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
        mu=r1(mu),
    )


@dataclass
class FinancialBaseline:
    """Passthrough-mode financial seed (§2.3 rest, §2.2 opening)."""

    pi_star: float
    G: float
    grow: float
    r0: float
    debt: np.ndarray
    spread: np.ndarray
    ebitda0: np.ndarray
    int0: np.ndarray
    dep0: np.ndarray
    tax0: np.ndarray
    payout: np.ndarray
    B: float
    W: float
    YD0: float
    alpha2: float
    tau_y: float
    pretax0: float
    gdp0: float
    vat: float
    transfers0: float
    wages0: float
    div0: float
    wo0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    icr0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bank_loans: float = 0.0
    bank_deposits: float = 0.0
    bank_equity: float = 0.0
    bank_reserves: float = 0.0
    bank_gb: float = 0.0
    cb_gb: float = 0.0
    hh_gb: float = 0.0
    bank_profit0: float = 0.0
    dep_rate0: float = 0.0


def compute_financial_baseline(
    real: RealBaseline,
    cfg: Config,
    io: IOTable,
    *,
    pi_star: float = 0.0,
    vat: float | None = None,
) -> FinancialBaseline:
    """Solve debt, payout, α2 and τ_y so π* is an exact real steady state."""
    assert cfg.dynamics is not None
    if io.n != real.S:
        raise ValueError("IO table and RealBaseline sector counts differ")
    dyn = cfg.dynamics
    x0 = real.flat(real.x0)
    s0 = real.flat(real.s0)
    m = real.flat(real.m)
    ell = real.flat(real.ell)
    k0 = real.flat(real.K0)
    v = real.flat(real.v)
    mu = real.flat(real.mu)
    n0 = real.flat(real.n0)
    g = float(np.exp(pi_star / 12.0))
    grow = (g - 1.0) / g
    r0 = cfg.edges.policy.taylor.r_neutral + pi_star
    nd = np.array([cfg.sectors.params(c).nd_ebitda for c in real.codes], dtype=float)
    ebitda0 = s0 - (mu + ell + m) * x0
    debt = nd * 12.0 * ebitda0
    s0_spread = dyn.firms.base_spread
    if dyn.credit.pricing == "uniform":
        spread = np.full(real.S, s0_spread)
    else:
        spread = s0_spread * np.maximum(nd, dyn.firms.spread_leverage_floor) / 2.5
    int0 = (r0 + spread) * debt / 12.0 / g
    delta_m = dyn.capex.delta_annual / 12.0
    dep0 = delta_m * v * k0
    tax0 = dyn.fiscal.corp_tax * np.maximum(ebitda0 - int0 - dep0, 0.0)
    prof = ebitda0 - int0 - tax0
    capex0 = dep0
    payout = (prof - capex0 + grow * debt) / prof
    gdp0 = float((s0 - (mu + m) * x0).sum())
    b = dyn.fiscal.debt_to_gdp * 12.0 * gdp0
    w_hh = b + float(debt.sum())
    vat_ = dyn.fiscal.vat if vat is None else float(vat)
    c0 = float(real.flat(real.C0).sum())
    yd0 = c0 * (1.0 + vat_) + real.res0 + grow * w_hh
    alpha2 = (c0 * (1.0 + vat_) - dyn.households.alpha1 * yd0) / w_hh
    wages0 = float(n0.sum())
    transfers0 = dyn.fiscal.benefit_replacement * (real.LF - float(n0.sum()))
    div0 = float((prof - capex0 + grow * debt).sum())
    interest_hh = float(int0.sum()) + r0 * b / 12.0 / g
    pretax0 = wages0 + div0 + interest_hh + transfers0
    tau_y = 1.0 - yd0 / pretax0
    icr0 = np.where(int0 > 1e-12, ebitda0 / int0, 1e6)
    fin = FinancialBaseline(
        pi_star=float(pi_star),
        G=g,
        grow=grow,
        r0=r0,
        debt=debt,
        spread=spread,
        ebitda0=ebitda0,
        int0=int0,
        dep0=dep0,
        tax0=tax0,
        payout=payout,
        B=float(b),
        W=float(w_hh),
        YD0=float(yd0),
        alpha2=float(alpha2),
        tau_y=float(tau_y),
        pretax0=float(pretax0),
        gdp0=float(gdp0),
        vat=vat_,
        transfers0=float(transfers0),
        wages0=wages0,
        div0=div0,
        icr0=icr0,
        wo0=np.zeros_like(debt),
    )
    if dyn.banks.mode == "full":
        from marketsim.real.banks import apply_full_initialiser

        fin = apply_full_initialiser(fin, real, cfg)
    return fin
