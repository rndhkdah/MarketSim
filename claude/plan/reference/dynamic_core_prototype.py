#!/usr/bin/env python3
"""
REFERENCE PROTOTYPE for Phase 2 (claude/plan/02-phase2-dynamic-layer.md) -- NOT production code.

Single region, NPC mass only, monthly step, no ledger, no banks, no typed edges. It exists to show that the
equations in the plan are stable, give hump-shaped impulse responses and an exact steady state at pi* = 0 and 2 %
with the unchanged Layer-1 config, and to pin the unit contract of edges.yaml:capex.coefficients.

Known differences from the plan: (1) credit_pricing="risk_based" scales the whole cost-of-capital gap by
max(nd,0.5)/2.5 (plan: only the spread term); the default "uniform" mode matches the plan (D14); (2) households hold firm debt directly (plan: through BANKSYS); (3) debt is a plain array, not ledger
postings. `smooth_mode="real"` is kept only to reproduce the finding that deflating by realised CPI removes a damper.

Run:  python claude/plan/reference/prototype_checks.py        (add --slow for the 100-year stochastic runs)
Needs numpy, scipy, pyyaml and the legacy tree claude/marketsim (or MARKETSIM_ROOT pointing at it).
"""
import os, sys
import numpy as np
import yaml

ROOT = os.environ.get("MARKETSIM_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "marketsim"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from build_io import build_A, build_final_demand, CODES, IDX, MU, FD_WEIGHTS, N  # noqa

SEC = yaml.safe_load(open(os.path.join(ROOT, "config/sectors.yaml")))
EDG = yaml.safe_load(open(os.path.join(ROOT, "config/edges.yaml")))
S = N
g = lambda c, k: SEC["sectors"][c].get(k, SEC["defaults"].get(k))
vec = lambda k: np.array([g(c, k) for c in CODES], dtype=float)

# ----------------------------------------------------------------------------
# production mode per sector (NEW config: dynamics.yaml:production_mode)
#   stock : make-to-stock, finished-goods inventory, backlog on stock-out
#   order : make-to-order, order book worked down at capacity
#   flow  : produced on demand, unmet demand is lost
# ----------------------------------------------------------------------------
MODE = dict(ENERGY="stock", MATERIALS="stock", AGRIFOOD="stock", SEMIS="stock",
            AUTOS="stock", STAPLES="stock", DISCRET="stock", HEALTH="stock",
            CAPGOODS="order", CONSTRUCT="order",
            UTILITIES="flow", TRANSPORT="flow", SOFTWARE="flow", TELECOM="flow",
            BIZSVC="flow", BANKS="flow", INSURANCE="flow", REALESTATE="flow")
is_stock = np.array([MODE[c] == "stock" for c in CODES])
is_order = np.array([MODE[c] == "order" for c in CODES])
is_flow = np.array([MODE[c] == "flow" for c in CODES])

IMPORT_PRIOR = dict(ENERGY=0.15, MATERIALS=0.10, SEMIS=0.12, AUTOS=0.08, CAPGOODS=0.08,
                    AGRIFOOD=0.05, STAPLES=0.04, DISCRET=0.03)
LEAK = dict(AGRIFOOD=0.015, STAPLES=0.008, SEMIS=0.006, AUTOS=0.006, DISCRET=0.008,
            HEALTH=0.006, MATERIALS=0.004, ENERGY=0.004)


def default_params():
    return dict(
        pi_star=0.0,           # set 0.02 for the inflationary steady state
        tau_e=3.0,             # months, adaptive sales expectation
        cover_scale=0.25,      # finished-goods cover (months) = cover_scale*3*inv_lag_q
        tau_inv_mult=1.0,      # inventory gap closed over tau_inv = mult*cover months (min 2)
        ob_scale=0.5,          # order-book lead time (months) = ob_scale*3*inv_lag_q
        n_in=1.0,              # input-inventory target, months of usage
        tau_in=2.0,            # months to close the input-inventory gap
        backlog_loss=0.10,     # share of unfilled stock-mode orders lost each month
        tau_b=3.0,             # months to work off backlog (stock mode)
        kap_u=1.2, gam_cover=0.35, cover_min=0.15,   # price tightness terms
        step_max=0.15,         # max |dlog p| per month -- bound the STEP, never the level
        alpha1=0.70,
        tau_yd=12.0,
        delta_a=0.06,          # annual depreciation
        capex_unit=0.01,       # UNIT CONTRACT: coefficients are pct-pts of K per year
        q_scale=0.1,
        tau_g=12.0,            # months, expected-growth smoothing
        anchor_g=0.75,         # weight on trend growth in expected sector growth
        sl_weight=0.85,        # supply-line weight: share of capacity under construction firms account for
        rate_cap=3.0,          # starts capped at rate_cap * delta
        res_share=0.20,        # share of baseline INVESTMENT that is residential (household-driven, to CONSTRUCT)
        res_rate_semi=-4.0,    # % residential starts per +100bp real-rate gap
        res_income_el=1.0,
        tau_hire=3.0, tau_fire=6.0, overtime=1.10,
        U_star=0.05, benefit=0.40, tau_c=0.20,
        anchor=0.6,            # weight on target in expected inflation
        dem_rate_lag_m=6.0,
        eps_x=0.8,
        debt_to_gdp=0.6,
        core_weight=0.0,       # weight on core (ex ENERGY, AGRIFOOD) inflation in the Taylor rule
        tau_u=1.0,             # months, smoothing of the utilisation signal used by capex
        kap_B=0.04,            # per year: tax-rate response to debt-ratio gap (fiscal reaction function)
        kap_D=0.03,            # per year: share of excess firm debt repaid out of dividends
        # --- monetary policy framework (see plan 02-... §2.14). Defaults reproduce the
        # simple quarterly Taylor rule bitwise; every extension is off by default.
        meetings_per_year=4,   # 8 = Fed-like calendar
        rate_step=0.0,         # 0.0025 = announce in 25bp increments
        deadband=0.0,          # do not move unless the smoothed target is this far from the current rate
        phi_u=0.0,             # unemployment-gap coefficient (dual mandate); 0 = output gap only
        phi_y_mult=1.0,        # scale on the config output-gap coefficient
        data_lag_m=0,          # months of publication lag on the inflation / unemployment the rule sees
        rstar_kappa=0.0,       # r* moves with trend growth
        rstar_tau_m=60.0,
        makeup=0.0,            # weight on the price-level gap (average-inflation targeting)
        makeup_decay=1.0,      # 1.0 = pure integrator (UNSTABLE, verified); <1 = leaky memory
        makeup_clip=0.05,      # bound on the price-level gap the rule reacts to
        sahm_cut=0.0,          # extra easing (annual decimal) when the Sahm rule triggers
        credit_pricing="uniform",      # "uniform": every firm borrows at policy + ONE global spread (decision D14)
                                       # "risk_based": spread and cost-of-capital sensitivity scale with nd_ebitda/2.5
        smooth_mode="nominal_drift",   # "nominal_drift": smooth nominal values, drift-compensated by expected inflation
                                       # "real": smooth CPI-deflated values (kills nominal income illusion)
    )


class Chain:
    """Discrete Erlang(k) delay with mean `mean_m` months. Mass-conserving stock form."""
    def __init__(self, k, mean_m, shape):
        self.k = k
        mean_m = np.asarray(mean_m, dtype=float) * np.ones(shape)
        self.a = k / (mean_m + k)                 # per-stage exit prob; mean lag = k(1-a)/a
        self.q = np.zeros((k,) + tuple(shape))

    def seed(self, flow):                         # steady-state fill for a constant inflow
        for i in range(self.k):
            self.q[i] = flow * (1 - self.a) / self.a

    def push(self, inflow):
        x = inflow
        for i in range(self.k):
            self.q[i] += x
            x = self.a * self.q[i]
            self.q[i] -= x
        return x


class Smooth:
    """Signal (non-conserving) Erlang smoother: used for typed-edge lags."""
    def __init__(self, k, mean_m, init):
        self.k, self.a = k, k / (mean_m + k)
        self.s = np.array([np.array(init, dtype=float).copy() for _ in range(k)])

    def push(self, u):
        x = u
        for i in range(self.k):
            self.s[i] += self.a * (x - self.s[i])
            x = self.s[i]
        return x


class Economy:
    def __init__(self, P=None, gdp_m=100.0):
        self.P = P = {**default_params(), **(P or {})}
        self.A = A = build_A()
        fd = build_final_demand()
        self.fd = fd
        Linv = np.linalg.inv(np.eye(S) - A)
        self.fC0, self.fI0, self.fG0, self.fX0 = [FD_WEIGHTS[k] * gdp_m * fd[k]
                                                  for k in ("HOUSEHOLD", "INVESTMENT", "GOVT", "EXPORTS")]
        d0 = self.fC0 + self.fI0 + self.fG0 + self.fX0
        invq_ = vec("inv_lag_q")
        cover_ = np.where(is_stock, P["cover_scale"] * 3 * np.maximum(invq_, 1), 0.0)
        leak_ = np.array([LEAK.get(c, 0.0) for c in CODES]) * is_stock
        Lam = np.diag(1 + leak_ * cover_)
        self.x0 = x0 = np.linalg.solve(np.eye(S) - Lam @ A, Lam @ d0)
        self.s0 = x0 / (1 + leak_ * cover_)
        self.mu = mu = np.array([MU[c] for c in CODES])
        ws = np.array([EDG["labour"]["wage_share"][c] for c in CODES])

        # non-competing imports carved out of value added; scaled so trade balances
        m = np.array([IMPORT_PRIOR.get(c, 0.01) for c in CODES])
        m *= self.fX0.sum() / (m * x0).sum()
        self.m = m
        self.va = va = 1.0 - mu - m
        self.ell = ws * va                          # wage cost per unit output at w=1
        self.fc = vec("fixed_cost")
        self.ustar = vec("util_target")
        self.eta, self.eps = vec("eta"), vec("eps_own")
        self.pt, self.ptlag = vec("pass_through"), vec("pass_lag_q")
        self.dem_rate = vec("dem_rate_semi")
        self.build_m = 3.0 * vec("build_lag_q")
        invq = vec("inv_lag_q")
        self.cover = np.where(is_stock, P["cover_scale"] * 3 * np.maximum(invq, 1), 0.0)
        self.tau_inv = np.maximum(2.0, P["tau_inv_mult"] * self.cover)
        self.tau_ob = np.where(is_order, P["ob_scale"] * 3 * np.maximum(invq, 1), 1.0)
        self.leak = np.array([LEAK.get(c, 0.0) for c in CODES]) * is_stock
        self.nd = vec("nd_ebitda")

        # capital: value per unit of monthly capacity from a uniform gross return
        gos = (1 - ws) * va
        self.K = x0 / self.ustar
        self.delta_m = P["delta_a"] / 12
        I_bus0 = self.fI0.sum() * (1 - P["res_share"])
        rhoK = P["delta_a"] * (gos * x0).sum() / I_bus0
        self.v = 12 * gos * self.ustar / rhoK          # cost of 1 unit monthly capacity
        self.rhoK = rhoK
        assert abs((self.delta_m * self.v * self.K).sum() - I_bus0) < 1e-9
        cap = EDG["capex"]
        self.route = np.zeros(S)
        for c, w in cap["routing"].items():
            self.route[IDX[c]] = w
        cc = cap["coefficients"]
        self.phi, self.psi, self.chi, self.qT = (cc["phi_accelerator"], cc["psi_utilisation"],
                                                 cc["chi_cost_of_capital"], cc["q_tobin"])
        self.spend_mean = np.minimum(3.0 * cap["lag_q"], self.build_m)

        # criticality: storable + energy/power inputs that matter
        share = A / mu[None, :]
        crit_sup = np.array([c in ("ENERGY", "UTILITIES", "MATERIALS", "AGRIFOOD", "SEMIS",
                                   "CAPGOODS", "AUTOS", "TRANSPORT") for c in CODES])
        self.crit = (share >= 0.05) & crit_sup[:, None]
        self.stor_in = is_stock[:, None] & (A > 0)       # inputs held as stocks by buyers

        tay = EDG["policy"]["taylor"]
        self.tay = tay
        self.r_n = tay["r_neutral"]
        self.reset()

    # ------------------------------------------------------------------ init
    def reset(self):
        P, x0 = self.P, self.x0
        self.t = 0
        self.p = np.ones(S); self.w = 1.0; self.pimp = 1.0
        self.pf = np.zeros(S); self.ps = np.zeros(S)          # fast / slow log-price filters
        self.x = x0.copy(); self.sales = self.s0.copy(); self.se = self.s0.copy()
        self.K = x0 / self.ustar
        self.inv = self.cover * self.s0                       # end-of-month stock (pre-leak)
        self.backlog = np.where(is_order, (self.tau_ob - 1.0) * x0, 0.0)
        self.Sin = np.where(self.stor_in, P["n_in"] * self.A * x0[None, :], 0.0)
        self.fill_prev = np.ones(S)
        # labour
        self.n = self.ell * x0
        self.LF = self.n.sum() / (1 - P["U_star"])
        # capacity pipeline + spending pipeline
        starts0 = self.delta_m * self.K
        self.pipe = Chain(3, self.build_m, (S,)); self.pipe.seed(starts0)
        self.spend = Chain(3, self.spend_mean, (S,)); self.spend.seed(self.v * starts0)
        self.g_e = np.zeros(S); self.u_s = self.ustar.copy()
        G = float(np.exp(P["pi_star"] / 12)); self.G = G
        self.core_hist = [G ** (k - 12) for k in range(13)]
        self.K0 = self.K.copy(); self.pipe0 = self.pipe.q.sum(0).copy()
        I0 = self.fI0.sum(); self.res0 = P["res_share"] * I0
        bus = self.fI0.copy(); bus[IDX["CONSTRUCT"]] -= self.res0
        self.route_bus = bus / bus.sum(); self.bus_scale = bus.sum() / I0
        self.res_s = Smooth(2, 3.0, self.res0)
        self.sales_ma = self.s0.copy()
        # policy
        self.pi_e = P["pi_star"]; self.r = self.r_n + P["pi_star"]; self.infl12 = P["pi_star"]
        self.r_rule = self.r; self.r_ann = self.r
        self.hist_pi = [P["pi_star"]] * 26; self.hist_gap = [0.0] * 26; self.hist_U = [P["U_star"]] * 26
        self.g_trend = 0.0; self.gdp_prev = None; self.plevel_gap = 0.0; self.n_moves = 0; self.move_size = 0.0
        self.cpi_hist = [G ** (k - 12) for k in range(13)]
        self.rate_gap_s = Smooth(2, P["dem_rate_lag_m"], 0.0)
        # firm debt from nd_ebitda
        eb0 = self.s0 - (self.mu + self.ell + self.m) * x0      # revenue on SALES, cost on OUTPUT
        self.eb0 = eb0
        self.debt = self.nd * 12 * eb0
        self.uniform = P["credit_pricing"] == "uniform"
        self.spread = np.full(S, 0.015) if self.uniform else 0.015 * np.maximum(self.nd, 0.2) / 2.5
        # baseline income accounting -> solve tax rate and household wealth
        r0 = self.r
        wages0 = self.n.sum() * self.w
        intf = 1.0 / G                    # interest is paid on LAST month's stock, measured in this month's prices
        int_j = (r0 + self.spread) * self.debt / 12 * intf
        interest0 = int_j.sum()
        gos0 = eb0.sum()
        dep0 = (self.delta_m * self.v * self.K).sum()
        ctax0 = (P["tau_c"] * np.maximum(eb0 - int_j - self.delta_m * self.v * self.K, 0)).sum()
        grow = (G - 1.0) / G              # real net issuance per unit of real stock that keeps stock/price constant
        self.div0_tot = gos0 - interest0 - ctax0 - self.fI0.sum() * (1 - P["res_share"]) + grow * self.debt.sum()
        prof_at = eb0 - int_j - P["tau_c"] * np.maximum(eb0 - int_j - self.delta_m * self.v * self.K, 0)
        capex0 = self.delta_m * self.v * self.K
        self.payout = (prof_at - capex0 + grow * self.debt) / prof_at      # per sector: zero net borrowing at baseline
        self.payout_min = float(self.payout.min())
        self.prof_s = prof_at.copy(); self.eb_s = eb0.copy(); self.p_level = 1.0
        self.B = P["debt_to_gdp"] * 12 * (self.s0 - (self.mu + self.m) * x0).sum()
        transfers0 = P["benefit"] * self.w * (self.LF - self.n.sum())
        C0 = self.fC0.sum()
        # household: C0 = a1*YD0 + a2*W0, YD0 - C0 = grow*W0, W0 = ratio*12*YD0
        self.W = self.B + self.debt.sum()
        YD0 = C0 + P["res_share"] * self.fI0.sum() + grow * self.W
        self.alpha2 = (C0 - P["alpha1"] * YD0) / self.W
        assert self.alpha2 > 0, "alpha1 too high for the residential carve-out"
        pre_tax = wages0 + self.div0_tot + interest0 + r0 * self.B / 12 * intf + transfers0
        self.tau_y = 1 - YD0 / pre_tax
        self.YD_e = YD0; self.YD0 = YD0; self.tau_y_eff = self.tau_y
        self.G_real = self.fG0.copy()
        self.gdp0 = (self.s0 - (self.mu + self.m) * x0).sum()     # spoilage is a loss of output
        self.gdp_prev = self.gdp0
        self.theta = self.fC0 / self.fC0.sum()
        self.markup = 1.0 / (self.A.sum(0) + self.ell + self.m)   # p*=1 at baseline
        self.sh = dict(dem=0.0, cost=np.zeros(S), mon=0.0, sup=np.zeros(S), fisc=0.0, row=0.0)
        self.log = []

    # ------------------------------------------------------------------ step
    def step(self):
        P, A = self.P, self.A
        sh = self.sh
        # 1 expectations
        self.se += (self.sales - self.se) / P["tau_e"]
        Pc = (self.theta * self.p).sum()
        # 2 final demand
        real_mode = P["smooth_mode"] == "real"
        gd = float(np.exp(self.pi_e / 12))                # expected one-month price drift
        YDe_real = self.YD_e if real_mode else self.YD_e / Pc
        y = YDe_real / self.YD0
        rate_gap = self.rate_gap_s.push((self.r - self.pi_e - self.r_n) * 100.0)   # in 100bp
        C_nom = (P["alpha1"] * YDe_real * Pc + self.alpha2 * self.W) * np.exp(sh["dem"])
        z = self.theta * y ** (self.eta - 1) * (self.p / Pc) ** (1 + self.eps) \
            * np.exp(self.dem_rate / 100.0 * rate_gap)
        hh_shift = (z.sum() / self.theta.sum())          # rate-sensitive goods shrink the budget too
        c = C_nom * hh_shift ** 0.5 * (z / z.sum()) / self.p
        gv = self.G_real * np.exp(sh["fisc"])
        ex = self.fX0 * np.exp(sh["row"]) * (self.p / (self.pimp * Pc ** 0 )) ** (-P["eps_x"])
        # 3 investment: starts -> spending (demand) and -> capacity (supply)
        self.u_s += (self.x / self.K - self.u_s) / P["tau_u"]
        u = self.u_s
        cc_gap = (self.r - self.pi_e - self.r_n) * 100.0 * np.ones(S)        # 100bp units
        pipe_now = self.pipe.q.sum(0)
        sl = P["sl_weight"] * self.ustar * (pipe_now - self.pipe0 * self.K / self.K0) / self.K
        g_exp = (1 - P["anchor_g"]) * self.g_e
        rate = P["delta_a"] + P["capex_unit"] * (
            self.phi * g_exp * 100.0 + self.psi * ((u - self.ustar) - sl) * 100.0
            - self.chi * cc_gap * (1.0 if self.uniform else np.maximum(self.nd, 0.5) / 2.5))
        rate = np.clip(rate, 0.0, P["rate_cap"] * P["delta_a"])
        starts = self.K * rate / 12
        spend_real = self.spend.push(self.v * starts)
        res_real = self.res0 * (YDe_real / self.YD0) ** P["res_income_el"] \
            * np.exp(P["res_rate_semi"] / 100.0 * rate_gap) * np.exp(sh["dem"])
        res_real = self.res_s.push(res_real)
        inv_goods = self.route_bus * spend_real.sum()
        inv_goods[IDX["CONSTRUCT"]] += res_real
        self.K = (1 - self.delta_m) * self.K + self.pipe.push(starts)
        Keff = self.K * np.exp(sh["sup"])
        # 4 plans
        plan = np.where(is_stock,
                        self.se + self.leak * self.inv + (self.cover * self.se - self.inv) / self.tau_inv
                        + self.backlog / P["tau_b"],
                        self.se)
        plan = np.clip(plan, 0, Keff)
        # labour cap
        nvar = np.maximum(self.n - self.fc * self.ell * self.K * self.ustar, 1e-9)
        lab_cap = nvar / np.maximum((1 - self.fc) * self.ell * np.exp(-sh["sup"]), 1e-9) * P["overtime"]
        cap = np.minimum(Keff, lab_cap)
        # input cap (storable critical inputs from stock; flow critical inputs via last fill)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self.stor_in & self.crit, self.Sin / np.where(A > 0, A, 1), np.inf)
        in_cap = ratio.min(axis=0)
        flow_crit = np.where((~self.stor_in) & self.crit, self.fill_prev[:, None], 1.0).min(axis=0)
        x_goods = np.minimum.reduce([plan, cap, in_cap]) * flow_crit
        # 5 orders (all on plans -> no simultaneity)
        O = A * plan[None, :]
        O = O + np.where(self.stor_in, (P["n_in"] * A * plan[None, :] - self.Sin) / P["tau_in"], 0.0)
        O = np.maximum(O, 0.0)
        final = c + gv + ex + inv_goods
        new_orders = O.sum(1) + final
        D = new_orders + np.where(is_stock, self.backlog, 0.0)
        # 6 supply and rationing
        x = np.where(is_stock, x_goods, 0.0)
        ob = self.backlog + new_orders
        x = np.where(is_order, np.minimum(ob / self.tau_ob, cap * flow_crit), x)
        x = np.where(is_flow, np.minimum(D, cap * flow_crit), x)
        inv_prev = self.inv.copy()
        avail = np.where(is_stock, x + (1 - self.leak) * self.inv, x)
        dem_eff = np.where(is_order, x, D)                # order mode delivers what it builds
        fill = np.minimum(1.0, avail / np.maximum(dem_eff, 1e-12))
        sales = fill * dem_eff
        self.inv = np.where(is_stock, avail - sales, 0.0)
        self.backlog = np.where(is_stock, (D - sales) * (1 - P["backlog_loss"]),
                                np.where(is_order, ob - x, 0.0))
        # deliveries to buyers: stock/flow pro-rata on D ; order-mode pro-rata on order book
        dshare = np.where(is_order, x / np.maximum(ob, 1e-12), fill)
        deliv = O * dshare[:, None]
        used = A * x[None, :]
        self.Sin = np.where(self.stor_in, np.maximum(self.Sin + deliv - used, 0.0), 0.0)
        self.fill_prev = fill
        self.x, self.sales = x, sales
        # 7 prices: cost-plus target x tightness, two-speed pass-through, bounded STEP
        nuc = A.T @ self.p + self.w * self.ell * np.exp(-sh["sup"]) + self.m * self.pimp
        cov = np.where(is_stock, self.inv / np.maximum(self.se, 1e-9), 1.0)
        tight = np.exp(P["kap_u"] * (x / self.K - self.ustar)) \
            * np.where(is_stock, (self.cover / np.maximum(cov, P["cover_min"])) ** P["gam_cover"], 1.0)
        drift = self.pi_e / 12
        lp_star = np.log(self.markup * nuc * tight) + drift + sh["cost"]
        self.pf += drift + 0.5 * (lp_star - self.pf - drift)
        a_s = 1.0 / (1.0 + 3.0 * self.ptlag)
        self.ps += drift + a_s * (lp_star - self.ps - drift)
        lp_new = self.pt * self.pf + (1 - self.pt) * self.ps
        dlp = np.clip(lp_new - np.log(self.p), -P["step_max"], P["step_max"])
        self.p = self.p * np.exp(dlp)
        self.pimp *= self.G                               # ROW prices drift at pi* (no FX in v1)
        Pn = (self.theta * self.p).sum()                  # CPI after this month's price update
        # 8 labour and wages
        n_star = self.ell * np.exp(-sh["sup"]) * (self.fc * self.K * self.ustar + (1 - self.fc) * plan)
        tau_n = np.where(n_star > self.n, P["tau_hire"], P["tau_fire"])
        self.n = self.n + (n_star - self.n) / tau_n
        self.n *= min(1.0, 0.995 * self.LF / self.n.sum())
        U = 1 - self.n.sum() / self.LF
        ph = EDG["labour"]["phillips_slope"]
        dw = self.pi_e + ph * (P["U_star"] - U) * 100 / 100
        if dw < 0:
            dw /= EDG["labour"]["wage_stickiness_q"]
        self.w *= np.exp(dw / 12)
        # 9 incomes
        rev = self.p * sales
        wages = self.w * self.n
        imp = self.pimp * self.m * x
        interest = (self.r + self.spread) * self.debt / 12
        ebitda = rev - (self.p[:, None] * used).sum(0) - wages - imp
        dep = self.delta_m * self.v * self.K * (self.route * self.p).sum()
        ctax = P["tau_c"] * np.maximum(ebitda - interest - dep, 0)
        prof = ebitda - interest - ctax
        if real_mode:
            self.prof_s += (prof / Pn - self.prof_s) / 6.0    # smoothed in REAL terms
            self.eb_s += (ebitda / Pn - self.eb_s) / 12.0
            debt_gap_real = self.debt / Pc - self.nd * 12 * np.maximum(self.eb_s, 0)
            div_j = (self.payout * np.maximum(self.prof_s, 0) - P["kap_D"] / 12 * debt_gap_real) * Pn
        else:                                                 # nominal, drift-compensated
            self.prof_s = self.prof_s * gd + (prof - self.prof_s * gd) / 6.0
            self.eb_s = self.eb_s * gd + (ebitda - self.eb_s * gd) / 12.0
            debt_gap = self.debt * gd - self.nd * 12 * np.maximum(self.eb_s, 0)
            div_j = self.payout * np.maximum(self.prof_s, 0) - P["kap_D"] / 12 * debt_gap
        div_j = np.maximum(div_j, 0.0)
        div = div_j.sum()
        pI = (self.route_bus * self.p).sum()
        capex_nom = pI * spend_real
        self.debt += capex_nom - (prof - div_j)
        transfers = P["benefit"] * self.w * (self.LF - self.n.sum())
        pre_tax = wages.sum() + div + interest.sum() + self.r * self.B / 12 + transfers
        gdp_nom_a = 12 * (rev - (self.p[:, None] * used).sum(0) - imp).sum()
        b_ratio = (self.B / Pc) / (gdp_nom_a / Pn) if real_mode else self.B * gd / gdp_nom_a
        tau_y = np.clip(self.tau_y + P["kap_B"] * (b_ratio - P["debt_to_gdp"]), 0.0, 0.6)
        self.tau_y_eff += (tau_y - self.tau_y_eff) / 12.0
        YD = (1 - self.tau_y_eff) * pre_tax
        C_spent = (self.p * c * np.where(is_order, 1.0, fill)).sum()
        self.W += YD - C_spent - self.p[IDX["CONSTRUCT"]] * res_real
        if real_mode:
            self.YD_e += (YD / Pn - self.YD_e) / P["tau_yd"]
        else:
            self.YD_e = self.YD_e * gd + (YD - self.YD_e * gd) / P["tau_yd"]
        G_nom = (self.p * gv).sum()
        self.B += G_nom + transfers + self.r * self.B / 12 - self.tau_y_eff * pre_tax - ctax.sum()
        # 10 growth expectations, inflation, policy (quarterly)
        self.sales_ma += (sales - self.sales_ma) / P["tau_g"]
        g_now = 12 * np.log(np.maximum(sales, 1e-9) / np.maximum(self.sales_ma, 1e-9)) / (P["tau_g"])
        self.g_e += (g_now - self.g_e) / P["tau_g"]
        cpi = (self.theta * self.p).sum()
        self.cpi_hist.append(cpi); self.cpi_hist.pop(0)
        self.infl12 = np.log(self.cpi_hist[-1] / self.cpi_hist[0])
        infl3 = 4 * np.log(self.cpi_hist[-1] / self.cpi_hist[-4])
        cw = self.theta.copy(); cw[IDX["ENERGY"]] = 0; cw[IDX["AGRIFOOD"]] = 0; cw /= cw.sum()
        self.core_hist.append((cw * self.p).sum()); self.core_hist.pop(0)
        core12 = np.log(self.core_hist[-1] / self.core_hist[0])
        core3 = 4 * np.log(self.core_hist[-1] / self.core_hist[-4])
        k_ = P["core_weight"]
        pol_infl = (1 - k_) * (0.5 * infl3 + 0.5 * self.infl12) + k_ * (0.5 * core3 + 0.5 * core12)
        self.pi_e += ((P["anchor"] * P["pi_star"] + (1 - P["anchor"]) * self.infl12) - self.pi_e) / 6.0
        gdp = (x - self.leak * inv_prev - (self.mu + self.m) * x).sum()
        gap = gdp / self.gdp0 - 1
        # ---- monetary policy framework -------------------------------------------------
        t_ = self.tay
        self.hist_pi.append(pol_infl); self.hist_gap.append(gap); self.hist_U.append(U)
        self.g_trend += (12 * np.log(max(gdp, 1e-9) / max(self.gdp_prev, 1e-9)) - self.g_trend) / P["rstar_tau_m"]
        self.gdp_prev = gdp
        self.plevel_gap = P["makeup_decay"] * self.plevel_gap + (pol_infl - P["pi_star"]) / 12
        mpy = P["meetings_per_year"]
        meeting = (self.t + 1) * mpy // 12 > self.t * mpy // 12
        if meeting:
            L = int(P["data_lag_m"])
            pi_s = self.hist_pi[-1 - L]; gap_s = self.hist_gap[-1 - L]; U_s = self.hist_U[-1 - L]
            rstar = self.r_n + P["rstar_kappa"] * self.g_trend
            target = rstar + P["pi_star"] + t_["phi_inflation"] * (pi_s - P["pi_star"]) \
                + P["phi_y_mult"] * t_["phi_output_gap"] * gap_s + P["phi_u"] * (P["U_star"] - U_s) \
                + P["makeup"] * float(np.clip(self.plevel_gap, -P["makeup_clip"], P["makeup_clip"]))
            if P["sahm_cut"] > 0 and len(self.hist_U) >= 15:
                u3 = np.mean(self.hist_U[-3 - L:len(self.hist_U) - L]); u_min = min(self.hist_U[-15 - L:-3 - L])
                if u3 - u_min >= 0.005:
                    target -= P["sahm_cut"]
            rho = t_["smoothing"] ** (4.0 / mpy)
            self.r_rule = rho * self.r_rule + (1 - rho) * target
            cand = self.r_rule if P["rate_step"] <= 0 else round(self.r_rule / P["rate_step"]) * P["rate_step"]
            if abs(cand - self.r_ann) > P["deadband"] - 1e-15:
                if cand != self.r_ann:
                    self.n_moves += 1; self.move_size += abs(cand - self.r_ann)
                self.r_ann = cand
        self.r = max(0.0, self.r_ann + sh["mon"])
        self.t += 1
        rec = dict(t=self.t, gdp=gdp, gap=gap, cpi=cpi, infl=self.infl12, r=self.r, U=U, x=x.copy(),
                   p=self.p.copy(), K=self.K.copy(), inv=self.inv.copy(), I=spend_real.sum() + res_real, Ires=res_real,
                   C=(c * fill).sum(), w=self.w, fill=fill.copy(), B=self.B, W=self.W,
                   YD=YD, div=div, wages=wages.sum(), interest=interest.sum(), transfers=transfers, ctax=ctax.sum(), pre_tax=pre_tax, prof=prof.copy(), ebitda=ebitda.copy(), debt=self.debt.sum(), backlog=self.backlog.copy(), sales=sales.copy())
        self.log.append(rec)
        return rec


def run(T, shock=None, P=None):
    e = Economy(P)
    out = []
    for t in range(T):
        if shock:
            shock(e, t)
        out.append(e.step())
    return e, out


if __name__ == "__main__":
    e, out = run(240)
    x0 = e.x0
    dev = max(np.abs(o["x"] / x0 - 1).max() for o in out)
    print(f"no-shock run, 240 months: max |x/x0-1| = {dev:.2e}; "
          f"final gap {out[-1]['gap']:+.2e}, infl {out[-1]['infl']:+.2e}, r {out[-1]['r']:.4f}, U {out[-1]['U']:.4f}")
