"""Monthly real-economy step (R1–R10) plus passthrough settlement."""

from __future__ import annotations

from typing import Any

import numpy as np

from marketsim.core.config import Config
from marketsim.core.erlang import ErlangSmoother
from marketsim.core.module import Phase
from marketsim.layer1.io import IOTable
from marketsim.ledger.opening import open_passthrough_books
from marketsim.real.capex import (
    cost_of_capital_gap,
    seed_pipelines,
    start_rate,
    step_capacity,
    supply_line,
)
from marketsim.real.cenbank import CentralBank
from marketsim.real.government import debt_ratio, step_tax_rate, tax_rate_target
from marketsim.real.households import consumption_nominal, household_basket, income_index, smooth_nominal
from marketsim.real.labour import step_labour
from marketsim.real.orders import order_matrix, supply_and_ration
from marketsim.real.prices import PriceState, sector_pass_through, step_prices, tightness, unit_cost
from marketsim.real.production import (
    expected_sales,
    flow_crit_fill,
    goods_output,
    input_cap,
    labour_cap,
    plan_output,
)
from marketsim.real.residential import ResidentialBlock
from marketsim.real.row import exports, import_bill
from marketsim.real.settlement import MonthFlows, settle_and_check
from marketsim.real.steady_state import (
    FinancialBaseline,
    RealBaseline,
    compute_financial_baseline,
    compute_real_baseline,
)


class RealEconomy:
    """NPC-mass monthly model. Shape is (S,) with R = 1 in Phase 2."""

    name = "real_economy"

    def __init__(self, cfg: Config, io: IOTable, *, pi_star: float = 0.0, check_sfc: bool = True) -> None:
        assert cfg.dynamics is not None
        self.cfg = cfg
        self.io = io
        self.real: RealBaseline = compute_real_baseline(io, cfg)
        self.fin: FinancialBaseline = compute_financial_baseline(self.real, cfg, io, pi_star=pi_star)
        self.check_sfc = check_sfc
        self.ledger = open_passthrough_books(cfg, self.real, self.fin)
        self._init_state()

    def _init_state(self) -> None:
        cfg, real, fin = self.cfg, self.real, self.fin
        dyn = cfg.dynamics
        s = real.S
        self.codes = real.codes
        self.A = self.io.A
        self.p = np.ones(s)
        self.w = 1.0
        self.prices = PriceState(p=self.p, pf=np.zeros(s), ps=np.zeros(s), p_imp=1.0)
        self.x = real.flat(real.x0).copy()
        self.sales = real.flat(real.s0).copy()
        self.se = real.flat(real.s0).copy()
        self.k = real.flat(real.K0).copy()
        self.inv = real.flat(real.inv0).copy()
        self.backlog = real.flat(real.backlog0).copy()
        self.s_in = real.S_in0[0].copy()
        self.fill_prev = np.ones(s)
        self.n = real.flat(real.n0).copy()
        self.pipe, self.spend, self.pipe0 = seed_pipelines(real, cfg)
        self.g_e = np.zeros(s)
        self.u_s = np.array([cfg.sectors.params(c).util_target for c in real.codes])
        self.fc = np.array([cfg.sectors.params(c).fixed_cost for c in real.codes])
        self.ustar = self.u_s.copy()
        self.nd = np.array([cfg.sectors.params(c).nd_ebitda for c in real.codes])
        self.pt, self.ptlag = sector_pass_through(cfg, real.codes)
        self.route = np.zeros(s)
        for c, w in cfg.edges.capex.routing.items():
            self.route[real.codes.index(c)] = w
        self.res = ResidentialBlock(cfg, real)
        self.sales_ma = real.flat(real.s0).copy()
        self.cb = CentralBank.from_config(cfg, fin.pi_star)
        self.rate_gap_s = ErlangSmoother(dyn.households.rate_lag.k, dyn.households.rate_lag.mean_m, 0.0)
        self.debt = fin.debt.copy()
        self.spread = fin.spread.copy()
        self.payout = fin.payout.copy()
        self.prof_s = (fin.ebitda0 - fin.int0 - fin.tax0).copy()
        self.eb_s = fin.ebitda0.copy()
        self.b = fin.B
        self.wealth = fin.W
        self.yd_e = fin.YD0
        self.tau_eff = fin.tau_y
        self.demand = household_basket(real, cfg, fin)
        self.theta = real.flat(real.C0) / real.flat(real.C0).sum()
        self.sh = {
            "dem": 0.0,
            "cost": np.zeros(s),
            "mon": 0.0,
            "sup": np.zeros(s),
            "fisc": 0.0,
            "row": 0.0,
            "imp": 0.0,
            "risk": 0.0,
            "ds": 0.0,
        }
        self.month = 0
        self.last_flows: MonthFlows | None = None

    def step_month(self) -> dict[str, Any]:
        cfg, real, fin, dyn = self.cfg, self.real, self.fin, self.cfg.dynamics
        a = self.A
        sh = self.sh
        s = real.S
        # R1
        self.se = expected_sales(self.se, self.sales, dyn.expectations.tau_sales_m)
        pc = float((self.theta * self.p).sum())
        g_d = float(np.exp(self.cb.pi_e / 12.0))
        y = income_index(self.yd_e, pc, fin.YD0)
        rate_gap = float(self.rate_gap_s.push((self.cb.r - self.cb.pi_e - self.cb.r_n) * 100.0))
        c_nom_tot = consumption_nominal(dyn.households.alpha1, self.yd_e, fin.alpha2, self.wealth, sh["dem"])
        c = self.demand.allocate(c_nom_tot, self.p, y, rate_gap)
        gv = real.flat(real.G0) * np.exp(sh["fisc"])
        ex = exports(real.flat(real.X0), self.p, self.prices.p_imp, dyn.row.export_price_elasticity, sh["row"])
        # R3
        self.u_s = self.u_s + (self.x / self.k - self.u_s) / dyn.capex.tau_util_m
        cap = cfg.edges.capex
        cc_gap = cost_of_capital_gap(
            self.cb.r,
            self.cb.pi_e,
            self.cb.r_n,
            sh["ds"],
            sh["risk"],
            self.nd,
            dyn.credit.pricing,
        )
        sl = supply_line(self.pipe.content(), self.pipe0, self.k, real.flat(real.K0), self.ustar, cap.supply_line_weight)
        q = np.zeros(s)
        rate = start_rate(
            delta=dyn.capex.delta_annual,
            unit_scale=cap.unit_scale,
            phi=cap.coefficients.phi_accelerator,
            psi=cap.coefficients.psi_utilisation,
            chi=cap.coefficients.chi_cost_of_capital,
            q_tobin_coef=cap.coefficients.q_tobin,
            q_scale=cap.q_scale,
            q_clip=dyn.capex.q_clip,
            g_e=self.g_e,
            anchor_growth=dyn.expectations.anchor_growth,
            u=self.u_s,
            ustar=self.ustar,
            sl=sl,
            cc_gap=cc_gap,
            ln_q=q,
            cap_mult=dyn.capex.start_rate_cap_mult,
        )
        starts = self.k * rate / 12.0
        spend_real = self.spend.push(real.flat(real.v) * starts)
        res_real = self.res.step(y, rate_gap, sh["dem"])
        inv_goods = real.flat(real.route_bus) * float(spend_real.sum())
        inv_goods = self.res.add_to_final(inv_goods, res_real)
        self.k = step_capacity(self.k, starts, self.pipe, dyn.capex.delta_annual / 12.0)
        k_eff = self.k * np.exp(sh["sup"])
        # R4
        plan = plan_output(
            self.se,
            self.inv,
            real.flat(real.cover),
            real.flat(real.leak),
            real.flat(real.tau_inv),
            self.backlog,
            dyn.production.tau_backlog_m,
            real.is_stock,
            k_eff,
        )
        lab = labour_cap(self.n, self.fc, real.flat(real.ell), self.k, self.ustar, dyn.production.overtime_cap, sh["sup"])
        in_c = input_cap(self.s_in, a, real.stor_in, real.crit)
        fcrit = flow_crit_fill(self.fill_prev, real.stor_in, real.crit)
        x_goods = goods_output(plan, k_eff, lab, in_c, fcrit)
        # R5–R6
        orders = order_matrix(
            a, plan, self.s_in, real.stor_in, dyn.production.input_cover_m, dyn.production.tau_input_m
        )
        final = c + gv + ex + inv_goods
        new_orders = orders.sum(axis=1) + final
        cap_out = np.minimum(k_eff, lab)
        inv_prev = self.inv.copy()
        supplied = supply_and_ration(
            x_goods=x_goods,
            new_orders=new_orders,
            final=final,
            orders=orders,
            a=a,
            inv=self.inv,
            backlog=self.backlog,
            s_in=self.s_in,
            leak=real.flat(real.leak),
            tau_ob=real.flat(real.tau_ob),
            cap=cap_out,
            flow_crit=fcrit,
            is_stock=real.is_stock,
            is_order=real.is_order,
            is_flow=real.is_flow,
            stor_in=real.stor_in,
            backlog_loss=dyn.production.backlog_loss,
        )
        self.x, self.sales = supplied.x, supplied.sales
        self.inv, self.backlog, self.s_in = supplied.inv, supplied.backlog, supplied.s_in
        self.fill_prev = supplied.fill
        dshare = supplied.dshare
        # R7
        nuc = unit_cost(a, self.p, self.w, real.flat(real.ell), real.flat(real.m), self.prices.p_imp, sh["sup"])
        tight = tightness(
            self.x,
            self.k,
            self.ustar,
            self.inv,
            self.se,
            real.flat(real.cover),
            real.is_stock,
            dyn.prices.kappa_util,
            dyn.prices.gamma_cover,
            dyn.prices.cover_floor,
        )
        self.prices = step_prices(
            self.prices,
            nuc=nuc,
            markup=real.flat(real.markup),
            tight=tight,
            z_cost=sh["cost"],
            pi_e=self.cb.pi_e,
            pt=self.pt,
            ptlag=self.ptlag,
            fast_mean_m=dyn.prices.fast_mean_m,
            step_max=dyn.prices.step_max_month,
            g=fin.G,
            dz_imp=sh["imp"],
        )
        self.p = self.prices.p
        # R8
        self.n, self.w, u = step_labour(
            self.n,
            self.w,
            ell=real.flat(real.ell),
            fc=self.fc,
            k=self.k,
            ustar=self.ustar,
            plan=plan,
            z_sup=sh["sup"],
            lf=real.LF,
            cfg=cfg,
            pi_e=self.cb.pi_e,
        )
        # R9
        used = a * self.x[None, :]
        rev = self.p * self.sales
        wages = self.w * self.n
        imp = import_bill(self.prices.p_imp, real.flat(real.m), self.x)
        interest = (self.cb.r + self.spread) * self.debt / 12.0
        ebitda = rev - (self.p[:, None] * used).sum(0) - wages - imp
        p_i = float((real.flat(real.route_bus) * self.p).sum())
        dep = (dyn.capex.delta_annual / 12.0) * real.flat(real.v) * self.k * float((self.route * self.p).sum())
        ctax = dyn.fiscal.corp_tax * np.maximum(ebitda - interest - dep, 0.0)
        prof = ebitda - interest - ctax
        self.prof_s = self.prof_s * g_d + (prof - self.prof_s * g_d) / dyn.firms.tau_profit_m
        self.eb_s = self.eb_s * g_d + (ebitda - self.eb_s * g_d) / dyn.firms.tau_ebitda_m
        debt_gap = self.debt * g_d - self.nd * 12.0 * np.maximum(self.eb_s, 0.0)
        div_j = np.maximum(self.payout * np.maximum(self.prof_s, 0.0) - dyn.firms.kappa_leverage / 12.0 * debt_gap, 0.0)
        capex_nom = p_i * spend_real
        d_debt = capex_nom - (prof - div_j)
        debt_prev = self.debt.copy()
        self.debt = self.debt + d_debt
        transfers = dyn.fiscal.benefit_replacement * self.w * (real.LF - float(self.n.sum()))
        pretax = float(wages.sum() + div_j.sum() + interest.sum() + self.cb.r * self.b / 12.0 + transfers)
        gdp_nom_a = 12.0 * float((rev - (self.p[:, None] * used).sum(0) - imp).sum())
        ratio = debt_ratio(self.b, gdp_nom_a, g_d)
        tau_tgt = tax_rate_target(fin.tau_y, dyn.fiscal.kappa_debt, ratio, dyn.fiscal.debt_to_gdp, dyn.fiscal.tax_rate_bounds)
        self.tau_eff = step_tax_rate(self.tau_eff, tau_tgt, dyn.fiscal.tau_tax_m)
        yd = (1.0 - self.tau_eff) * pretax
        c_nom_sec = self.p * c * np.where(real.is_order, 1.0, dshare)
        c_spent = float(c_nom_sec.sum())
        res_nom = float(self.p[real.codes.index("CONSTRUCT")] * res_real)
        self.wealth = self.wealth + yd - c_spent - res_nom
        self.yd_e = smooth_nominal(self.yd_e, yd, dyn.households.tau_income_m, g_d)
        g_nom_sec = self.p * gv
        g_nom = float(g_nom_sec.sum())
        income_tax = self.tau_eff * pretax
        deficit = g_nom + transfers + self.cb.r * self.b / 12.0 - income_tax - float(ctax.sum())
        self.b = self.b + deficit
        # R10
        self.sales_ma = self.sales_ma + (self.sales - self.sales_ma) / dyn.expectations.tau_growth_m
        g_now = 12.0 * np.log(np.maximum(self.sales, 1e-9) / np.maximum(self.sales_ma, 1e-9)) / dyn.expectations.tau_growth_m
        self.g_e = self.g_e + (g_now - self.g_e) / dyn.expectations.tau_growth_m
        cpi = float((self.theta * self.p).sum())
        core_w = self.theta.copy()
        core_w[real.codes.index("ENERGY")] = 0.0
        core_w[real.codes.index("AGRIFOOD")] = 0.0
        core_w /= core_w.sum()
        core = float((core_w * self.p).sum())
        self.cb.observe_prices(cpi, core)
        gdp = float((self.x - real.flat(real.leak) * inv_prev - (real.flat(real.mu) + real.flat(real.m)) * self.x).sum())
        gap = gdp / fin.gdp0 - 1.0
        self.cb.maybe_meet(gap, sh["mon"])
        i_nom = self.p * inv_goods * np.where(real.is_order, 1.0, dshare)
        # residential is already inside inv_goods on CONSTRUCT — split it back out of i_nom for the tag
        i_bus = i_nom.copy()
        i_bus[real.codes.index("CONSTRUCT")] -= res_nom
        flows = MonthFlows(
            p=self.p.copy(),
            sales=self.sales.copy(),
            c_nom=c_nom_sec,
            g_nom=g_nom_sec,
            i_nom=i_bus,
            res_nom=res_nom,
            ex_nom=self.p * ex * np.where(real.is_order, 1.0, dshare),
            imp_nom=imp,
            deliv_nom=self.p[:, None] * supplied.deliveries,
            wages=wages,
            transfers=transfers,
            income_tax=income_tax,
            corp_tax=ctax,
            interest_loans=interest,
            interest_bonds=self.cb.r * (self.b - deficit) / 12.0,
            dividends=div_j,
            d_debt=self.debt - debt_prev,
            deficit=deficit,
            vat=0.0,
        )
        self.last_flows = flows
        if self.check_sfc:
            settle_and_check(self.ledger, real, flows, tick=self.month + 1)
        self.month += 1
        return {
            "t": self.month,
            "gdp": gdp,
            "gap": gap,
            "cpi": cpi,
            "x": self.x.copy(),
            "U": u,
            "r": self.cb.r,
        }

    def on_phase(self, ctx: Any, phase: Phase) -> None:
        if phase is Phase.REAL and ctx.world.clock.calendar.is_month_end(ctx.tick):
            self.step_month()

    def reset(self, ctx: Any) -> None:
        del ctx
        self.ledger = open_passthrough_books(self.cfg, self.real, self.fin)
        self._init_state()

    def to_state(self) -> dict[str, Any]:
        return {"month": self.month, "x": self.x.copy(), "p": self.p.copy(), "r": self.cb.r}

    def from_state(self, state: dict[str, Any]) -> None:
        self.month = int(state["month"])
        self.x = np.asarray(state["x"], dtype=float)
        self.p = np.asarray(state["p"], dtype=float)
        self.cb.r = float(state["r"])
