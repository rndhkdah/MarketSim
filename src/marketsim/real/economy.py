"""Monthly real-economy step (R1–R10) plus passthrough settlement."""

from __future__ import annotations

from typing import Any

import numpy as np

from marketsim.core.config import Config
from marketsim.core.erlang import ErlangSmoother
from marketsim.core.module import Phase
from marketsim.layer1.io import IOTable
from marketsim.ledger.opening import GOVT_MIX, open_passthrough_books
from marketsim.real.aggregates import Aggregates, Published, expenditure_gdp, production_gdp
from marketsim.real.banks import bank_month_flows, open_full_books
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
from marketsim.real.settlement import MonthFlows, settle_and_check, settle_month
from marketsim.real.shocks import ShockBus
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
        self.ledger = self._open_books()
        self._init_state()

    def _open_books(self):
        dyn = self.cfg.dynamics
        assert dyn is not None
        if dyn.banks.mode == "full":
            return open_full_books(self.cfg, self.real, self.fin)
        return open_passthrough_books(self.cfg, self.real, self.fin)

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
        self.bank_deposits = fin.bank_deposits
        self.bank_equity = fin.bank_equity
        self.bank_reserves = fin.bank_reserves
        self.bank_gb = fin.bank_gb
        self.cb_gb = fin.cb_gb
        self.hh_gb = fin.hh_gb if fin.hh_gb else fin.B
        self.wealth = fin.W
        self.yd_e = fin.YD0
        self.tau_eff = fin.tau_y
        self.demand = household_basket(real, cfg, fin)
        self.theta = real.flat(real.C0) / real.flat(real.C0).sum()
        self.bus = ShockBus.from_config(cfg, real.codes)
        self.sh = self.bus.states
        self.month = 0
        self.last_flows: MonthFlows | None = None
        self.last_agg: Aggregates | None = None
        self.pub = Published()
        self.pub.push(
            gdp=fin.gdp0,
            cpi=1.0,
            core_cpi=1.0,
            u=dyn.labour.u_star,
            infl=fin.pi_star,
        )

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
        rate_gap = float(
            self.rate_gap_s.push((self.cb.r - self.cb.pi_e - self.cb.r_n) * 100.0 + float(sh["ds"]) * 100.0)
        )
        c_nom_tot = consumption_nominal(dyn.households.alpha1, self.yd_e, fin.alpha2, self.wealth, sh["dem"])
        c = self.demand.allocate(c_nom_tot, self.p, y, rate_gap)
        gv = real.flat(real.G0) * np.exp(sh["fisc"])
        ex = exports(real.flat(real.X0), self.p, self.prices.p_imp, dyn.row.export_price_elasticity, sh["row"])
        recon = self.bus.consume_recon()
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
            g_e=self.g_e + dyn.shocks.animal_spirits_weight * float(sh["dem"]),
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
        final = c + gv + ex + inv_goods + recon
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
        full = dyn.banks.mode == "full"
        wo = np.zeros(s)
        dep_int = 0.0
        bank_div = 0.0
        grow_now = (g_d - 1.0) / g_d
        if full:
            wo, dep_int, bank_div, _profit, _c = bank_month_flows(
                r=self.cb.r,
                debt=self.debt,
                nd=self.nd,
                ebitda=ebitda,
                interest=interest,
                deposits=self.bank_deposits,
                reserves=self.bank_reserves,
                gb=self.bank_gb,
                equity=self.bank_equity,
                grow=grow_now,
                g=g_d,
                fin=fin,
                cfg=cfg,
            )
        self.prof_s = self.prof_s * g_d + (prof - self.prof_s * g_d) / dyn.firms.tau_profit_m
        self.eb_s = self.eb_s * g_d + (ebitda - self.eb_s * g_d) / dyn.firms.tau_ebitda_m
        debt_gap = self.debt * g_d - self.nd * 12.0 * np.maximum(self.eb_s, 0.0)
        div_j = np.maximum(self.payout * np.maximum(self.prof_s, 0.0) - dyn.firms.kappa_leverage / 12.0 * debt_gap, 0.0)
        capex_nom = p_i * spend_real
        d_debt = capex_nom - (prof - div_j)
        self.debt = self.debt - wo + d_debt
        transfers = dyn.fiscal.benefit_replacement * self.w * (real.LF - float(self.n.sum()))
        if full:
            pretax = float(wages.sum() + div_j.sum() + bank_div + dep_int + self.cb.r * self.hh_gb / 12.0 + transfers)
        else:
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
        b_begin = self.b
        deficit = g_nom + transfers + self.cb.r * b_begin / 12.0 - income_tax - float(ctax.sum())
        if full and b_begin > 1e-12:
            d_bank_gb = deficit * (self.bank_gb / b_begin)
            d_cb_gb = deficit * (self.cb_gb / b_begin)
        else:
            d_bank_gb = 0.0
            d_cb_gb = 0.0
        self.b = b_begin + deficit
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
        leak = real.flat(real.leak)
        mu = real.flat(real.mu)
        mvec = real.flat(real.m)
        gdp = production_gdp(self.x, leak, inv_prev, mu, mvec)
        ex_real = ex * np.where(real.is_order, 1.0, dshare)
        d_inv = self.x - leak * inv_prev - self.sales
        gdp_exp = expenditure_gdp(
            c,
            gv,
            float(spend_real.sum() + res_real),
            ex,
            mvec * self.x,
            d_inv,
        )
        gap = gdp / fin.gdp0 - 1.0
        self.cb.maybe_meet(gap, sh["mon"])
        self.last_agg = Aggregates(
            gdp_prod_real=gdp,
            gdp_exp_real=gdp_exp,
            gdp_prod_nom=float((rev - (self.p[:, None] * used).sum(0) - imp).sum() + (self.p * d_inv).sum()),
            gdp_exp_nom=float(c_spent + g_nom + float((p_i * spend_real).sum()) + res_nom + float((self.p * ex_real).sum() - imp.sum()) + float((self.p * d_inv).sum())),
            cpi=cpi,
            core_cpi=core,
            u=u,
            utilisation=float((self.x / self.k).mean()),
            x=self.x.copy(),
            govt_balance=-deficit,
            debt_gdp=self.b / (12.0 * max(gdp * cpi, 1e-12)),
            trade_balance=float((self.p * ex_real).sum() - imp.sum()),
            saving_rate=float((yd - c_spent - res_nom) / max(yd, 1e-12)),
            leverage=float(self.debt.sum() / max(float(np.maximum(self.eb_s, 1e-12).sum()) * 12.0, 1e-12)),
            pi12=self.cb.pi12(),
        )
        self.pub.push(gdp=gdp, cpi=cpi, core_cpi=core, u=u, infl=self.cb.pi12())
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
            interest_bonds=self.cb.r * b_begin / 12.0,
            dividends=div_j,
            d_debt=d_debt,
            deficit=deficit,
            vat=0.0,
            interest_deposits=dep_int,
            bank_dividends=bank_div,
            writeoffs=wo if full else None,
            interest_reserves=self.cb.r * self.bank_reserves / 12.0 if full else 0.0,
            d_bank_gb=d_bank_gb,
            d_cb_gb=d_cb_gb,
            d_res=d_cb_gb,
            loan_holder="BANKSYS" if full else "",
            interest_bonds_hh=self.cb.r * self.hh_gb / 12.0 if full else None,
            interest_bonds_bank=self.cb.r * self.bank_gb / 12.0 if full else 0.0,
            interest_bonds_cb=self.cb.r * self.cb_gb / 12.0 if full else 0.0,
        )
        self.last_flows = flows
        if self.check_sfc:
            settle_and_check(self.ledger, real, flows, tick=self.month + 1)
        else:
            settle_month(self.ledger, real, flows, tick=self.month + 1)
        if full:
            self._refresh_bank_sheet()
        self.bus.decay()
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

    def _refresh_bank_sheet(self) -> None:
        from marketsim.ledger.sfc import net_financial_assets

        led = self.ledger
        self.bank_deposits = -led.position("BANKSYS", "DEP")
        self.bank_reserves = led.position("BANKSYS", "RES")
        self.bank_gb = sum(led.position("BANKSYS", inst) for inst, _ in GOVT_MIX)
        self.cb_gb = sum(led.position("CB", inst) for inst, _ in GOVT_MIX)
        self.hh_gb = sum(led.position("HH:0", inst) for inst, _ in GOVT_MIX)
        self.bank_equity = float(net_financial_assets(led)[led.entities.id("BANKSYS")])

    def reset(self, ctx: Any) -> None:
        del ctx
        self.ledger = self._open_books()
        self._init_state()

    def published(self, series: str, lag: int = 0) -> float:
        return self.pub.published(series, lag)

    def inject(self, kind: str, magnitude: float, **kwargs: Any) -> float:
        """``ShockBus.inject`` — the only public shock entry point (§2.9)."""
        kwargs.setdefault("economy", self)
        return self.bus.inject(kind, magnitude, **kwargs)

    def to_state(self) -> dict[str, Any]:
        cb = self.cb
        return {
            "month": self.month,
            "p": self.p.copy(),
            "w": self.w,
            "prices": {"p": self.prices.p.copy(), "pf": self.prices.pf.copy(), "ps": self.prices.ps.copy(), "p_imp": self.prices.p_imp},
            "x": self.x.copy(),
            "sales": self.sales.copy(),
            "se": self.se.copy(),
            "k": self.k.copy(),
            "inv": self.inv.copy(),
            "backlog": self.backlog.copy(),
            "s_in": self.s_in.copy(),
            "fill_prev": self.fill_prev.copy(),
            "n": self.n.copy(),
            "pipe": self.pipe.to_state(),
            "spend": self.spend.to_state(),
            "g_e": self.g_e.copy(),
            "u_s": self.u_s.copy(),
            "res": self.res.smoother.to_state(),
            "sales_ma": self.sales_ma.copy(),
            "cb": {
                "r_rule": cb.r_rule,
                "r": cb.r,
                "pi_e": cb.pi_e,
                "cpi_hist": list(cb.cpi_hist),
                "core_hist": list(cb.core_hist),
                "month": cb.month,
            },
            "rate_gap_s": self.rate_gap_s.to_state(),
            "debt": self.debt.copy(),
            "prof_s": self.prof_s.copy(),
            "eb_s": self.eb_s.copy(),
            "b": self.b,
            "bank_deposits": self.bank_deposits,
            "bank_equity": self.bank_equity,
            "bank_reserves": self.bank_reserves,
            "bank_gb": self.bank_gb,
            "cb_gb": self.cb_gb,
            "hh_gb": self.hh_gb,
            "wealth": self.wealth,
            "yd_e": self.yd_e,
            "tau_eff": self.tau_eff,
            "sh": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.sh.items()},
            "bus": self.bus.to_state(),
            "ledger": self.ledger.to_state(),
            "pub": self.pub.to_state(),
            "pi_star": self.fin.pi_star,
        }

    def from_state(self, state: dict[str, Any]) -> None:
        from marketsim.core.erlang import ErlangChain
        from marketsim.ledger.journal import Ledger

        self.month = int(state["month"])
        self.p = np.asarray(state["p"], dtype=float)
        self.w = float(state["w"])
        pr = state["prices"]
        self.prices = PriceState(
            p=np.asarray(pr["p"], dtype=float),
            pf=np.asarray(pr["pf"], dtype=float),
            ps=np.asarray(pr["ps"], dtype=float),
            p_imp=float(pr["p_imp"]),
        )
        self.x = np.asarray(state["x"], dtype=float)
        self.sales = np.asarray(state["sales"], dtype=float)
        self.se = np.asarray(state["se"], dtype=float)
        self.k = np.asarray(state["k"], dtype=float)
        self.inv = np.asarray(state["inv"], dtype=float)
        self.backlog = np.asarray(state["backlog"], dtype=float)
        self.s_in = np.asarray(state["s_in"], dtype=float)
        self.fill_prev = np.asarray(state["fill_prev"], dtype=float)
        self.n = np.asarray(state["n"], dtype=float)
        self.pipe = ErlangChain.from_state(state["pipe"])
        self.spend = ErlangChain.from_state(state["spend"])
        self.g_e = np.asarray(state["g_e"], dtype=float)
        self.u_s = np.asarray(state["u_s"], dtype=float)
        self.res.smoother = ErlangSmoother.from_state(state["res"])
        self.sales_ma = np.asarray(state["sales_ma"], dtype=float)
        cb = state["cb"]
        self.cb.r_rule = float(cb["r_rule"])
        self.cb.r = float(cb["r"])
        self.cb.pi_e = float(cb["pi_e"])
        self.cb.cpi_hist = [float(x) for x in cb["cpi_hist"]]
        self.cb.core_hist = [float(x) for x in cb["core_hist"]]
        self.cb.month = int(cb["month"])
        self.rate_gap_s = ErlangSmoother.from_state(state["rate_gap_s"])
        self.debt = np.asarray(state["debt"], dtype=float)
        self.prof_s = np.asarray(state["prof_s"], dtype=float)
        self.eb_s = np.asarray(state["eb_s"], dtype=float)
        self.b = float(state["b"])
        self.bank_deposits = float(state.get("bank_deposits", 0.0))
        self.bank_equity = float(state.get("bank_equity", 0.0))
        self.bank_reserves = float(state.get("bank_reserves", 0.0))
        self.bank_gb = float(state.get("bank_gb", 0.0))
        self.cb_gb = float(state.get("cb_gb", 0.0))
        self.hh_gb = float(state.get("hh_gb", self.b))
        self.wealth = float(state["wealth"])
        self.yd_e = float(state["yd_e"])
        self.tau_eff = float(state["tau_eff"])
        self.sh = {k: (np.asarray(v, dtype=float) if isinstance(v, list) else v) for k, v in state["sh"].items()}
        if "bus" in state:
            self.bus.from_state(state["bus"])
            self.sh = self.bus.states
        self.ledger = Ledger.from_state(state["ledger"])
        self.pub = Published.from_state(state["pub"])


def make_real_world(config_dir, seed: int = 0, *, pi_star: float = 0.0, check_sfc: bool = True):
    """World with a single RealEconomy module (R = 1)."""
    from marketsim.core.config import load_config
    from marketsim.layer1.io import load_io, resolve_io_path
    from marketsim.world import World

    cfg = load_config(config_dir)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=check_sfc)
    return World.create(config_dir, seed=seed, modules=[eco])
