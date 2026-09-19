"""Credit gate, collateral index, spreads and rationing exponents (§2.7)."""

from __future__ import annotations

from typing import Any

import numpy as np

from marketsim.core.config import Config, TypedEdge
from marketsim.core.erlang import ErlangSmoother
from marketsim.core.gates import asymmetric_gate, logistic_gate
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.pricing.provider import StubAssetPriceProvider
from marketsim.real.steady_state import FinancialBaseline, RealBaseline

COLLATERAL_ELASTICITY = 0.55
COLLATERAL_UP = 0.6
COLLATERAL_DOWN = 1.8
LAM_SNAP = 1e-12


def capital_gate(c: float, cfg: Config) -> float:
    """Bank-capital logistic gate. Dimensionless; 1 at the 0.125 baseline."""
    g = cfg.edges.credit.bank_capital_gate
    return logistic_gate(
        c,
        midpoint=g.midpoint,
        steepness=g.steepness,
        baseline=g.baseline_capital_ratio,
        normalise_at_baseline=g.normalise_at_baseline,
    )


def collateral_index(
    ln_gap: float,
    elasticity: float = COLLATERAL_ELASTICITY,
    up: float = COLLATERAL_UP,
    down: float = COLLATERAL_DOWN,
) -> float:
    """``Λ_coll = asymmetric_gate(ln(V_RE / V_RE,trend))``. Level-based, no ratchet."""
    return asymmetric_gate(ln_gap, elasticity, up, down)


def corporate_spread(s0: float, s_gate: float, s_loss: float, gate: float, ll_bar: float, ll_bar0: float) -> float:
    """One global corporate spread (D14). Annual decimal."""
    return s0 + s_gate * (1.0 - gate) + s_loss * (ll_bar - ll_bar0)


def _snap_one(x: float) -> float:
    return 1.0 if abs(x - 1.0) < LAM_SNAP else float(x)


def _mean_m(edge: TypedEdge) -> float:
    return 3.0 * float(edge.lag_q)


class CreditBlock:
    """Gate × collateral, per-edge Erlang-smoothed Λ, and the global spread."""

    def __init__(self, cfg: Config, real: RealBaseline, fin: FinancialBaseline, provider: StubAssetPriceProvider) -> None:
        assert cfg.dynamics is not None
        self.cfg = cfg
        self.codes = real.codes
        self.idx = {c: i for i, c in enumerate(real.codes)}
        self.provider = provider
        self.enabled = cfg.dynamics.credit.gate_enabled
        self.s0 = cfg.dynamics.credit.s0
        self.s_gate = cfg.dynamics.credit.s_gate
        self.s_loss = cfg.dynamics.credit.s_loss
        nd = np.array([cfg.sectors.params(c).nd_ebitda for c in real.codes], dtype=float)
        self.ll_bar0 = float((cfg.dynamics.banks.ll0 * (nd / 2.5)).mean())
        self.credit_edges = list(cfg.edges.credit.edges)
        self.edge_s = [ErlangSmoother(e.erlang_k, _mean_m(e), 1.0) for e in self.credit_edges]
        self.lam_tilde: dict[tuple[str, str], float] = {(e.channel, e.dst): 1.0 for e in self.credit_edges}
        coll = cfg.edges.collateral[0] if cfg.edges.collateral else None
        mean_m = 3.0 * (coll.lag_q if coll is not None else 2.0)
        k = coll.erlang_k if coll is not None else 2
        self.lam_s = ErlangSmoother(k, mean_m, 1.0)
        years = cfg.dynamics.credit.collateral_trend_years
        self.trend_tau = max(years * 12.0, 1.0)
        v0 = float(provider.values(fin.r0, fin.pi_star, 0.0)[real.codes.index("REALESTATE")])
        self.v_trend = v0
        self.v_re = v0
        self.lam = 1.0
        self.lam_coll = 1.0
        self.gate = 1.0
        self.spread = self.s0
        self.capital_requirement = cfg.dynamics.banks.capital_target
        self.ltv_cap = 1.0

    def update(
        self,
        *,
        capital: float,
        r: float,
        pi_e: float,
        z_risk: float,
        ll_bar: float,
    ) -> float:
        """Refresh gate, collateral, spread and per-edge ``Λ̃``. Returns ``Λ = g · Λ_coll``."""
        if not self.enabled:
            self.gate = 1.0
            self.lam_coll = 1.0
            self.lam = 1.0
            self.spread = self.s0
            for e, sm in zip(self.credit_edges, self.edge_s, strict=True):
                self.lam_tilde[(e.channel, e.dst)] = float(sm.push(1.0))
            return 1.0
        g = self.cfg.edges.credit.bank_capital_gate
        delta = self.capital_requirement - g.baseline_capital_ratio
        self.gate = logistic_gate(
            capital,
            midpoint=g.midpoint + delta,
            steepness=g.steepness,
            baseline=g.baseline_capital_ratio + delta,
            normalise_at_baseline=g.normalise_at_baseline,
        )
        self.v_re = self.provider.v_re(r, pi_e, z_risk, self.codes)
        self.v_trend += (self.v_re - self.v_trend) / self.trend_tau
        rel = max(self.v_re, 1e-12) / max(self.v_trend, 1e-12)
        gap = 0.0 if abs(rel - 1.0) < LAM_SNAP else float(np.log(rel))
        raw = collateral_index(gap) * float(self.ltv_cap)
        self.lam_coll = _snap_one(float(self.lam_s.push(raw)))
        self.lam = _snap_one(self.gate * self.lam_coll)
        push = 1.0 if self.lam == 1.0 else self.lam
        for e, sm in zip(self.credit_edges, self.edge_s, strict=True):
            self.lam_tilde[(e.channel, e.dst)] = _snap_one(float(sm.push(push)))
        self.spread = corporate_spread(self.s0, self.s_gate, self.s_loss, self.gate, ll_bar, self.ll_bar0)
        return self.lam

    def _edge_lam(self, channel: str, dst: str) -> float:
        return self.lam_tilde.get((channel, dst), self.lam)

    def scale_starts(self, starts: np.ndarray) -> np.ndarray:
        if not self.enabled or self.lam == 1.0:
            return starts
        out = np.asarray(starts, dtype=float).copy()
        generic_e = next((e for e in self.credit_edges if e.channel == "lending_capacity" and e.dst == "ALL_SECTORS"), None)
        if generic_e is not None:
            out = out * self._edge_lam("lending_capacity", "ALL_SECTORS") ** generic_e.elasticity
        for e in self.credit_edges:
            if e.channel == "lending_capacity" and e.dst in self.idx:
                out[self.idx[e.dst]] = starts[self.idx[e.dst]] * self._edge_lam(e.channel, e.dst) ** e.elasticity
        return out

    def scale_residential(self, res: float) -> float:
        if not self.enabled or self.lam == 1.0:
            return res
        e = next((x for x in self.credit_edges if x.channel == "residential"), None)
        if e is None:
            return float(res * self.lam**0.90)
        return float(res * self._edge_lam("residential", e.dst) ** e.elasticity)

    def scale_consumption(self, c: np.ndarray) -> np.ndarray:
        if not self.enabled or self.lam == 1.0:
            return c
        out = np.asarray(c, dtype=float).copy()
        for e in self.credit_edges:
            if e.channel == "hh_credit" and e.dst in self.idx:
                out[self.idx[e.dst]] = c[self.idx[e.dst]] * self._edge_lam(e.channel, e.dst) ** e.elasticity
        return out

    def to_state(self) -> dict[str, Any]:
        return {
            "lam_s": self.lam_s.to_state(),
            "edge_s": [s.to_state() for s in self.edge_s],
            "lam_tilde": {f"{k[0]}|{k[1]}": v for k, v in self.lam_tilde.items()},
            "v_trend": self.v_trend,
            "v_re": self.v_re,
            "lam": self.lam,
            "lam_coll": self.lam_coll,
            "gate": self.gate,
            "spread": self.spread,
            "capital_requirement": self.capital_requirement,
            "ltv_cap": self.ltv_cap,
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.lam_s = ErlangSmoother.from_state(state["lam_s"])
        if "edge_s" in state:
            self.edge_s = [ErlangSmoother.from_state(s) for s in state["edge_s"]]
        if "lam_tilde" in state:
            self.lam_tilde = {}
            for key, val in state["lam_tilde"].items():
                ch, dst = key.split("|", 1)
                self.lam_tilde[(ch, dst)] = float(val)
        self.v_trend = float(state["v_trend"])
        self.v_re = float(state["v_re"])
        self.lam = float(state["lam"])
        self.lam_coll = float(state["lam_coll"])
        self.gate = float(state["gate"])
        self.spread = float(state["spread"])
        if "capital_requirement" in state:
            self.capital_requirement = float(state["capital_requirement"])
        if "ltv_cap" in state:
            self.ltv_cap = float(state["ltv_cap"])


def write_off_bank_equity(
    ledger: Ledger,
    debt: np.ndarray,
    codes: tuple[str, ...],
    fraction: float,
    *,
    tick: int,
    region: int = 0,
) -> np.ndarray:
    """Write off ``fraction`` of bank equity as a pro-rata loan write-off. Returns ``wo`` by firm."""
    from marketsim.ledger.sfc import net_financial_assets

    eq = float(net_financial_assets(ledger)[ledger.entities.id("BANKSYS")])
    amt = max(0.0, fraction * eq)
    total = float(np.maximum(debt, 0.0).sum())
    if amt < 1e-14 or total < 1e-14:
        return np.zeros_like(debt)
    wo = amt * np.maximum(debt, 0.0) / total
    for i, code in enumerate(codes):
        if wo[i] < 1e-14:
            continue
        firm = f"NPC:{region}:{code}"
        ledger.post(
            Tx(tick, "loan_writeoff", (Entry("BANKSYS", "LOAN", -float(wo[i])), Entry(firm, "LOAN", float(wo[i]))))
        )
    return wo
