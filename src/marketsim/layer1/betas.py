"""Pure-function Layer-1 betas (B2). No import-time computation of the table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from marketsim.core.config import Config, load_config
from marketsim.layer1.io import IOTable, load_io, resolve_io_path


@dataclass(frozen=True)
class BetasParams:
    disc_pass: float = 0.35
    refi_share: float = 0.25
    revenue_link: float = 0.12
    da_tax_factor: float = 0.70
    cyclical_leontief: float = 0.55
    cyclical_eta: float = 0.35
    cyclical_invest: float = 0.10
    oil_own: float = 4.80
    oil_cost_scale: float = 1.15
    credit_scale: float = 0.35
    idio_floor: float = 0.10
    idio_scale: float = 0.06
    demo_n: int = 240
    demo_demand_g_vol: float = 0.035
    demo_demand_oil_vol: float = 0.0025
    demo_demand_credit_vol: float = 0.002
    demo_demand_taylor: float = 0.085
    demo_supply_g_vol: float = 0.003
    demo_supply_oil_vol: float = 0.032
    demo_supply_credit_vol: float = 0.003
    demo_supply_taylor: float = 0.58
    demo_supply_g_mean: float = -0.003
    demo_supply_oil_mean: float = 0.010
    demo_idio: float = 0.22
    demo_bond_noise: float = 0.0005


@dataclass(frozen=True)
class Betas:
    codes: tuple[str, ...]
    beta_growth: np.ndarray
    beta_rate: np.ndarray
    beta_rate_discount: np.ndarray
    beta_rate_refi: np.ndarray
    beta_rate_revenue: np.ndarray
    beta_oil: np.ndarray
    beta_credit: np.ndarray
    idio_vol: np.ndarray
    mcap: np.ndarray
    params: BetasParams = field(default_factory=BetasParams)

    def column(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def as_matrix(self) -> np.ndarray:
        """18×6 published columns: growth, rate, oil, credit, idio_vol, mcap."""
        return np.column_stack(
            [
                self.beta_growth,
                self.beta_rate,
                self.beta_oil,
                self.beta_credit,
                self.idio_vol,
                self.mcap,
            ]
        )


def _vec(cfg: Config, key: str) -> np.ndarray:
    return np.array([getattr(cfg.sectors.params(c), key) for c in cfg.codes], dtype=float)


def _credit_elasticities(cfg: Config) -> np.ndarray:
    codes = list(cfg.codes)
    idx = {c: i for i, c in enumerate(codes)}
    el = np.zeros(len(codes))
    for edge in cfg.edges.credit.edges:
        if edge.dst == "ALL_SECTORS":
            el[:] = edge.elasticity
    for edge in cfg.edges.credit.edges:
        if edge.dst in idx:
            el[idx[edge.dst]] = edge.elasticity
    return el


def derive_betas(io: IOTable, cfg: Config, params: BetasParams | None = None) -> Betas:
    """Earnings / valuation loadings on growth, the policy rate (3 channels), oil and credit."""
    p = params or BetasParams()
    codes = tuple(io.codes)
    n = io.n
    x0 = io.L @ io.baseline_final_demand(100.0)
    d_cyc = 0.70 * io.final_demand["HOUSEHOLD"] + 0.30 * io.final_demand["INVESTMENT"]
    rev = (io.L @ d_cyc) / np.maximum(x0, 1e-9)
    rev_z = rev / float(np.mean(rev))

    eta = _vec(cfg, "eta")
    routing = np.zeros(n)
    for code, w in cfg.edges.capex.routing.items():
        routing[io.index[code]] = w
    inv_z = routing / (float(np.mean(routing[routing > 0])) if np.any(routing > 0) else 1.0)

    cyc = p.cyclical_leontief * rev_z + p.cyclical_eta * (eta - 1.0) + p.cyclical_invest * inv_z
    ol = 1.0 / np.maximum(1.0 - _vec(cfg, "fixed_cost"), 0.25)
    beta_growth = p.da_tax_factor * ol * cyc

    duration = _vec(cfg, "cf_duration")
    nd = _vec(cfg, "nd_ebitda")
    dem_rate = _vec(cfg, "dem_rate_semi")
    ch_disc = -p.disc_pass * duration
    ch_refi = -p.refi_share * nd
    ch_rev = p.revenue_link * dem_rate
    beta_rate = ch_disc + ch_refi + ch_rev
    for code, fin in cfg.sectors.financials.items():
        i = io.index[code]
        # nim_rate_beta is the passthrough-mode banking channel (Phase 6).
        row = beta_rate.copy()
        row[i] = row[i] + fin.nim_rate_beta + fin.float_rate_beta - p.disc_pass * 0.25 * fin.bond_mtm_duration
        beta_rate = row

    energy = io.index["ENERGY"]
    cost = io.G[:, energy]
    pt = _vec(cfg, "pass_through")
    va = np.maximum(1.0 - io.mu, 0.15)
    beta_oil = -p.oil_cost_scale * p.da_tax_factor * cost * (1.0 - 0.40 * pt) / va
    oil = beta_oil.copy()
    oil[energy] += p.oil_own
    beta_oil = oil

    el = _credit_elasticities(cfg)
    beta_credit = -p.credit_scale * el * np.maximum(nd, 0.2)

    idio = p.idio_floor + p.idio_scale * (np.abs(beta_growth) + 0.25 * np.abs(beta_oil))
    mcap = np.array([cfg.sectors.market_cap_weights[c] for c in codes], dtype=float)
    mcap = mcap / mcap.sum()

    return Betas(
        codes=codes,
        beta_growth=np.asarray(beta_growth, dtype=float),
        beta_rate=np.asarray(beta_rate, dtype=float),
        beta_rate_discount=np.asarray(ch_disc, dtype=float),
        beta_rate_refi=np.asarray(ch_refi, dtype=float),
        beta_rate_revenue=np.asarray(ch_rev, dtype=float),
        beta_oil=np.asarray(beta_oil, dtype=float),
        beta_credit=np.asarray(beta_credit, dtype=float),
        idio_vol=np.asarray(idio, dtype=float),
        mcap=mcap,
        params=p,
    )


def _paths(
    rng: np.random.Generator,
    n: int,
    g_vol: float,
    oil_vol: float,
    credit_vol: float,
    taylor: float,
    g_mean: float = 0.0,
    oil_mean: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = g_mean + g_vol * rng.standard_normal(n)
    oil = oil_mean + oil_vol * rng.standard_normal(n)
    credit = credit_vol * rng.standard_normal(n)
    rate = taylor * (g + 0.25 * oil)
    return g, oil, credit, rate


def _equity_bond(
    betas: Betas,
    g: np.ndarray,
    oil: np.ndarray,
    credit: np.ndarray,
    rate: np.ndarray,
    rng: np.random.Generator,
    params: BetasParams,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = g.shape[0]
    s = len(betas.codes)
    r = (
        betas.beta_growth[:, None] * g[None, :]
        + betas.beta_rate[:, None] * rate[None, :]
        + betas.beta_oil[:, None] * oil[None, :]
        + betas.beta_credit[:, None] * credit[None, :]
        + params.demo_idio * betas.idio_vol[:, None] * rng.standard_normal((s, n))
    )
    bond = -duration * rate + params.demo_bond_noise * rng.standard_normal(n)
    index = betas.mcap @ r
    return r, index, bond


@dataclass(frozen=True)
class RegimeDemo:
    demand_corr: float
    supply_corr: float
    demand_sector_corr: np.ndarray
    supply_sector_corr: np.ndarray
    n_sign_flips: int


def regime_demo(betas: Betas, seed: int, params: BetasParams | None = None, duration: float = 7.0) -> RegimeDemo:
    """Factor Monte-Carlo of equity vs the duration-7 bond index.

    Demand: growth shocks, Taylor rates up → stocks and bonds move opposite.
    Supply: oil / cost-push, inflation-driven rates → both risk assets sell off.
    """
    p = params or betas.params
    rng = np.random.default_rng(int(seed))
    n = p.demo_n
    g, oil, credit, rate = _paths(
        rng, n, p.demo_demand_g_vol, p.demo_demand_oil_vol, p.demo_demand_credit_vol, p.demo_demand_taylor
    )
    r_d, idx_d, b_d = _equity_bond(betas, g, oil, credit, rate, rng, p, duration)
    g, oil, credit, rate = _paths(
        rng,
        n,
        p.demo_supply_g_vol,
        p.demo_supply_oil_vol,
        p.demo_supply_credit_vol,
        p.demo_supply_taylor,
        g_mean=p.demo_supply_g_mean,
        oil_mean=p.demo_supply_oil_mean,
    )
    r_s, idx_s, b_s = _equity_bond(betas, g, oil, credit, rate, rng, p, duration)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    dem = np.array([_corr(r_d[i], b_d) for i in range(len(betas.codes))])
    sup = np.array([_corr(r_s[i], b_s) for i in range(len(betas.codes))])
    flips = int(np.sum((dem < 0) & (sup > 0)))
    return RegimeDemo(
        demand_corr=_corr(idx_d, b_d),
        supply_corr=_corr(idx_s, b_s),
        demand_sector_corr=dem,
        supply_sector_corr=sup,
        n_sign_flips=flips,
    )


def render_betas_md(betas: Betas) -> str:
    cols = ("beta_growth", "beta_rate", "beta_oil", "beta_credit", "idio_vol", "mcap")
    lines = [
        "# Derived Layer-1 betas",
        "",
        "Reconstructed (no legacy bit-identical file). Units: growth / oil / credit are",
        "log-value per unit factor; `beta_rate` is log-value per 100 bp of the policy rate.",
        "",
        "| code | growth | rate | oil | credit | idio_vol | mcap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, code in enumerate(betas.codes):
        rows = [
            f"{betas.beta_growth[i]:+.4f}",
            f"{betas.beta_rate[i]:+.4f}",
            f"{betas.beta_oil[i]:+.4f}",
            f"{betas.beta_credit[i]:+.4f}",
            f"{betas.idio_vol[i]:.4f}",
            f"{betas.mcap[i]:.4f}",
        ]
        lines.append(f"| {code} | " + " | ".join(rows) + " |")
    lines.append("")
    lines.append(f"Columns: {', '.join(cols)}.")
    return "\n".join(lines) + "\n"


def write_artifacts(betas: Betas, config_dir: Path) -> None:
    np.savez(
        config_dir / "betas.npz",
        codes=np.array(betas.codes),
        beta_growth=betas.beta_growth,
        beta_rate=betas.beta_rate,
        beta_oil=betas.beta_oil,
        beta_credit=betas.beta_credit,
        idio_vol=betas.idio_vol,
        mcap=betas.mcap,
    )
    (config_dir / "betas.md").write_text(render_betas_md(betas))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive Layer-1 betas (pure functions)")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir or (Path(__file__).resolve().parents[3] / "config"))
    io = load_io(resolve_io_path(cfg))
    betas = derive_betas(io, cfg)
    write_artifacts(betas, cfg.config_dir)
    demo = regime_demo(betas, seed=7)
    print(f"wrote {cfg.config_dir / 'betas.md'} and betas.npz")
    print(
        f"regime demo seed 7: demand {demo.demand_corr:+.2f}  "
        f"supply {demo.supply_corr:+.2f}  flips {demo.n_sign_flips}/18"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
