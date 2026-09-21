"""Phase-2 parameter sweep and stochastic harness (T2.24)."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy

UNIT_SCALES = (0.0075, 0.01)
SUPPLY_LINE = (0.85, 1.0)
KAPPA_UTIL = (0.6, 1.2)
ANCHOR_GROWTH = (0.5, 0.75)
LEAK_MULT = (0.0, 1.0, 3.0)

STOCH = {
    "dem": (0.90, 0.004),
    "sup": (0.95, 0.0015),
    "cost": (0.93, 0.02),
}


def _leaks(cfg, mult: float) -> dict[str, float]:
    assert cfg.dynamics is not None
    return {k: float(v) * mult for k, v in cfg.dynamics.production.leak_per_month.items()}


def cell_overrides(cfg, unit_scale: float, sl: float, kappa: float, anchor: float, leak_mult: float) -> dict[str, Any]:
    return {
        "edges.capex.unit_scale": unit_scale,
        "edges.capex.supply_line_weight": sl,
        "dynamics.prices.kappa_util": kappa,
        "dynamics.expectations.anchor_growth": anchor,
        "dynamics.production.leak_per_month": _leaks(cfg, leak_mult),
    }


def all_cells(cfg) -> list[dict[str, Any]]:
    cells = []
    for us, sl, ku, ag, lm in product(UNIT_SCALES, SUPPLY_LINE, KAPPA_UTIL, ANCHOR_GROWTH, LEAK_MULT):
        cells.append(cell_overrides(cfg, us, sl, ku, ag, lm))
    return cells


def run_bounded(config_dir: Path, overrides: dict[str, Any], months: int = 120, pi_star: float = 0.0) -> dict[str, float]:
    cfg = load_config(config_dir, overrides)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=False)
    gdp0 = eco.fin.gdp0
    max_gap = 0.0
    for _ in range(months):
        rec = eco.step_month()
        if not np.isfinite(rec["gdp"]):
            return {"finite": 0.0, "max_abs_gap": float("inf")}
        max_gap = max(max_gap, abs(rec["gdp"] / gdp0 - 1.0))
    return {"finite": 1.0, "max_abs_gap": max_gap}


def run_stochastic(
    config_dir: Path,
    seed: int,
    months: int = 1200,
    pi_star: float = 0.02,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """AR(1) demand / supply / ENERGY cost-push; π* = 2 %."""
    cfg = load_config(config_dir, overrides)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=False)
    for k in ("dem", "sup", "cost", "imp"):
        eco.bus.rho[k] = 1.0
    rng = np.random.default_rng(seed)
    energy = eco.codes.index("ENERGY")
    construct = eco.codes.index("CONSTRUCT")
    gdp0 = eco.fin.gdp0
    gaps = np.zeros(months)
    u = np.zeros(months)
    x = np.zeros((months, eco.real.S))
    gdp = np.zeros(months)
    c = np.zeros(months)
    inv = np.zeros(months)
    z_dem = z_sup = z_cost = 0.0
    for t in range(months):
        z_dem = STOCH["dem"][0] * z_dem + rng.normal(0.0, STOCH["dem"][1])
        z_sup = STOCH["sup"][0] * z_sup + rng.normal(0.0, STOCH["sup"][1])
        z_cost = STOCH["cost"][0] * z_cost + rng.normal(0.0, STOCH["cost"][1])
        eco.sh["dem"] = z_dem
        eco.sh["sup"][:] = z_sup
        eco.sh["cost"][energy] = z_cost
        rec = eco.step_month()
        gaps[t] = rec["gdp"] / gdp0 - 1.0
        u[t] = rec["U"]
        x[t] = rec["x"]
        gdp[t] = rec["gdp"]
        fl = eco.last_flows
        assert fl is not None
        c[t] = float((fl.c_nom / np.maximum(eco.p, 1e-12)).sum())
        inv[t] = float((fl.i_nom / np.maximum(eco.p, 1e-12)).sum() + fl.res_nom / max(eco.p[construct], 1e-12))
    yoy = 12.0 * np.log(np.maximum(x[12:], 1e-12) / np.maximum(x[:-12], 1e-12))
    sd_yoy = yoy.std(axis=0)
    rank = np.argsort(-sd_yoy)
    first = float(np.max(np.abs(gaps[:240])))
    last = float(np.max(np.abs(gaps[-240:])))
    return {
        "seed": seed,
        "finite": bool(np.isfinite(gaps).all() and np.isfinite(x).all()),
        "max_abs_gap": float(np.max(np.abs(gaps))),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "envelope_ratio": last / max(first, 1e-12),
        "sd_yoy": {eco.codes[i]: float(sd_yoy[i]) for i in range(len(eco.codes))},
        "rank": [eco.codes[i] for i in rank],
        "sd_i_over_sd_gdp": float(inv.std() / max(gdp.std(), 1e-12)),
        "sd_c_over_sd_gdp": float(c.std() / max(gdp.std(), 1e-12)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-2 sweep: unit_scale × sl × kappa_util × anchor × leak")
    p.add_argument("--config", type=Path, default=Path("config"))
    p.add_argument("--months", type=int, default=120)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    rows = []
    for cell in all_cells(cfg):
        rec = run_bounded(args.config, cell, months=args.months)
        rec["cell"] = {k: v for k, v in cell.items() if k != "dynamics.production.leak_per_month"}
        rec["leak_mult"] = None
        rows.append(rec)
        print(json.dumps({**rec["cell"], "max_abs_gap": rec["max_abs_gap"], "finite": rec["finite"]}))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
