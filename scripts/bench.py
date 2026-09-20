"""Daily-tick throughput: real economy (T2.25) and 25-name market book (T6.23)."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from marketsim.core.rng import RngHub
from marketsim.layer1.build_io import CODES
from marketsim.market.background import BackgroundFlow
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FlowCfg
from marketsim.pricing.mispricing import Mispricing
from marketsim.real.economy import make_real_world

# Gate 8: 18 NPC equities + 4 bond/pool names + 3 commodities.
MARKET_EXTRAS: tuple[str, ...] = (
    "GB_BILL",
    "GB_NOTE",
    "GB_BOND",
    "CORP_POOL",
    "OIL",
    "METALS",
    "GRAINS",
)


def market_symbols() -> tuple[str, ...]:
    """25 published names, no agent-firm books."""
    return tuple(f"EQ:NPC:{c}" for c in CODES) + MARKET_EXTRAS


def bench(config_dir: Path, ticks: int, warmup: int, seed: int = 0) -> dict[str, float]:
    world = make_real_world(config_dir, seed=seed, pi_star=0.0, check_sfc=False)
    world.step(warmup)
    t0 = time.perf_counter()
    world.step(ticks)
    dt = time.perf_counter() - t0
    rate = ticks / max(dt, 1e-12)
    return {"ticks": float(ticks), "seconds": dt, "ticks_per_s": rate}


def _book(seed: int) -> tuple[RngHub, BackgroundFlow, Mispricing, ImpactKernel]:
    symbols = market_symbols()
    n = len(symbols)
    rng = RngHub(seed)
    flow = BackgroundFlow(symbols, np.full(n, 100.0), FlowCfg())
    idio = np.full(n, 0.14)
    w = np.full(n, 1.0 / n)
    mis = Mispricing(idio, w)
    kn = ImpactKernel(shape=(n,))
    return rng, flow, mis, kn


def market_step(
    rng: RngHub,
    flow: BackgroundFlow,
    mis: Mispricing,
    kn: ImpactKernel,
) -> np.ndarray:
    """One no-agent market tick. Returns signed volume ``q`` (cr), shape ``(25,)``."""
    q = flow.step(rng, mis.xi)
    xi_i = kn.step(q, flow.adv, 0.02)
    mis.step(rng, impact=xi_i)
    return q


def market_bench(ticks: int, warmup: int, seed: int = 0) -> dict[str, float]:
    """Gate 8: 25 instruments, no agents. ``ticks_per_s`` and a state hash."""
    rng, flow, mis, kn = _book(seed)
    for _ in range(warmup):
        market_step(rng, flow, mis, kn)
    t0 = time.perf_counter()
    last = np.zeros(len(market_symbols()))
    for _ in range(ticks):
        last = market_step(rng, flow, mis, kn)
    dt = time.perf_counter() - t0
    digest = hashlib.sha256(last.tobytes() + mis.xi.tobytes()).hexdigest()
    return {
        "ticks": float(ticks),
        "seconds": dt,
        "ticks_per_s": ticks / max(dt, 1e-12),
        "n_instruments": float(len(market_symbols())),
        "hash": digest,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark World.step and the 25-name market book")
    p.add_argument("--config", type=Path, default=Path("config"))
    p.add_argument("--ticks", type=int, default=5040)
    p.add_argument("--warmup", type=int, default=21)
    p.add_argument("--market", action="store_true", help="T6.23 25-instrument book (no agents)")
    args = p.parse_args()
    if args.market:
        rec = market_bench(args.ticks, args.warmup)
        print(
            f"{rec['n_instruments']:.0f} names, {rec['ticks']:.0f} ticks in "
            f"{rec['seconds']:.3f}s → {rec['ticks_per_s']:.0f} ticks/s"
        )
        return
    rec = bench(args.config, args.ticks, args.warmup)
    print(f"{rec['ticks']:.0f} ticks in {rec['seconds']:.3f}s → {rec['ticks_per_s']:.0f} ticks/s")


if __name__ == "__main__":
    main()
