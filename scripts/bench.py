"""Daily-tick throughput with the real economy only (T2.25)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from marketsim.real.economy import make_real_world


def bench(config_dir: Path, ticks: int, warmup: int, seed: int = 0) -> dict[str, float]:
    world = make_real_world(config_dir, seed=seed, pi_star=0.0, check_sfc=False)
    world.step(warmup)
    t0 = time.perf_counter()
    world.step(ticks)
    dt = time.perf_counter() - t0
    rate = ticks / max(dt, 1e-12)
    return {"ticks": float(ticks), "seconds": dt, "ticks_per_s": rate}


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark World.step with RealEconomy (R=1)")
    p.add_argument("--config", type=Path, default=Path("config"))
    p.add_argument("--ticks", type=int, default=5040)
    p.add_argument("--warmup", type=int, default=21)
    args = p.parse_args()
    rec = bench(args.config, args.ticks, args.warmup)
    print(f"{rec['ticks']:.0f} ticks in {rec['seconds']:.3f}s → {rec['ticks_per_s']:.0f} ticks/s")


if __name__ == "__main__":
    main()
