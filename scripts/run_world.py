"""In-process single-agent loop (T5.18). Price / production / capex on autopilot otherwise."""

from __future__ import annotations

import argparse

from marketsim.core.config import default_config_dir
from marketsim.firms.levers import FirmDecision
from marketsim.world import FirmAgentModule, World


def run(ticks: int, seed: int = 0) -> World:
    world = World.create(default_config_dir(), seed=seed, modules=[FirmAgentModule()])
    for t in range(ticks):
        world.submit(
            "alice",
            FirmDecision(
                firm_id="acme",
                operator="alice",
                posted_price={("INDUSTRIAL", "AUTOS"): 1.0 + 0.001 * (t % 3)},
                target_output={("INDUSTRIAL", "AUTOS"): 5.0},
                expand_capacity={("INDUSTRIAL", "AUTOS"): 0.0},
            ),
        )
        world.step(1)
    return world


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    world = run(args.ticks, args.seed)
    print(world.observe("alice"))


if __name__ == "__main__":
    main()
