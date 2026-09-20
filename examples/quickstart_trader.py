"""In-process trader quickstart (T7.15 / §7.2).

Creates a world, registers an agent, observes, posts one limit order, and
steps one day. Talks to ``create_app`` through ``HttpClient`` — the same
path the HTTP SDK tests use. Cash is cr; ``qty`` is shares; ticks are days.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from marketsim.api.rest import create_app
from marketsim.api.schemas import AgentRegistration, OrderRequest, WorldSpec
from marketsim.api.ws import include_ws
from marketsim.sdk.client import HttpClient

DEFAULT_SYMBOL = "EQ:NPC:AUTOS"  # first published NPC equity style id
DEFAULT_QTY = 10  # shares
DEFAULT_PRICE = 1.0  # cr / share (baseline index)


def run_trader(
    config_dir: str | Path,
    *,
    seed: int = 7,
    symbol: str = DEFAULT_SYMBOL,
    qty: int = DEFAULT_QTY,
    price: float = DEFAULT_PRICE,
) -> dict[str, Any]:
    """Drive one trader through REST. Returns a JSON-safe summary (unitless ids, cr, days)."""
    app = include_ws(create_app(config_dir=config_dir))
    client = HttpClient(app=app)
    try:
        status = client.create_world(WorldSpec(seed=seed, mode="professional", run_mode="lockstep"))
        acct = client.register_agent(AgentRegistration(agent_id="trader", starting_capital=100.0))
        client.bind(token=acct.token)
        before = client.observe()
        acks = client.submit_orders(
            "trader",
            OrderRequest(agent_id="trader", side="buy", qty=qty, symbol=symbol, price=price),
        )
        after_step = client.step(1)
        after = client.observe()
        return {
            "world_id": status.world_id,
            "account": acct.account,
            "tick_before": before.tick,
            "tick_after": after.tick,
            "order_id": acks[0].order_id,
            "order_status": acks[0].status,
            "state_hash": after_step.state_hash,
            "observe_keys": sorted(after.model_dump().keys()),
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="In-process trader against create_app")
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = run_trader(args.config, seed=args.seed)
    print(summary)


if __name__ == "__main__":
    main()
