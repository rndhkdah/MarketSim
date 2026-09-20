"""In-process firm-operator quickstart (T7.15 / §7.2).

Founds a firm, posts a partial lever decision (vacancies only; other levers
stay on autopilot), and reads operator state. Cash is cr; vacancies are persons;
ticks are days.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from marketsim.api.rest import create_app
from marketsim.api.schemas import AgentRegistration, FirmDecision, FoundFirmRequest, WorldSpec
from marketsim.sdk.client import HttpClient

DEFAULT_FIRM = "acme"
DEFAULT_CAPITAL_CR = 50.0
DEFAULT_VACANCIES = 2.0  # persons


def run_operator(
    config_dir: str | Path,
    *,
    seed: int = 7,
    firm_id: str = DEFAULT_FIRM,
    capital: float = DEFAULT_CAPITAL_CR,
    vacancies: float = DEFAULT_VACANCIES,
) -> dict[str, Any]:
    """Drive one operator through REST. Returns a JSON-safe summary."""
    app = create_app(config_dir=config_dir)
    client = HttpClient(app=app)
    try:
        status = client.create_world(WorldSpec(seed=seed, mode="professional", run_mode="lockstep"))
        acct = client.register_agent(AgentRegistration(agent_id="operator", starting_capital=capital))
        client.bind(token=acct.token)
        found = client._http.post(
            client._world_url("/firms"),
            json=FoundFirmRequest(firm_id=firm_id, operator="operator", capital=capital).model_dump(mode="json"),
            headers=client._headers(),
        )
        found.raise_for_status()
        decision = FirmDecision(firm_id=firm_id, operator="operator", vacancies=vacancies)
        echoed = client.submit_decisions("operator", decision)
        live = client._http.get(
            client._world_url(f"/firms/{firm_id}/state"),
            headers=client._headers(),
        )
        live.raise_for_status()
        books = live.json()
        stepped = client.step(1)
        return {
            "world_id": status.world_id,
            "account": acct.account,
            "firm_id": found.json()["firm_id"],
            "operator": books["operator"],
            "live": books["live"],
            "vacancies": echoed[0].vacancies,
            "tick": stepped.tick,
            "state_hash": stepped.state_hash,
        }
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="In-process operator against create_app")
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = run_operator(args.config, seed=args.seed)
    print(summary)


if __name__ == "__main__":
    main()
