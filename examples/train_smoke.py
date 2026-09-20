"""T7.14 / §7.1 gate 1 — PettingZoo multi-agent training smoke.

Four random policies (seeded ``action_space.sample``) and two scripted idle
agents share one ``MarketSimParallelEnv``. Invariant counters (SFC/ledger when
a ledger is present, NaN/inf in observe/step payloads, unexpected exceptions)
must stay zero. Default ``pz_env`` does not attach ``RealEconomy``; World.step
still raises ``SFCError`` if a real-economy module with ``check_sfc`` is added.
Randomness is the World seed plus ``action_space.seed`` — no global RNG.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.core.config import default_config_dir
from marketsim.core.errors import SFCError
from marketsim.ledger.journal import Ledger
from marketsim.ledger.sfc import assert_consistent
from marketsim.sdk.curriculum import episode_metrics
from marketsim.sdk.pz_env import MarketSimParallelEnv

# §7.1 gate 1: 50,000 ticks (days). Do not shorten.
GATE1_TICKS = 50_000
# CI-length loop (days); 8–20 so ``make check`` exercises the example.
SHORT_SMOKE_TICKS = 16
DEFAULT_SEED = 0  # World root seed (unitless)

RANDOM_AGENT_IDS: tuple[str, ...] = ("rand_0", "rand_1", "rand_2", "rand_3")
SCRIPTED_AGENT_IDS: tuple[str, ...] = ("script_0", "script_1")
AGENT_IDS: tuple[str, ...] = RANDOM_AGENT_IDS + SCRIPTED_AGENT_IDS

INVARIANT_KEYS: tuple[str, ...] = (
    "sfc_violations",
    "nan_inf_obs",
    "nan_inf_reward",
    "exceptions",
)


def zero_invariants() -> dict[str, int]:
    """Fresh unitless counters. All stay 0 on a clean run."""
    return {key: 0 for key in INVARIANT_KEYS}


def count_nonfinite(value: Any) -> int:
    """Count NaN/inf in nested mappings, sequences, and numeric arrays (unitless)."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.dtype.kind in {"f", "c"}:
            return int(value.size - int(np.isfinite(value).sum()))
        if value.dtype == object:
            return sum(count_nonfinite(item) for item in value.ravel())
        return 0
    if isinstance(value, (float, np.floating)):
        return 0 if math.isfinite(float(value)) else 1
    if isinstance(value, (int, np.integer, bool, np.bool_, str, bytes)) or value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(count_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_nonfinite(item) for item in value)
    return 0


def iter_ledgers(world: Any) -> list[Ledger]:
    """Ledgers on ``world`` or its modules, if any (cr positions)."""
    found: list[Ledger] = []
    seen: set[int] = set()

    def consider(obj: Any) -> None:
        led = getattr(obj, "ledger", None)
        if isinstance(led, Ledger) and id(led) not in seen:
            seen.add(id(led))
            found.append(led)

    consider(world)
    for mod in getattr(world, "modules", ()) or ():
        consider(mod)
    return found


def check_ledgers(world: Any, counters: dict[str, int]) -> None:
    """Run ``assert_consistent`` on every attached ledger (SFC; cr books)."""
    for ledger in iter_ledgers(world):
        try:
            assert_consistent(ledger)
        except SFCError:
            counters["sfc_violations"] += 1


def make_parallel_env(
    config_dir: str | Path,
    *,
    seed: int,
    ticks: int,
    agent_ids: Sequence[str] = AGENT_IDS,
) -> MarketSimParallelEnv:
    """Gate-1 PettingZoo env. ``ticks`` / horizon are days; ``seed`` is unitless."""
    horizon = max(int(ticks), 2)
    return MarketSimParallelEnv(
        config_dir,
        seed=int(seed),
        agent_ids=tuple(str(aid) for aid in agent_ids),
        horizon=horizon,
    )


def policy_action(env: MarketSimParallelEnv, agent_id: str) -> dict[str, Any]:
    """Factored Dict action. Scripted = idle; random = seeded ``action_space.sample``."""
    if agent_id in SCRIPTED_AGENT_IDS:
        return env.idle_action(agent_id)
    return env.action_space(agent_id).sample()


def collect_actions(env: MarketSimParallelEnv) -> dict[str, dict[str, Any]]:
    """One action per live agent (ids unitless)."""
    return {aid: policy_action(env, aid) for aid in env.agents}


def attach_invariant_hooks(env: MarketSimParallelEnv, counters: dict[str, int]) -> None:
    """Wrap ``reset`` / ``step`` / ``observe`` and count invariant failures (unitless)."""
    orig_reset = env.reset
    orig_step = env.step

    def _wrap_observe() -> None:
        client = env.client
        if getattr(client, "_t714_observe_wrapped", False):
            return
        orig_observe = client.observe

        def observe(agent_id: str | None = None) -> dict[str, Any]:
            raw = orig_observe(agent_id)
            counters["nan_inf_obs"] += count_nonfinite(raw)
            return raw

        client.observe = observe  # type: ignore[method-assign]
        client._t714_observe_wrapped = True

    def reset(
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        try:
            observations, infos = orig_reset(seed=seed, options=options)
        except SFCError:
            counters["sfc_violations"] += 1
            raise
        except Exception:
            counters["exceptions"] += 1
            raise
        _wrap_observe()
        counters["nan_inf_obs"] += count_nonfinite(observations)
        check_ledgers(env.world, counters)
        return observations, infos

    def step(actions: dict[str, Any]) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        try:
            observations, rewards, terminations, truncations, infos = orig_step(actions)
        except SFCError:
            counters["sfc_violations"] += 1
            raise
        except Exception:
            counters["exceptions"] += 1
            raise
        counters["nan_inf_obs"] += count_nonfinite(observations)
        counters["nan_inf_reward"] += count_nonfinite(rewards)
        check_ledgers(env.world, counters)
        return observations, rewards, terminations, truncations, infos

    env.reset = reset  # type: ignore[method-assign]
    env.step = step  # type: ignore[method-assign]


def _record_infos(
    infos: Mapping[str, Mapping[str, Any]],
    net_worths: dict[str, list[float]],
    bankrupt_flags: dict[str, list[bool]],
) -> None:
    """Append published net worth (cr) and bankruptcy flags (unitless)."""
    for agent_id, info in infos.items():
        if agent_id in net_worths:
            net_worths[agent_id].append(float(info.get("net_worth") or 0.0))
            bankrupt_flags[agent_id].append(bool(info.get("bankrupt")))


def run_training_smoke(
    ticks: int,
    seed: int = DEFAULT_SEED,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the six-agent loop for ``ticks`` days.

    Returns a summary dict whose ``invariants`` values are unitless counts.
    ``ticks`` in the summary is simulated days actually stepped.
    """
    if int(ticks) < 1:
        raise ValueError("ticks must be >= 1 (days)")
    config_path = Path(config_dir) if config_dir is not None else default_config_dir()
    counters = zero_invariants()
    ticks_done = 0
    net_worths: dict[str, list[float]] = {aid: [] for aid in AGENT_IDS}
    bankrupt_flags: dict[str, list[bool]] = {aid: [] for aid in AGENT_IDS}
    env: MarketSimParallelEnv | None = None
    hooks_on = False
    try:
        env = make_parallel_env(config_path, seed=int(seed), ticks=int(ticks))
        attach_invariant_hooks(env, counters)
        hooks_on = True
        _obs, infos = env.reset(seed=int(seed))
        _record_infos(infos, net_worths, bankrupt_flags)
        for _ in range(int(ticks)):
            if not env.agents:
                break
            _obs, _rewards, _term, _trunc, infos = env.step(collect_actions(env))
            ticks_done += 1
            _record_infos(infos, net_worths, bankrupt_flags)
    except SFCError:
        if not hooks_on:
            counters["sfc_violations"] += 1
    except Exception:
        if not hooks_on:
            counters["exceptions"] += 1
    finally:
        if env is not None:
            env.close()

    metrics = {
        aid: asdict(
            episode_metrics(
                net_worths[aid],
                scenario_id="train_smoke",
                bankrupt_flags=bankrupt_flags[aid],
            )
        )
        for aid in AGENT_IDS
    }
    return {
        "ticks": ticks_done,  # days
        "requested_ticks": int(ticks),  # days
        "seed": int(seed),
        "config": str(config_path),
        "random_agents": list(RANDOM_AGENT_IDS),
        "scripted_agents": list(SCRIPTED_AGENT_IDS),
        "invariants": counters,
        "metrics": metrics,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI. ``--ticks`` is days; ``--seed`` is the World root seed (unitless)."""
    parser = argparse.ArgumentParser(description="T7.14 multi-agent training smoke (§7.1 gate 1)")
    parser.add_argument("--ticks", type=int, default=GATE1_TICKS, help="episode length (days)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="World root seed (unitless)")
    parser.add_argument("--config", type=Path, default=None, help="config directory")
    args = parser.parse_args(argv)
    summary = run_training_smoke(ticks=args.ticks, seed=args.seed, config_dir=args.config)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
