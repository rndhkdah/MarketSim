"""PettingZoo parallel wrapper (T7.09 / §7.3).

Several agents share one ``World``. Each sees the same factored ``Dict`` action
as ``MarketSimEnv`` (order slots + lever ``Box`` + autopilot mask). Observations
are the T7.08 encoder. Dynamics use the ``World`` seed only — no global RNG,
no wall-clock. Episode length is ``horizon`` days; a bankrupt agent is dropped
from the next step's observations / rewards / dones / infos.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from marketsim.sdk.gym_env import (
    MarketSimEnv,
    RewardMode,
    dividends_cr,
    firm_equity_cr,
    is_bankrupt,
    net_worth_cr,
)
from marketsim.sdk.local import LocalClient
from marketsim.world import FirmAgentModule, World


def _firm_ids(
    agent_ids: Sequence[str],
    firm_ids: Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str]:
    """Map each agent id to a firm id (unitless). Missing keys default to the agent id."""
    if firm_ids is None:
        return {aid: aid for aid in agent_ids}
    if isinstance(firm_ids, Mapping):
        return {aid: str(firm_ids.get(aid, aid)) for aid in agent_ids}
    if len(firm_ids) != len(agent_ids):
        raise ValueError("firm_ids length must match agent_ids")
    return {aid: str(fid) for aid, fid in zip(agent_ids, firm_ids, strict=True)}


class MarketSimParallelEnv(ParallelEnv[str, dict[str, np.ndarray], dict[str, Any]]):
    """Multi-agent PettingZoo env over one ``World`` / ``LocalClient``.

    Reward units match ``MarketSimEnv``: operator cr; trader dimensionless.
    ``terminated`` is bankruptcy; ``truncated`` is ``horizon`` days.
    """

    metadata = {"render_modes": [], "name": "marketsim_parallel_v0"}

    def __init__(
        self,
        config_dir: str | Path,
        *,
        seed: int = 0,
        agent_ids: Sequence[str] = ("alice", "bob"),
        firm_ids: Sequence[str] | Mapping[str, str] | None = None,
        n_order_slots: int = 2,
        instruments: Sequence[str] | None = None,
        horizon: int = 8,
        reward_mode: RewardMode = "trader",
        cell: tuple[str, str] = ("INDUSTRIAL", "AUTOS"),
        modules: Sequence[Any] | None = None,
    ) -> None:
        ids = tuple(str(a) for a in agent_ids)
        if not ids:
            raise ValueError("agent_ids must be non-empty")
        if len(set(ids)) != len(ids):
            raise ValueError("agent_ids must be unique")
        self.config_dir = Path(config_dir)
        self._seed = int(seed)
        self.possible_agents: list[str] = list(ids)
        self.agents: list[str] = []
        self._firm_of = _firm_ids(self.possible_agents, firm_ids)
        self._modules: list[Any] = list(modules) if modules is not None else [FirmAgentModule()]
        self._world: World | None = None
        self._client: LocalClient | None = None
        self.horizon = int(horizon)  # days
        # One Gym view per agent for spaces, decode, and per-agent reward state.
        self._views: dict[str, MarketSimEnv] = {
            aid: MarketSimEnv(
                self.config_dir,
                seed=self._seed,
                agent_id=aid,
                firm_id=self._firm_of[aid],
                n_order_slots=n_order_slots,
                instruments=instruments,
                horizon=horizon,
                reward_mode=reward_mode,
                cell=cell,
                modules=self._modules,
            )
            for aid in self.possible_agents
        }
        self.observation_spaces = {aid: view.observation_space for aid, view in self._views.items()}
        self.action_spaces = {aid: view.action_space for aid, view in self._views.items()}

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def _build(self) -> None:
        self._world = World.create(self.config_dir, seed=self._seed, modules=self._modules)
        self._client = LocalClient(self._world)
        self._bind_views()

    def _bind_views(self) -> None:
        for view in self._views.values():
            view._world = self._world
            view._client = self._client
            view._seed = self._seed

    @property
    def world(self) -> World:
        if self._world is None:
            self._build()
        assert self._world is not None
        return self._world

    @property
    def client(self) -> LocalClient:
        if self._client is None:
            self._build()
        assert self._client is not None
        return self._client

    def idle_action(self, agent: str | None = None) -> dict[str, Any]:
        """No-op orders; every lever masked (autopilot). Deterministic, no RNG."""
        aid = self.possible_agents[0] if agent is None else str(agent)
        return self._views[aid].idle_action()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        """Rewind to tick 0 (days). ``seed`` is the shared ``World`` root seed."""
        del options
        if seed is not None:
            new_seed = int(seed)
            for offset, agent in enumerate(self.possible_agents):
                self.action_space(agent).seed(new_seed + offset)
            if self._world is None or new_seed != self._world.seed:
                self._seed = new_seed
                self._build()
            else:
                self.world.reset()
                self._client = LocalClient(self.world)
                self._bind_views()
        elif self._world is None:
            self._build()
        else:
            self.world.reset()
            self._client = LocalClient(self.world)
            self._bind_views()
        self.agents = self.possible_agents[:]
        observations: dict[str, dict[str, np.ndarray]] = {}
        infos: dict[str, dict[str, Any]] = {}
        for agent_id in self.agents:
            view = self._views[agent_id]
            view._clear_episode()
            raw = self.client.observe(agent_id)
            view._raw = raw
            view._prev_nw = net_worth_cr(raw)
            view._prev_equity = firm_equity_cr(raw, view.firm_id)
            observations[agent_id] = view.encode_observation(raw)
            infos[agent_id] = self._info(raw, firm_id=view.firm_id)
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance one tick (day). Dying agents appear in this return, then drop."""
        if not self.agents:
            return {}, {}, {}, {}, {}
        live = list(self.agents)
        for agent_id in sorted(live):
            self._apply_action(agent_id, actions.get(agent_id))
        self.client.step(1)
        observations: dict[str, dict[str, np.ndarray]] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        still: list[str] = []
        for agent_id in live:
            view = self._views[agent_id]
            raw = self.client.observe(agent_id)
            view._raw = raw
            nw = net_worth_cr(raw)
            equity = firm_equity_cr(raw, view.firm_id)
            div = dividends_cr(raw, view.firm_id)
            reward = view._reward(nw, equity, div)
            view._prev_nw = nw
            view._prev_equity = equity
            terminated = is_bankrupt(raw, net_worth=nw, firm_id=view.firm_id)
            truncated = int(raw.get("tick") or 0) >= self.horizon
            observations[agent_id] = view.encode_observation(raw)
            rewards[agent_id] = float(reward)
            terminations[agent_id] = bool(terminated)
            truncations[agent_id] = bool(truncated)
            infos[agent_id] = self._info(raw, firm_id=view.firm_id)
            if not terminated and not truncated:
                still.append(agent_id)
        self.agents = still
        return observations, rewards, terminations, truncations, infos

    def _apply_action(self, agent_id: str, action: Mapping[str, Any] | None) -> None:
        view = self._views[agent_id]
        act = view.idle_action() if action is None else action
        orders = view.decode_orders(act)
        decision = view.decode_firm(act)
        view.last_orders = orders
        view.last_decision = decision
        if orders:
            self.client.submit_orders(agent_id, orders)
        self.client.submit_decisions(agent_id, decision)

    def _info(self, raw: Mapping[str, Any], *, firm_id: str) -> dict[str, Any]:
        nw = net_worth_cr(raw)
        return {
            "tick": int(raw.get("tick") or 0),  # days
            "net_worth": float(nw),  # cr
            "bankrupt": bool(is_bankrupt(raw, net_worth=nw, firm_id=firm_id)),
        }

    def render(self) -> None:
        """Headless — no frame (same as ``MarketSimEnv``)."""
        return None

    def close(self) -> None:
        self._world = None
        self._client = None
        self.agents = []
        for view in self._views.values():
            view.close()
