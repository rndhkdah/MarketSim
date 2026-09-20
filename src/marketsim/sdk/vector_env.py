"""Spawn-based vector of ``MarketSimEnv`` worlds (T7.12 / §7.3).

Each worker is a child process started with the ``spawn`` context — ``World``
is not fork-safe. Per-worker seeds come from ``SeedSequence(root).spawn(n)``
then ``int(ss.generate_state(1)[0])`` (unitless). Observations are a list of
``MarketSimEnv`` encoder dicts (not stacked); actions are a list of the same
factored Dict. Shared-memory + epoch handshake keeps reset/step off the pipe
(pipe IPC cannot meet the 0.75 scaling bar on this cheap env).
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from marketsim.firms.firm import LEVERS
from marketsim.sdk.gym_env import MarketSimEnv, default_instruments

# Encoder widths (T7.08 / §7.3). SHM layout only — not a price cap.
_N_NEWS = 2
_N_MACRO = 2
_N_FIRM_FEATS = 4
_CMD_RESET = 1
_CMD_STEP = 2
_CMD_CLOSE = 3
# Control-plane liveness poll: every 2**16 spins, not simulated time.
_LIVENESS_MASK = 0xFFFF
_JOIN_S = 10.0  # process teardown timeout (wall seconds, not ticks)


def worker_seeds(root_seed: int, n: int) -> tuple[int, ...]:
    """Per-worker ``World`` seeds (unitless).

    ``np.random.SeedSequence(root_seed).spawn(n)``, then worker ``i`` uses
    ``int(spawned[i].generate_state(1)[0])`` as the ``MarketSimEnv`` seed.
    """
    spawned = np.random.SeedSequence(int(root_seed)).spawn(int(n))
    return tuple(int(ss.generate_state(1)[0]) for ss in spawned)


def _obs_layout(n_inst: int) -> tuple[tuple[tuple[str, int, int], ...], int]:
    """Return ``((key, offset, size), ...), total`` for the encoder (counts)."""
    parts: tuple[tuple[str, int], ...] = (
        ("tick", 1),
        ("net_worth", 1),
        ("portfolio", n_inst * 2),
        ("market", n_inst),
        ("news", _N_NEWS),
        ("macro", _N_MACRO),
        ("firm", _N_FIRM_FEATS),
    )
    layout: list[tuple[str, int, int]] = []
    off = 0
    for key, size in parts:
        layout.append((key, off, size))
        off += size
    return tuple(layout), off


def _one_idle(n_order_slots: int) -> dict[str, Any]:
    """Single no-op Dict action. Mask 0 = autopilot; slot 0 = no order."""
    k = int(n_order_slots)
    firm = {name: np.zeros(1, dtype=np.float32) for name in LEVERS}
    firm["mask"] = np.zeros(len(LEVERS), dtype=np.int8)
    return {
        "orders": {
            "instrument": np.zeros(k, dtype=np.int64),
            "side": np.zeros(k, dtype=np.int64),
            "type": np.zeros(k, dtype=np.int64),
            "size": np.zeros(k, dtype=np.float32),
            "limit_offset": np.zeros(k, dtype=np.float32),
        },
        "firm": firm,
    }


def _as_vec(value: Any, n: int, dtype: np.dtype[Any]) -> np.ndarray:
    arr = np.zeros(n, dtype=dtype)
    if value is None:
        return arr
    raw = np.asarray(value).reshape(-1)
    take = min(int(raw.size), n)
    if take:
        arr[:take] = raw[:take]
    return arr


class _Ctrl:
    """Spawn-context shared buffers. Counts are per-env unless noted."""

    def __init__(self, ctx: Any, n: int, k: int, obs_size: int) -> None:
        n_levers = len(LEVERS)
        self.epoch = ctx.Value("q", 0, lock=False)
        self.cmd = ctx.Value("i", 0, lock=False)
        self.use_idle = ctx.Value("i", 1, lock=False)
        self.done = ctx.Array("q", n, lock=False)
        self.seeds = ctx.Array("Q", n, lock=False)
        self.reward = ctx.Array("d", n, lock=False)
        self.terminated = ctx.Array("i", n, lock=False)
        self.truncated = ctx.Array("i", n, lock=False)
        self.info_tick = ctx.Array("q", n, lock=False)
        self.info_nw = ctx.Array("d", n, lock=False)
        self.info_bankrupt = ctx.Array("i", n, lock=False)
        self.obs = ctx.Array("f", n * obs_size, lock=False)
        self.act_inst = ctx.Array("q", n * k, lock=False)
        self.act_side = ctx.Array("q", n * k, lock=False)
        self.act_type = ctx.Array("q", n * k, lock=False)
        self.act_size = ctx.Array("f", n * k, lock=False)
        self.act_offset = ctx.Array("f", n * k, lock=False)
        self.act_firm = ctx.Array("f", n * n_levers, lock=False)
        self.act_mask = ctx.Array("b", n * n_levers, lock=False)


def _obs_view(ctrl: _Ctrl) -> np.ndarray:
    """float32 view of the shared encoder buffer (counts, not cr)."""
    return np.ctypeslib.as_array(ctrl.obs)


def _write_obs(ctrl: _Ctrl, idx: int, obs: Mapping[str, np.ndarray], layout: Any, obs_size: int) -> None:
    buf = _obs_view(ctrl)
    base = idx * obs_size
    for key, off, size in layout:
        dest = buf[base + off : base + off + size]
        vals = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        n = min(size, int(vals.size))
        dest[:n] = vals[:n]
        if n < size:
            dest[n:] = 0.0


def _write_info(ctrl: _Ctrl, idx: int, info: Mapping[str, Any]) -> None:
    ctrl.info_tick[idx] = int(info.get("tick") or 0)  # days
    ctrl.info_nw[idx] = float(info.get("net_worth") or 0.0)  # cr
    ctrl.info_bankrupt[idx] = int(bool(info.get("bankrupt")))


def _action_is_idle(action: Mapping[str, Any]) -> bool:
    """True if every order slot is empty and every lever is masked (unitless)."""
    orders = action.get("orders") if isinstance(action, Mapping) else None
    firm = action.get("firm") if isinstance(action, Mapping) else None
    orders_m = orders if isinstance(orders, Mapping) else {}
    firm_m = firm if isinstance(firm, Mapping) else {}
    inst = orders_m.get("instrument")
    if inst is not None and bool(np.any(np.asarray(inst) != 0)):
        return False
    mask = firm_m.get("mask")
    return mask is None or not bool(np.any(np.asarray(mask) != 0))


def _unpack_action(ctrl: _Ctrl, idx: int, k: int) -> dict[str, Any]:
    base = idx * k
    n_levers = len(LEVERS)
    mbase = idx * n_levers
    firm = {name: np.array([ctrl.act_firm[mbase + j]], dtype=np.float32) for j, name in enumerate(LEVERS)}
    firm["mask"] = np.array([ctrl.act_mask[mbase + j] for j in range(n_levers)], dtype=np.int8)
    return {
        "orders": {
            "instrument": np.array([ctrl.act_inst[base + j] for j in range(k)], dtype=np.int64),
            "side": np.array([ctrl.act_side[base + j] for j in range(k)], dtype=np.int64),
            "type": np.array([ctrl.act_type[base + j] for j in range(k)], dtype=np.int64),
            "size": np.array([ctrl.act_size[base + j] for j in range(k)], dtype=np.float32),
            "limit_offset": np.array([ctrl.act_offset[base + j] for j in range(k)], dtype=np.float32),
        },
        "firm": firm,
    }


def _worker_main(spec: dict[str, Any], ctrl: _Ctrl, errors: Any) -> None:
    """Child process. Must be spawned — never forked."""
    idx = int(spec["idx"])
    k = int(spec["n_order_slots"])
    layout, obs_size = _obs_layout(int(spec["n_inst"]))
    env = MarketSimEnv(
        spec["config_dir"],
        seed=int(spec["seed"]),
        agent_id=str(spec["agent_id"]),
        firm_id=str(spec["firm_id"]),
        n_order_slots=k,
        instruments=tuple(spec["instruments"]),
        horizon=int(spec["horizon"]),
        reward_mode=spec["reward_mode"],
        cell=tuple(spec["cell"]),
    )
    idle = env.idle_action()
    last = 0
    try:
        while True:
            while int(ctrl.epoch.value) == last:
                pass
            last = int(ctrl.epoch.value)
            cmd = int(ctrl.cmd.value)
            if cmd == _CMD_CLOSE:
                env.close()
                ctrl.done[idx] = last
                return
            if cmd == _CMD_RESET:
                obs, info = env.reset(seed=int(ctrl.seeds[idx]))
                _write_obs(ctrl, idx, obs, layout, obs_size)
                _write_info(ctrl, idx, info)
            elif cmd == _CMD_STEP:
                action = idle if int(ctrl.use_idle.value) else _unpack_action(ctrl, idx, k)
                obs, reward, terminated, truncated, info = env.step(action)
                _write_obs(ctrl, idx, obs, layout, obs_size)
                ctrl.reward[idx] = float(reward)
                ctrl.terminated[idx] = int(terminated)
                ctrl.truncated[idx] = int(truncated)
                _write_info(ctrl, idx, info)
            ctrl.done[idx] = last
    except Exception:
        try:
            errors.put((idx, traceback.format_exc()))
        finally:
            ctrl.done[idx] = last if last else -1


class VectorEnv:
    """N ``MarketSimEnv`` copies in spawn workers (Gymnasium SyncVectorEnv-style).

    ``reset`` / ``step`` are batched. Observations are ``list[dict]`` with the
    T7.08 encoder units (tick days, cash cr, qty shares, prices index). Rewards
    follow ``reward_mode`` (operator: cr; trader: dimensionless).
    """

    def __init__(
        self,
        config_dir: str | Path,
        *,
        n: int = 2,
        seed: int = 0,
        agent_id: str = "alice",
        firm_id: str = "acme",
        n_order_slots: int = 2,
        instruments: Sequence[str] | None = None,
        horizon: int = 8,
        reward_mode: str = "trader",
        cell: tuple[str, str] = ("INDUSTRIAL", "AUTOS"),
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if n_order_slots < 1:
            raise ValueError("n_order_slots must be >= 1")
        self.config_dir = Path(config_dir)
        self.n = int(n)
        self.num_envs = self.n
        self._root_seed = int(seed)
        self.agent_id = str(agent_id)
        self.firm_id = str(firm_id)
        self.n_order_slots = int(n_order_slots)
        self.instruments: tuple[str, ...] = tuple(instruments) if instruments else default_instruments(2)
        self.horizon = int(horizon)  # days
        self.reward_mode = reward_mode
        self.cell: tuple[str, str] = (str(cell[0]), str(cell[1]))
        self.start_method = "spawn"
        self._seeds = worker_seeds(self._root_seed, self.n)
        self._layout, self._obs_size = _obs_layout(len(self.instruments))
        self._closed = False
        self._packed_idle = True  # SHM action slots start at zero
        ctx = mp.get_context("spawn")
        if ctx.get_start_method() != "spawn":
            raise RuntimeError("VectorEnv requires multiprocessing start method 'spawn'")
        self._ctrl = _Ctrl(ctx, self.n, self.n_order_slots, self._obs_size)
        self._errors = ctx.Queue()
        self._procs: list[Any] = []
        for i, wseed in enumerate(self._seeds):
            self._ctrl.seeds[i] = int(wseed)
            spec = {
                "idx": i,
                "config_dir": str(self.config_dir),
                "seed": int(wseed),
                "agent_id": self.agent_id,
                "firm_id": self.firm_id,
                "n_order_slots": self.n_order_slots,
                "instruments": self.instruments,
                "n_inst": len(self.instruments),
                "horizon": self.horizon,
                "reward_mode": self.reward_mode,
                "cell": self.cell,
            }
            proc = ctx.Process(
                target=_worker_main,
                args=(spec, self._ctrl, self._errors),
                name=f"marketsim-vec-{i}",
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)

    @property
    def processes(self) -> tuple[Any, ...]:
        """Worker ``Process`` objects (unitless handles)."""
        return tuple(self._procs)

    @property
    def closed(self) -> bool:
        """True after :meth:`close` (unitless)."""
        return self._closed

    @property
    def seeds(self) -> tuple[int, ...]:
        """Current per-worker ``World`` seeds (unitless)."""
        return self._seeds

    def idle_action(self) -> list[dict[str, Any]]:
        """N no-op Dict actions. Deterministic, no RNG. Same units as ``MarketSimEnv``."""
        return [_one_idle(self.n_order_slots) for _ in range(self.n)]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
        """Reset every worker. ``seed`` is the vector root (unitless). Tick = 0 days."""
        del options
        self._ensure_open()
        if seed is not None:
            self._root_seed = int(seed)
            self._seeds = worker_seeds(self._root_seed, self.n)
        for i, wseed in enumerate(self._seeds):
            self._ctrl.seeds[i] = int(wseed)
        self._issue(_CMD_RESET)
        obs = [self._read_obs(i) for i in range(self.n)]
        info = [self._read_info(i) for i in range(self.n)]
        return obs, info

    def step(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Advance each worker one tick (day). ``actions`` length ``n``."""
        self._ensure_open()
        if len(actions) != self.n:
            raise ValueError(f"expected {self.n} actions, got {len(actions)}")
        idle = all(_action_is_idle(a) for a in actions)
        self._ctrl.use_idle.value = int(idle)
        if not (idle and self._packed_idle):
            for i, action in enumerate(actions):
                self._pack_action(i, action)
            self._packed_idle = idle
        self._issue(_CMD_STEP)
        obs = [self._read_obs(i) for i in range(self.n)]
        reward = np.array([self._ctrl.reward[i] for i in range(self.n)], dtype=np.float64)
        terminated = np.array([bool(self._ctrl.terminated[i]) for i in range(self.n)])
        truncated = np.array([bool(self._ctrl.truncated[i]) for i in range(self.n)])
        info = [self._read_info(i) for i in range(self.n)]
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Signal CLOSE, then ``join`` every worker (seconds on the wall, not ticks)."""
        if self._closed:
            return
        self._closed = True
        try:
            if any(p.is_alive() for p in self._procs):
                self._issue(_CMD_CLOSE)
        except Exception:
            pass
        for proc in self._procs:
            proc.join(timeout=_JOIN_S)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_JOIN_S)

    def __enter__(self) -> VectorEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n}, seed={self._root_seed})"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VectorEnv is closed")

    def _issue(self, cmd: int) -> None:
        self._ctrl.cmd.value = int(cmd)
        self._ctrl.epoch.value = int(self._ctrl.epoch.value) + 1
        self._wait()

    def _wait(self) -> None:
        target = int(self._ctrl.epoch.value)
        spins = 0
        while True:
            if all(int(self._ctrl.done[i]) == target for i in range(self.n)):
                return
            spins += 1
            if spins & _LIVENESS_MASK == 0:
                for i, proc in enumerate(self._procs):
                    if not proc.is_alive() and int(self._ctrl.done[i]) != target:
                        self._raise_worker(i)

    def _raise_worker(self, idx: int) -> None:
        tb = f"worker {idx} exited (code={self._procs[idx].exitcode})"
        try:
            while True:
                wid, text = self._errors.get_nowait()
                if int(wid) == idx:
                    tb = text
                    break
        except Empty:
            pass
        raise RuntimeError(f"vector worker {idx} failed:\n{tb}")

    def _read_obs(self, idx: int) -> dict[str, np.ndarray]:
        buf = _obs_view(self._ctrl)
        base = idx * self._obs_size
        out: dict[str, np.ndarray] = {}
        for key, off, size in self._layout:
            out[key] = np.array(buf[base + off : base + off + size], dtype=np.float32)
        return out

    def _read_info(self, idx: int) -> dict[str, Any]:
        return {
            "tick": int(self._ctrl.info_tick[idx]),  # days
            "net_worth": float(self._ctrl.info_nw[idx]),  # cr
            "bankrupt": bool(self._ctrl.info_bankrupt[idx]),
        }

    def _pack_action(self, idx: int, action: Mapping[str, Any]) -> None:
        orders = action.get("orders") if isinstance(action, Mapping) else None
        firm = action.get("firm") if isinstance(action, Mapping) else None
        orders_m = orders if isinstance(orders, Mapping) else {}
        firm_m = firm if isinstance(firm, Mapping) else {}
        k = self.n_order_slots
        base = idx * k
        inst = _as_vec(orders_m.get("instrument"), k, np.dtype(np.int64))
        side = _as_vec(orders_m.get("side"), k, np.dtype(np.int64))
        typ = _as_vec(orders_m.get("type"), k, np.dtype(np.int64))
        size = _as_vec(orders_m.get("size"), k, np.dtype(np.float32))
        offset = _as_vec(orders_m.get("limit_offset"), k, np.dtype(np.float32))
        for j in range(k):
            self._ctrl.act_inst[base + j] = int(inst[j])
            self._ctrl.act_side[base + j] = int(side[j])
            self._ctrl.act_type[base + j] = int(typ[j])
            self._ctrl.act_size[base + j] = float(size[j])
            self._ctrl.act_offset[base + j] = float(offset[j])
        n_levers = len(LEVERS)
        mbase = idx * n_levers
        mask = _as_vec(firm_m.get("mask"), n_levers, np.dtype(np.int8))
        for j, name in enumerate(LEVERS):
            val = _as_vec(firm_m.get(name), 1, np.dtype(np.float32))
            self._ctrl.act_firm[mbase + j] = float(val[0])
            self._ctrl.act_mask[mbase + j] = int(mask[j])


SyncVectorEnv = VectorEnv
