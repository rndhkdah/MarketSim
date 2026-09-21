"""Daily-tick throughput: real economy (T2.25), 25-name market book (T6.23), Phase 7 gate 3 (T7.13)."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import os
import platform
import pstats
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.core.config import load_config
from marketsim.core.module import Phase
from marketsim.core.rng import RngHub
from marketsim.firms.levers import FirmDecision
from marketsim.layer1.build_io import CODES
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.market.background import BackgroundFlow
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FlowCfg
from marketsim.pricing.mispricing import Mispricing
from marketsim.real.economy import RealEconomy, make_real_world
from marketsim.sdk.local import LocalClient
from marketsim.world import FirmAgentModule, World

# Gate 8 / T6.23: 18 NPC equities + 4 bond/pool names + 3 commodities.
MARKET_EXTRAS: tuple[str, ...] = (
    "GB_BILL",
    "GB_NOTE",
    "GB_BOND",
    "CORP_POOL",
    "OIL",
    "METALS",
    "GRAINS",
)

# 07-phase7-api-sdk.md §7.1 gate 3 floors (ticks / s).
GATE3_SMALL_TICKS_PER_S = 1000.0
GATE3_AGENTS_TICKS_PER_S = 300.0
GATE3_N_AGENTS = 8  # scripted idle agents (count)
# Default REST/WS comparison length (days). Same n on LocalClient / REST / WS.
DEFAULT_TRANSPORT_TICKS = 252


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


def configured_regions(config_dir: Path) -> int:
    """Region count from ``regions.yaml`` (count). Shipped bundle is R = 3."""
    cfg = load_config(config_dir)
    if cfg.regions is None:
        return 1
    return len(cfg.regions.regions)


class MarketBookModule:
    """25-name no-agent book on ``World.step`` (T7.13 / T6.23). ``q`` is cr."""

    name = "market_book"

    def __init__(self) -> None:
        self.flow: BackgroundFlow | None = None
        self.mis: Mispricing | None = None
        self.kn: ImpactKernel | None = None
        self.last_q: np.ndarray = np.zeros(0)

    def reset(self, ctx: Any) -> None:
        symbols = market_symbols()
        n = len(symbols)
        self.flow = BackgroundFlow(symbols, np.full(n, 100.0), FlowCfg())
        self.mis = Mispricing(np.full(n, 0.14), np.full(n, 1.0 / n))
        self.kn = ImpactKernel(shape=(n,))
        self.last_q = np.zeros(n)
        ctx.world.rng.stream("flow")

    def on_phase(self, ctx: Any, phase: Phase) -> None:
        if phase is not Phase.MARKET:
            return
        assert self.flow is not None and self.mis is not None and self.kn is not None
        self.last_q = market_step(ctx.world.rng, self.flow, self.mis, self.kn)

    def to_state(self) -> dict[str, Any]:
        xi = np.zeros(0) if self.mis is None else self.mis.xi
        return {"q": self.last_q.copy(), "xi": np.asarray(xi, dtype=float).copy()}

    def from_state(self, state: dict[str, Any]) -> None:
        self.last_q = np.asarray(state["q"], dtype=float)
        # ``Mispricing.xi`` is derived (I + s + n); restore impact only.
        if self.mis is not None and "xi" in state:
            self.mis.last_impact = np.asarray(state["xi"], dtype=float)


def make_small_world(config_dir: Path, seed: int = 0, *, with_real: bool = True) -> World:
    """In-process small world: 25-name book, no agents.

    ``regions.yaml`` is R = 3. Live ``RealEconomy.step_month`` stays R = 1
    (QUESTIONS T3.15). ``with_real`` attaches the monthly engine (SFC off).
    """
    modules: list[Any] = [MarketBookModule()]
    if with_real:
        cfg = load_config(config_dir)
        io = load_io(resolve_io_path(cfg))
        modules.insert(0, RealEconomy(cfg, io, pi_star=0.0, check_sfc=False))
    return World.create(config_dir, seed=seed, modules=modules)


def small_world_bench(
    config_dir: Path,
    ticks: int,
    warmup: int,
    seed: int = 0,
    *,
    with_real: bool = True,
) -> dict[str, float]:
    """Gate 3 line 1. ``ticks`` / ``warmup`` are days; ``ticks_per_s`` is 1/s."""
    n_r = configured_regions(config_dir)
    world = make_small_world(config_dir, seed=seed, with_real=with_real)
    live_r = 1.0
    for mod in world.modules:
        if isinstance(mod, RealEconomy):
            live_r = float(mod.R)
    world.step(warmup)
    t0 = time.perf_counter()
    world.step(ticks)
    dt = time.perf_counter() - t0
    return {
        "ticks": float(ticks),
        "seconds": dt,
        "ticks_per_s": ticks / max(dt, 1e-12),
        "n_regions_config": float(n_r),
        "n_regions_live": live_r,
        "n_instruments": float(len(market_symbols())),
        "n_agents": 0.0,
        "with_real": 1.0 if with_real else 0.0,
    }


def _idle_decision(agent_id: str) -> FirmDecision:
    """Autopilot firm decision (all levers omitted). Ids are unitless."""
    return FirmDecision(firm_id=agent_id, operator=agent_id)


def scripted_agents_bench(
    config_dir: Path,
    ticks: int,
    warmup: int,
    seed: int = 0,
    n_agents: int = GATE3_N_AGENTS,
    *,
    with_real: bool = True,
) -> dict[str, float]:
    """Gate 3 line 2: ``n_agents`` idle submits then ``World.step`` (days)."""
    world = make_small_world(config_dir, seed=seed, with_real=with_real)
    world.modules.append(FirmAgentModule())
    world.reset()
    client = LocalClient(world)
    agents = tuple(f"agent{i:02d}" for i in range(int(n_agents)))

    def _round() -> None:
        for aid in agents:
            client.submit_decisions(aid, _idle_decision(aid))
        client.step(1)

    for _ in range(warmup):
        _round()
    t0 = time.perf_counter()
    for _ in range(ticks):
        _round()
    dt = time.perf_counter() - t0
    return {
        "ticks": float(ticks),
        "seconds": dt,
        "ticks_per_s": ticks / max(dt, 1e-12),
        "n_agents": float(n_agents),
        "n_instruments": float(len(market_symbols())),
        "n_regions_config": float(configured_regions(config_dir)),
    }


def pz_agents_bench(
    config_dir: Path,
    ticks: int,
    warmup: int,
    seed: int = 0,
    n_agents: int = GATE3_N_AGENTS,
) -> dict[str, float]:
    """Optional SDK path: PettingZoo env, ``n_agents`` idle Dict actions / day."""
    from marketsim.sdk.pz_env import MarketSimParallelEnv

    ids = tuple(f"agent{i:02d}" for i in range(int(n_agents)))
    env = MarketSimParallelEnv(
        config_dir,
        seed=seed,
        agent_ids=ids,
        horizon=int(warmup) + int(ticks) + 2,
        n_order_slots=2,
        modules=[MarketBookModule(), FirmAgentModule()],
    )
    env.reset(seed=seed)
    idle = {aid: env.idle_action(aid) for aid in env.possible_agents}
    for _ in range(warmup):
        env.step(idle)
    t0 = time.perf_counter()
    for _ in range(ticks):
        env.step(idle)
    dt = time.perf_counter() - t0
    env.close()
    return {
        "ticks": float(ticks),
        "seconds": dt,
        "ticks_per_s": ticks / max(dt, 1e-12),
        "n_agents": float(n_agents),
    }


def transport_bench(
    config_dir: Path,
    ticks: int,
    warmup: int,
    seed: int = 0,
) -> dict[str, float]:
    """REST / WS vs ``LocalClient.step`` on the same WorldManager world (days).

    Uses Starlette ``TestClient`` (in-process ASGI, no TCP). ``WorldManager``
    worlds have no extra modules — this isolates transport + ``WorldStatus`` hash.
    """
    from fastapi.testclient import TestClient

    from marketsim.api.rest import create_app, world_admin_token
    from marketsim.api.sessions import WorldManager
    from marketsim.api.ws import include_ws

    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    include_ws(app, manager=mgr)
    with TestClient(app) as http:
        created = http.post(
            "/v1/worlds",
            json={"seed": int(seed), "mode": "professional", "run_mode": "lockstep"},
        )
        created.raise_for_status()
        status = created.json()
        wid = str(status["world_id"])
        admin = world_admin_token(wid, int(status["seed"]))
        admin_h = {"Authorization": f"Bearer {admin}"}
        acct = http.post(
            f"/v1/worlds/{wid}/agents",
            json={"agent_id": "alice", "role": "agent", "starting_capital": 0.0},
            headers=admin_h,
        )
        acct.raise_for_status()
        agent_h = {"Authorization": f"Bearer {acct.json()['token']}"}
        local = LocalClient(mgr.world(wid))
        local.step(warmup)

        t0 = time.perf_counter()
        for _ in range(ticks):
            local.step(1)
        dt_local = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(ticks):
            resp = http.post(f"/v1/worlds/{wid}/step", json={"n": 1}, headers=admin_h)
            resp.raise_for_status()
        dt_rest = time.perf_counter() - t0

        hub = app.state.ws_hub
        with http.websocket_connect(f"/v1/worlds/{wid}/stream", headers=agent_h) as ws:
            first = ws.receive_json()
            if first.get("type") != "subscribed":
                raise RuntimeError(f"expected subscribed, got {first!r}")
            t0 = time.perf_counter()
            for _ in range(ticks):
                local.step(1)
                tick = int(mgr.world(wid).clock.tick)
                hub.publish(wid, "tick", {"tick": tick}, tick=tick)
                msg = ws.receive_json()
                if msg.get("type") != "tick":
                    raise RuntimeError(f"expected tick event, got {msg!r}")
            dt_ws = time.perf_counter() - t0

    local_tps = ticks / max(dt_local, 1e-12)
    rest_tps = ticks / max(dt_rest, 1e-12)
    ws_tps = ticks / max(dt_ws, 1e-12)
    return {
        "ticks": float(ticks),
        "local_seconds": dt_local,
        "rest_seconds": dt_rest,
        "ws_seconds": dt_ws,
        "local_ticks_per_s": local_tps,
        "rest_ticks_per_s": rest_tps,
        "ws_ticks_per_s": ws_tps,
        "rest_overhead_x": dt_rest / max(dt_local, 1e-12),
        "ws_overhead_x": dt_ws / max(dt_local, 1e-12),
        "rest_extra_us_per_tick": 1.0e6 * (dt_rest - dt_local) / max(ticks, 1),
        "ws_extra_us_per_tick": 1.0e6 * (dt_ws - dt_local) / max(ticks, 1),
    }


def profile_top(
    func: Callable[..., Any],
    *args: Any,
    n: int = 5,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """cProfile ``func``. ``tottime_s`` / ``cumtime_s`` are wall seconds."""
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    stats = pstats.Stats(pr)
    ranked = sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)
    rows: list[dict[str, Any]] = []
    for (filename, line, name), (_cc, nc, tt, ct, _callers) in ranked[:n]:
        rows.append(
            {
                "function": f"{filename}:{line}({name})",
                "ncalls": float(nc),
                "tottime_s": float(tt),
                "cumtime_s": float(ct),
            }
        )
    return rows


def machine_note() -> str:
    """Host label for the report (not sim state)."""
    cpu = platform.processor() or platform.machine()
    return (
        f"{platform.system()} {platform.release()} · {cpu} · "
        f"{os.cpu_count() or 0} CPUs · CPython {platform.python_version()}"
    )


def _print_rec(label: str, rec: dict[str, Any]) -> None:
    tps = rec.get("ticks_per_s")
    if tps is not None:
        print(
            f"{label}: {rec.get('ticks', 0):.0f} ticks in {rec.get('seconds', 0):.3f}s "
            f"→ {float(tps):.0f} ticks/s"
        )
        return
    print(
        f"{label}: local {rec['local_ticks_per_s']:.0f} ticks/s · "
        f"REST {rec['rest_ticks_per_s']:.0f} ticks/s "
        f"({rec['rest_overhead_x']:.1f}×, +{rec['rest_extra_us_per_tick']:.0f} µs/tick) · "
        f"WS {rec['ws_ticks_per_s']:.0f} ticks/s "
        f"({rec['ws_overhead_x']:.1f}×, +{rec['ws_extra_us_per_tick']:.0f} µs/tick)"
    )


def format_gate3_report(
    *,
    small: dict[str, float],
    agents: dict[str, float],
    transport: dict[str, float],
    pz: dict[str, float] | None,
    hotspots: list[dict[str, Any]],
    profiled: str | None,
    warmup: int,
) -> str:
    """Markdown for ``claude/plan/reports/phase7-throughput.md``. Units: ticks/s, days."""
    small_ok = float(small["ticks_per_s"]) >= GATE3_SMALL_TICKS_PER_S
    agents_ok = float(agents["ticks_per_s"]) >= GATE3_AGENTS_TICKS_PER_S
    gate_ok = small_ok and agents_ok
    live_r = int(small.get("n_regions_live", 1))
    cfg_r = int(small.get("n_regions_config", 3))
    lines = [
        "# Phase 7 throughput (T7.13 / gate 3)",
        "",
        f"Measured {time.strftime('%Y-%m-%d')}. {machine_note()}.",
        "Wall time is `time.perf_counter` (measurement only; not sim state).",
        "",
        "Small world is `World.step` with the shipped `regions.yaml` (R = 3), the",
        "T6.23 25-name book (`MarketBookModule` on `Phase.MARKET`), and the monthly",
        f"`RealEconomy` (SFC off). Live monthly stepper is R = {live_r} (QUESTIONS T3.15);",
        f"configured geometry is R = {cfg_r}. No agents on line 1.",
        "",
        "## Gate 3",
        "",
        "| line | spec floor | measured | ticks (days) | warmup (days) | wall | result |",
        "|---|---|---|---|---|---|---|",
        (
            f"| small world (R_cfg = {cfg_r}, R_live = {live_r}, "
            f"{int(small['n_instruments'])} names, 0 agents) "
            f"| ≥ {GATE3_SMALL_TICKS_PER_S:.0f} ticks/s "
            f"| **{small['ticks_per_s']:.0f}** ticks/s "
            f"| {small['ticks']:.0f} | {warmup} | {small['seconds']:.3f} s "
            f"| {'pass' if small_ok else 'MISS'} |"
        ),
        (
            f"| {int(agents['n_agents'])} scripted idle agents "
            f"| ≥ {GATE3_AGENTS_TICKS_PER_S:.0f} ticks/s "
            f"| **{agents['ticks_per_s']:.0f}** ticks/s "
            f"| {agents['ticks']:.0f} | {warmup} | {agents['seconds']:.3f} s "
            f"| {'pass' if agents_ok else 'MISS'} |"
        ),
        "",
        (
            f"Gate 3 (engine lines): **{'passed' if gate_ok else 'missed'}**."
            " REST/WS has no ticks/s floor — see below."
        ),
        "",
        "Eight-agent line: each tick submits one autopilot `FirmDecision` per agent",
        "(`LocalClient.submit_decisions`) then `World.step(1)`. Same 25-name book.",
        "",
    ]
    if pz is not None:
        lines.extend(
            [
                "SDK wrapper (not the gate line; book + FirmAgentModule only, no monthly",
                f"engine): `MarketSimParallelEnv` with the same {int(pz['n_agents'])} idle Dict actions.",
                "",
                "| path | measured | ticks (days) | wall |",
                "|---|---|---|---|",
                (
                    f"| SDK ParallelEnv | **{pz['ticks_per_s']:.0f}** ticks/s "
                    f"| {pz['ticks']:.0f} | {pz['seconds']:.3f} s |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## REST / WS overhead",
            "",
            "Same `n` ticks on one `WorldManager` world (empty module list — that is",
            "what `POST /v1/worlds` builds today). Starlette `TestClient` / in-process",
            "ASGI; no TCP. REST `POST …/step` returns `WorldStatus` (includes",
            "`state_hash`). WS path is `LocalClient.step(1)` + `StreamHub.publish`",
            "(tick) + one `receive_json`. T7.05 does not auto-publish on `World.step`.",
            "",
            "| path | ticks (days) | wall | ticks/s | vs LocalClient | extra µs / tick |",
            "|---|---|---|---|---|---|",
            (
                f"| LocalClient.step(1) | {transport['ticks']:.0f} "
                f"| {transport['local_seconds']:.6f} s "
                f"| **{transport['local_ticks_per_s']:.0f}** | 1.0× | 0 |"
            ),
            (
                f"| REST POST /step n=1 | {transport['ticks']:.0f} "
                f"| {transport['rest_seconds']:.3f} s "
                f"| **{transport['rest_ticks_per_s']:.0f}** "
                f"| {transport['rest_overhead_x']:.1f}× "
                f"| {transport['rest_extra_us_per_tick']:.0f} |"
            ),
            (
                f"| WS (step + publish + recv) | {transport['ticks']:.0f} "
                f"| {transport['ws_seconds']:.3f} s "
                f"| **{transport['ws_ticks_per_s']:.0f}** "
                f"| {transport['ws_overhead_x']:.1f}× "
                f"| {transport['ws_extra_us_per_tick']:.0f} |"
            ),
            "",
        ]
    )
    if hotspots:
        lines.extend(
            [
                f"## cProfile hot spots ({profiled})",
                "",
                "Top 5 by cumulative seconds. Proposed fixes only — no Rust port (T9.05).",
                "",
                "| rank | function | ncalls | tottime (s) | cumtime (s) |",
                "|---|---|---|---|---|",
            ]
        )
        for i, row in enumerate(hotspots, start=1):
            lines.append(
                f"| {i} | `{row['function']}` | {row['ncalls']:.0f} "
                f"| {row['tottime_s']:.4f} | {row['cumtime_s']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Proposed fixes (if a floor was missed):",
                "",
                "1. Keep SFC and debug journal off on the throughput path (already off here).",
                "2. Avoid `World.state_hash()` on every REST `step` — hash on demand / observe.",
                "3. Cache `WorldStatus` fields that do not change intra-tick.",
                "4. Vectorise leftover Python in `MarketBookModule` callers; do not loop sectors.",
                "5. For the SDK env, skip unused observation encode on idle scripted agents.",
                "",
            ]
        )
    lines.extend(
        [
            "## Method",
            "",
            "- Warmup ticks are discarded; timed region is `time.perf_counter` around `step`.",
            "- Randomness in the book uses `ctx.world.rng` / `stream(\"flow\")`.",
            "- Existing `bench()` (T2.25) and `market_bench()` (T6.23) are unchanged.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_gate3(
    config_dir: Path,
    ticks: int,
    warmup: int,
    seed: int,
    n_agents: int,
    transport_ticks: int,
    *,
    with_pz: bool = True,
    force_profile: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Run all gate-3 benches. Returns `(markdown, records)`."""
    small = small_world_bench(config_dir, ticks, warmup, seed)
    agents = scripted_agents_bench(config_dir, ticks, warmup, seed, n_agents=n_agents)
    pz: dict[str, float] | None = None
    if with_pz:
        pz = pz_agents_bench(config_dir, ticks, warmup, seed, n_agents=n_agents)
    transport = transport_bench(config_dir, transport_ticks, min(warmup, 21), seed)
    _print_rec("small world", small)
    _print_rec(f"{n_agents} scripted agents", agents)
    if pz is not None:
        _print_rec("sdk ParallelEnv", pz)
    _print_rec("transport", transport)
    small_ok = float(small["ticks_per_s"]) >= GATE3_SMALL_TICKS_PER_S
    agents_ok = float(agents["ticks_per_s"]) >= GATE3_AGENTS_TICKS_PER_S
    hotspots: list[dict[str, Any]] = []
    profiled: str | None = None
    if not small_ok or force_profile:
        profiled = "small_world_bench"
        hotspots = profile_top(small_world_bench, config_dir, ticks, warmup, seed)
    if not agents_ok:
        profiled = "scripted_agents_bench"
        hotspots = profile_top(
            scripted_agents_bench, config_dir, ticks, warmup, seed, n_agents
        )
    md = format_gate3_report(
        small=small,
        agents=agents,
        transport=transport,
        pz=pz,
        hotspots=hotspots,
        profiled=profiled,
        warmup=warmup,
    )
    records = {"small": small, "agents": agents, "pz": pz, "transport": transport}
    print("GATE 3", "PASS" if small_ok and agents_ok else "MISS")
    return md, records


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark World.step and the 25-name market book")
    p.add_argument("--config", type=Path, default=Path("config"))
    p.add_argument("--ticks", type=int, default=5040)
    p.add_argument("--warmup", type=int, default=21)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--market", action="store_true", help="T6.23 25-instrument book (no agents)")
    p.add_argument("--gate3", action="store_true", help="T7.13 Phase 7 gate 3 suite")
    p.add_argument("--small-world", action="store_true", help="T7.13 small-world World.step only")
    p.add_argument("--scripted", action="store_true", help="T7.13 8 scripted agents only")
    p.add_argument("--transport", action="store_true", help="T7.13 REST/WS overhead only")
    p.add_argument("--agents", type=int, default=GATE3_N_AGENTS, help="scripted agent count")
    p.add_argument(
        "--transport-ticks",
        type=int,
        default=DEFAULT_TRANSPORT_TICKS,
        help="REST/WS comparison ticks (days)",
    )
    p.add_argument("--profile", action="store_true", help="cProfile top-5 (always, with --gate3)")
    p.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="write phase7-throughput.md (implies --gate3)",
    )
    args = p.parse_args()
    if args.market:
        rec = market_bench(args.ticks, args.warmup, seed=args.seed)
        print(
            f"{rec['n_instruments']:.0f} names, {rec['ticks']:.0f} ticks in "
            f"{rec['seconds']:.3f}s → {rec['ticks_per_s']:.0f} ticks/s"
        )
        return
    if args.write_report is not None or args.gate3:
        md, _recs = run_gate3(
            args.config,
            args.ticks,
            args.warmup,
            args.seed,
            args.agents,
            args.transport_ticks,
            force_profile=args.profile,
        )
        if args.write_report is not None:
            args.write_report.parent.mkdir(parents=True, exist_ok=True)
            args.write_report.write_text(md, encoding="utf-8")
            print(f"wrote {args.write_report}")
        elif args.gate3:
            print(md)
        return
    if args.small_world:
        rec = small_world_bench(args.config, args.ticks, args.warmup, seed=args.seed)
        _print_rec("small world", rec)
        return
    if args.scripted:
        rec = scripted_agents_bench(
            args.config, args.ticks, args.warmup, seed=args.seed, n_agents=args.agents
        )
        _print_rec(f"{args.agents} scripted agents", rec)
        return
    if args.transport:
        rec = transport_bench(args.config, args.transport_ticks, args.warmup, seed=args.seed)
        _print_rec("transport", rec)
        return
    rec = bench(args.config, args.ticks, args.warmup)
    print(f"{rec['ticks']:.0f} ticks in {rec['seconds']:.3f}s → {rec['ticks_per_s']:.0f} ticks/s")


if __name__ == "__main__":
    main()
