# Phase 7 throughput (T7.13 / gate 3)

Measured 2026-09-20. Linux 6.12.94+ · x86_64 · 4 CPUs · CPython 3.12.3.
Wall time is `time.perf_counter` (measurement only; not sim state).

## Gate 3

Spec (`07-phase7-api-sdk.md` §7.1): ≥ **1,000 ticks/s** for a small world
(R = 3, 25 instruments, no agents); ≥ **300 ticks/s** with 8 scripted agents;
REST/WS overhead measured. Floors are not loosened.

Small world is a real `World.step`: shipped `regions.yaml` (R = 3), the T6.23
25-name book (`MarketBookModule` on `Phase.MARKET`), and monthly `RealEconomy`
with SFC off. Live `step_month` remains **R = 1** (QUESTIONS T3.15); there is
no R = 3 monthly orchestrator yet. No agents on line 1.

| line | spec floor | measured | ticks (days) | warmup (days) | wall | result |
|---|---|---|---|---|---|---|
| small world (R_cfg = 3, R_live = 1, 25 names, 0 agents) | ≥ 1,000 ticks/s | **5,488** ticks/s | 2,520 | 42 | 0.459 s | pass |
| 8 scripted idle agents | ≥ 300 ticks/s | **4,786** ticks/s | 2,520 | 42 | 0.527 s | pass |

**Gate 3 passed** (engine lines). Headroom ≈ 5.5× and 16×. No cProfile pass:
the floors were met (T9.05 Rust port not started).

Eight-agent line: each tick submits one autopilot `FirmDecision` per agent
(`LocalClient.submit_decisions`, all levers omitted) then `World.step(1)` on
the same small world. That is the card’s “World + 8 idle/scripted submits”
path.

SDK wrapper (not the gate line; book + `FirmAgentModule` only, no monthly
engine): `MarketSimParallelEnv` with the same 8 idle Dict actions.

| path | measured | ticks (days) | wall |
|---|---|---|---|
| SDK ParallelEnv | **3,033** ticks/s | 2,520 | 0.831 s |

Same-host baselines (unchanged helpers):

| helper | measured | ticks (days) | warmup (days) |
|---|---|---|---|
| `bench()` T2.25 RealEconomy R = 1, SFC off | 9,120 ticks/s | 2,520 | 42 |
| `market_bench()` T6.23 25-name book | 15,173 ticks/s | 2,520 | 42 |

The small-world rate sits between those two: every daily tick runs the book;
month-end still runs `RealEconomy.step_month`.

## REST / WS overhead

Same **252** ticks (days) on one `WorldManager` world. `POST /v1/worlds` builds
an empty module list today, so this isolates transport + `WorldStatus` (which
includes `state_hash`) + `BondDesk.on_advance`. Starlette `TestClient` /
in-process ASGI; no TCP. WS path is `LocalClient.step(1)` + `StreamHub.publish`
(tick) + one `receive_json`. T7.05 does not auto-publish on `World.step`.

| path | ticks (days) | wall | ticks/s | vs LocalClient | extra µs / tick |
|---|---|---|---|---|---|
| LocalClient.step(1) | 252 | 0.000257 s | **979,938** | 1.0× | 0 |
| REST POST `/v1/worlds/{wid}/step` n=1 | 252 | 0.175 s | **1,438** | 681× | 694 |
| WS (step + publish + recv) | 252 | 0.028 s | **8,858** | 111× | 112 |

REST is ~681× the empty in-process step because the empty `World.step` is ~1 µs
and each POST builds a hashed `WorldStatus`. WS is cheaper than REST because it
does not serialise that status object. Neither number is a gate floor.

## Method

- Warmup ticks are discarded; the timed region is `time.perf_counter` around `step`.
- Book randomness uses `ctx.world.rng` / `stream("flow")`.
- CLI: `scripts/bench.py --gate3` (existing `--market` and default `bench()` unchanged).
- Repro: `python scripts/bench.py --gate3 --ticks 2520 --warmup 42 --transport-ticks 252`.
