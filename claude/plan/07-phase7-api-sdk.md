# Phase 7 — API and SDK

**Goal.** One API for both uses (D1): REST for control and queries, WebSocket for streams, an in-process binding with
the same schemas for training throughput; lockstep mode for training and evaluation, real-time mode for the game;
Gymnasium / PettingZoo wrappers with factored action spaces; deterministic replay. The game UI is just another client.

## 7.1 Gate

1. **Multi-agent training run end-to-end:** PettingZoo parallel env, 4 random + 2 scripted agents, 50,000 ticks, all
   invariants on, no violation.
2. **Lockstep determinism:** same seed + same actions → same state hash regardless of submission order or timing;
   replay from the log reproduces every per-tick hash.
3. **Throughput (in-process):** ≥ 1,000 ticks/s for a small world (R = 3, 25 instruments, no agents); ≥ 300 ticks/s with
   8 scripted agents; REST/WS overhead measured and documented.
4. Schemas versioned (`v1`), OpenAPI published, SDK covers every endpoint, examples run in CI.
5. Information rules hold through the API: no hidden state, professional-mode lags and blackouts enforced.
6. A `policymaker` client can run fiscal and monetary policy and the debt office end-to-end through the API; with none
   registered the authorities stay on autopilot. Agents can bid in bond auctions and trade every bond bucket.

## 7.2 Surface

REST, all under `/v1/worlds/{wid}` unless noted:

| group | endpoints |
|---|---|
| world | `POST /v1/worlds` (seed, scenario, overrides, mode, run mode) · `POST …/reset` · `POST …/step` (lockstep driver / admin) · `GET …` status · `DELETE …` |
| agents | `POST …/agents` → token, account, limits |
| observe | `GET …/observe?since=<seq>` — market data, published macro releases, news, portfolio, own firms, others' lagged reports |
| orders | `POST …/orders` (both venues; idempotency key) · `DELETE …/orders/{oid}` · `GET …/orders` · `GET …/fills` |
| firms | `POST …/firms` found · `POST …/firms/{fid}/decisions` (batched, any lever subset) · `GET …/firms/{fid}/state` (operator only) · `GET …/firms/{fid}/financials` · `POST …/firms/{fid}/plants` · `POST …/firms/{fid}/financing` · `GET …/firms/{fid}/reports` |
| equity | `POST …/firms/{fid}/listing` · `…/issue` · `…/buyback` · `…/dividends` · `GET …/firms/{fid}/captable` |
| goods / labour | `GET …/goods` (reference prices, demand indicators, competitor prices) · `GET …/labour/{region}` · `GET …/regions` |
| bonds | `GET …/bonds` (curve, bucket prices, outstanding, published holdings) · `GET …/bonds/auctions` (calendar, sizes, results) · `POST …/bonds/auctions/{aid}/bids` |
| policy (role `policymaker`) | `GET …/government/state` (budget, debt, issuance plan) · `POST …/government/decisions` · `GET …/cenbank/state` · `POST …/cenbank/decisions` |
| replay | `GET …/replay` · `POST …/export` · `POST …/save` · `POST /v1/worlds/load` |

WebSocket `…/stream`: ticks, fills, news, reports, data releases; per-agent filtering; sequence numbers; resume by
`since`. gRPC is optional later (T9.06). `marketsim.sdk.LocalClient(world)` exposes the same methods and schema objects
without serialisation. Per-agent account, token, limits (orders per tick, decisions per tick, open orders).

**Lockstep:** the tick advances when every registered agent has acted (possibly with an empty action) or timed out;
actions are applied in `(agent id, client sequence)` order. **Real-time:** wall-clock pacing at `ticks_per_second`; late
inputs land on the next tick.

## 7.3 RL contract

Observation: own books live, others lagged, market and macro data as published, encoded news, portfolio.
Reward: operators — Δ(equity value) + dividends (not short-run profit); traders — risk-adjusted P&L; both report net
worth (cash + market value of holdings + own-firm equity). Episode = horizon or bankruptcy. Curriculum by levers
(price → production → hiring → capex → financing) and by event severity. Domain randomisation per episode; fixed
evaluation scenarios; collusion detection (T5.19); parallel worlds.
Policy agents (`policymaker` role): observation = published macro data, budget and debt, curve; actions = any subset of the
`02-…` §2.13 levers; reward = −[(π − π\*)² + λ_y·gap² + λ_b·(b − b\*)²] or a custom scorer.
Action space (factored `Dict`): `orders` — K slots × (instrument, side, type, size as a fraction of buying power or
position, limit offset in spreads); `firm` — one normalised `Box` per lever with a mask for levers left on autopilot.

---

## 7.4 Task cards

### T7.01 — Versioned API schemas
**Depends:** T6.12, T5.06 · **Size:** M · **Files:** `src/marketsim/api/schemas.py`, `tests/unit/api/test_schemas.py`, `tests/golden/data/api_schema_v1.json`
**Build:** pydantic v2 models: `WorldSpec`, `WorldStatus`, `AgentRegistration`, `Observation`, `OrderRequest`, `OrderAck`,
`Fill`, `FirmDecision`, `FoundFirmRequest`, `ListingRequest`, `CorporateAction`, `GoodsView`, `LabourView`, `NewsItem`,
`Report`, `ErrorModel{code, message, details}`. **Tests:** round trips; golden JSON-schema snapshot (a change requires a
version bump); `Observation` field whitelist contains no hidden state.

### T7.02 — Sessions, worlds, accounts
**Depends:** T7.01 · **Size:** M · **Files:** `src/marketsim/api/sessions.py`, `tests/unit/api/test_sessions.py`
**Build:** `WorldManager` (many worlds), agent registration → token + `AGENT:<id>` entity + starting capital as a tagged
`capital_transfer`, roles (admin / agent / observer), per-agent limits. **Tests:** world isolation; auth; limits.

### T7.03 — In-process client
**Depends:** T7.02 · **Size:** S · **Files:** `src/marketsim/sdk/local.py`, `tests/unit/sdk/test_local.py`
**Tests:** method parity with the REST client (introspection test); results identical to direct `World` calls.

### T7.04 — REST endpoints
**Depends:** T7.02 · **Size:** L · **Files:** `src/marketsim/api/rest.py`, `tests/integration/api/test_rest.py`
**Read first:** §7.2. **Build:** FastAPI routers per group, OpenAPI, error model, pagination, idempotency keys.
**Tests:** per endpoint — happy path, auth failure, validation failure; operator-only endpoints return 403 to others;
professional-mode lags visible through the API.

### T7.05 — WebSocket streams
**Depends:** T7.04 · **Size:** M · **Files:** `src/marketsim/api/ws.py`, `tests/integration/api/test_ws.py`
**Tests:** resume yields no gaps or duplicates; an agent sees only its own fills; slow-consumer policy (coalesce ticks).

### T7.06 — Lockstep barrier and real-time pacing
**Depends:** T7.02 · **Size:** M · **Files:** `src/marketsim/api/lockstep.py`, `tests/integration/api/test_lockstep.py`
**Tests:** gate 2; timeout → empty action; permuted submission order → identical hash; late real-time input lands next tick.

### T7.07 — Python SDK (HTTP / WS)
**Depends:** T7.04, T7.05 · **Size:** M · **Files:** `src/marketsim/sdk/client.py`, `tests/integration/sdk/test_client.py`
**Build:** sync and async clients, typed returns, reconnect with `since`. **Tests:** against a live test server.

### T7.08 — Gymnasium single-agent environment
**Depends:** T7.03 · **Size:** M · **Files:** `src/marketsim/sdk/gym_env.py`, `tests/unit/sdk/test_gym_env.py`
**Read first:** §7.3. **Build:** factored action space, fixed-size observation encoder, reward options, termination,
autopilot masks. **Tests:** `gymnasium.utils.env_checker`; seeding deterministic; masked levers ignored.

### T7.09 — PettingZoo parallel environment
**Depends:** T7.08 · **Size:** M · **Files:** `src/marketsim/sdk/pz_env.py`, `tests/unit/sdk/test_pz_env.py`
**Tests:** `parallel_api_test`; an agent can go bankrupt mid-episode and is removed cleanly.

### T7.10 — Replay, export, save / load
**Depends:** T7.06 · **Size:** M · **Files:** `src/marketsim/scenarios/replay.py`, `src/marketsim/api/replay.py`, `tests/integration/test_replay.py`
**Build:** event-sourced log (seed, config hash + overrides, randomisation draws, every agent input, scripted events);
deterministic replay; export series to `.npz` (parquet optional); snapshots. **Tests:** replay reproduces all hashes;
save → load → continue equals uninterrupted; export schema stable.

### T7.11 — Scenario packs, curricula, evaluation sets
**Depends:** T4.09, T7.08 · **Size:** M · **Files:** `config/scenarios/*.yaml`, `src/marketsim/sdk/curriculum.py`, `tests/unit/sdk/test_curriculum.py`
**Build:** curricula by levers and event severity; fixed evaluation scenarios; metrics (net worth, Sharpe, max drawdown,
market share, bankruptcies). **Tests:** every scenario loads and runs one simulated year.

### T7.12 — Vectorised parallel worlds
**Depends:** T7.08 · **Size:** M · **Files:** `src/marketsim/sdk/vector_env.py`, `tests/integration/test_vector_env.py`
**Build:** multiprocessing (spawn) vector env; worker seeds from `SeedSequence.spawn`. **Tests:** per-worker determinism;
N workers ≥ 0.75·N× single throughput.

### T7.13 — Throughput and profiling
**Depends:** T7.12 · **Size:** S · **Files:** `scripts/bench.py`, `claude/plan/reports/phase7-throughput.md`
**Tests:** gate 3. If missed: profile, list the top 5 hot spots, propose fixes — do not start a Rust port.

### T7.14 — End-to-end multi-agent training smoke run
**Depends:** T7.09, T7.11 · **Size:** M · **Files:** `examples/train_smoke.py`, `tests/integration/test_training_smoke.py`
**Tests:** gate 1 (`slow`); summary report with invariant counters all zero.

### T7.15 — API documentation and quickstart
**Depends:** T7.07 · **Size:** S · **Files:** `docs/api.md`, `examples/quickstart_trader.py`, `examples/quickstart_operator.py`
**Tests:** examples run in CI against an in-process server.

### T7.16 — Policy-maker role and endpoints (D13)
**Depends:** T7.04, T2.31 · **Size:** M · **Files:** `src/marketsim/api/rest.py`, `src/marketsim/api/sessions.py`, `src/marketsim/sdk/policy_env.py`, `tests/integration/api/test_policy_api.py`
**Build:** `policymaker` role bound to one authority (`GOVT` or `CENBANK`), the four endpoints, decision validation echoing
clipped levers, `PolicyEnv` (Gymnasium) with the §7.3 reward. **Tests:** gate 6; a non-policymaker gets 403; a policymaker
cannot open a trading account in professional mode; registering one switches that authority from autopilot to `agent` and
deregistering switches it back.

### T7.17 — Bond-market endpoints
**Depends:** T7.04, T6.26 · **Size:** S · **Files:** `src/marketsim/api/rest.py`, `src/marketsim/api/schemas.py`, `tests/integration/api/test_bonds_api.py`
**Tests:** an agent sees the calendar, bids, wins or loses at the stop-out yield, and receives coupons and redemptions; schema
snapshot bumped to include bond objects.
