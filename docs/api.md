# marketsim API v1

One surface for the game client, a trading/operating agent, and RL training (D1).
Ticks are **simulated days** (21 days = 1 month). Cash fields are **cr**. Rates
are **annual decimals** unless the name ends in `_m` (per month) or `_d` (per day).

Schema version: `v1.1` (`marketsim.api.schemas.API_SCHEMA_VERSION`). Changing a
wire model requires a version bump and a new golden under `tests/golden/data/`.
The frozen `api_schema_v1.json` is not edited in place.

## Transports

| Binding | Module | Use |
|---|---|---|
| In-process | `marketsim.sdk.local.LocalClient` | Training throughput; same method names as the HTTP SDK |
| REST | `POST/GET /v1/worlds/…` (`marketsim.api.rest.create_app`) | Control, queries, orders, policy |
| WebSocket | `GET /v1/worlds/{wid}/stream?since=` | Ticks, fills, news, reports, releases |
| HTTP SDK | `marketsim.sdk.client.HttpClient` / `AsyncHttpClient` | Typed wrappers; WS reconnect with `since` |
| Gymnasium | `marketsim.sdk.gym_env.MarketSimEnv` | Single-agent factored `Dict` action |
| PettingZoo | `marketsim.sdk.pz_env.MarketSimParallelEnv` | Multi-agent; bankrupt agents are removed |
| Policy | `marketsim.sdk.policy_env.PolicyEnv` | `policymaker` quadratic loss (§7.3) |

`LOCAL_METHODS` (`observe`, `submit`, `submit_orders`, `submit_decisions`, `step`,
`reset`, `state_hash`, `status`) is the contract every client exposes.

## Auth

`POST /v1/worlds` returns a world id. The bootstrap admin bearer is
`SHA-256("{wid}:admin:{seed}:admin")` (`world_admin_token`).
`POST …/agents` returns a per-agent token. Send `Authorization: Bearer <token>`.

Roles: `admin`, `agent`, `observer`, `policymaker`. A policymaker binds one
authority (`GOVT` or `CENBANK`). In **professional** mode a policymaker cannot
open a trading account (non-zero starting capital or `POST …/orders` → 403).

Limits per agent per tick: 32 orders, 8 firm decisions, 64 open orders.

## REST (`/v1/worlds`)

| Group | Endpoints |
|---|---|
| World | `POST /v1/worlds` · `GET …/{wid}` · `POST …/reset` · `POST …/step` · `DELETE …/{wid}` |
| Agents | `POST …/agents` · `DELETE …/agents/{aid}` |
| Observe | `GET …/observe?since=` — published fields only (no ξ, arb capital, unpublished vintages) |
| Orders | `POST …/orders` (Idempotency-Key) · `DELETE …/orders/{oid}` · `GET …/orders` · `GET …/fills` |
| Firms | `POST …/firms` · `POST …/firms/{fid}/decisions` · `GET …/state` (operator) · `GET …/financials` · `GET …/reports` |
| Goods / labour | `GET …/goods` · `GET …/labour/{region}` · `GET …/regions` |
| Bonds | `GET …/bonds` · `GET …/bonds/auctions` · `POST …/bonds/auctions/{aid}/bids` |
| Policy | `GET/POST …/government/…` · `GET/POST …/cenbank/…` (role `policymaker`) |

Errors are `{code, message, details}` (`ErrorModel`). 401 missing/invalid token;
403 role; 404 unknown id; 422 validation.

## WebSocket

`GET /v1/worlds/{wid}/stream?since=<seq>` with a Bearer token. Event `seq` is a
per-world unitless counter starting at 1. Resume sends `seq > since` with no
gaps or duplicates. Fills and addressed news are filtered to the caller. Live
delivery coalesces ticks (keep the latest; never drop fills).

## Lockstep and replay

Lockstep applies actions in `(agent_id, sequence)` order (`LockstepBarrier`).
`ReplayLog` stores seed, config hash, agent inputs and scripted events.
`replay()` reproduces every per-tick `state_hash`. Export `.npz` keys:
`tick`, `state_hash`, `seed`, `config_hash`.

## Policy (D13)

With no policymaker registered, `GOVT` and `CENBANK` stay on **autopilot**.
Registering one switches that authority to **agent**; deregistering restores the
home control. `POST` decisions echo **clipped** levers (tax rates ∈ [0, 0.6],
tariff ≤ 0.5, policy rate ∈ [0, 0.20]).

## Examples

```bash
python examples/quickstart_trader.py --config config
python examples/quickstart_operator.py --config config
```

Both scripts talk to an in-process FastAPI app (`create_app` + `HttpClient`).
They are the CI smoke for this page (`tests/integration/test_quickstart.py`).
