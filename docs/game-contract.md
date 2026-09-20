# Game layer contract (T8.07)

The game UI is an API client. It talks to the same v1 surface as a trader, an
operator, or an RL harness (`docs/api.md`). The engine prices assets and steps
the world; strategies, pause/speed, slots, score banners, and flavour text live
here.

Schema version: `v1.1`. Ticks are **simulated days** (21 = 1 month). Cash and
scores are **cr**. Rates are annual decimals. NPC scale and event severity are
dimensionless.

## Transport

| Need | Use |
|---|---|
| Create / step / observe / orders / firms | REST `POST/GET /v1/worlds/…` (`create_app`) |
| Typed client | `marketsim.sdk.client.HttpClient` |
| Live ticks, fills, news | WebSocket `GET …/stream?since=` |
| Engine snapshot helpers | `write_snapshot` / `load_snapshot` (`marketsim.api.replay`) |
| Campaign overlay | `marketsim.scenarios.campaign` |

Auth is the v1 bearer (`world_admin_token` then the agent token). Observe
publishes only the T7.01 whitelist — no ξ, arb capital, or unpublished vintages.

`CampaignSpec.world_spec()` sends `mode=game` and the campaign seed. It does
**not** retune `sectors.yaml` / `edges.yaml`. Difficulty knobs stay on the
campaign object.

## Pacing (pause / speed)

`WorldSpec.run_mode` is `lockstep` or `realtime`.

| Mode | Who advances the clock | Speed |
|---|---|---|
| `lockstep` | The UI calls `POST …/step` when every player has acted (empty action allowed) | Wall-clock is ignored |
| `realtime` | The UI paces `POST …/step` at `ticks_per_second` (ticks/s) | Late inputs land on the **next** tick (T7.06) |

`PacingControls` is the UI object:

- `pause()` — do not call `step`. `due_ticks(elapsed_s)` returns 0.
- `resume()` — clear the flag. Does not itself step.
- `set_speed(ticks_per_second)` — ticks/s; must be > 0.
- `interval_s()` — wall-clock period in seconds. The engine never calls
  `time.time()`; the UI supplies elapsed seconds.

There is no extra REST pause verb. `WorldStatus.status` may read `running`;
pause is a client-side gate in front of `POST …/step`.

## Difficulty

```
difficulty = founding_capital × npc_scale × event_severity
```

| Knob | Units | Applied through |
|---|---|---|
| `founding_capital` | cr | `AgentRegistration.starting_capital` and `FoundFirmRequest.capital` |
| `npc_scale` | dimensionless | Game-layer factor (NPC mass vs the player) |
| `event_severity` | dimensionless | Game-layer factor (T7.11 calm→severe is the narrative scale) |

The product is a **cr-equivalent index**, not a new yaml key and not a shock
injected into a sector. A host may later map `npc_scale` onto the existing
`firms.npc_initial_share` override, or pick a T7.11 severity tier, without
changing Layer-1 elasticities.

## Scoring

Each observe, compute a `Scorecard` from **published** fields:

| Field | Units | Source |
|---|---|---|
| `net_worth` | cr | `Observation.portfolio.net_worth` (cash + MV of holdings + own-firm equity). If the book is empty, the UI may show the founding capital it just posted. |
| `valuation` | cr | Own-firm published books (`equity` / `equity_value` / `book_equity`). 0 if unpublished. |
| `market_share` | dimensionless | Published goods/report share. 0 if unpublished. |

Do not invent a sector beta or a hand-set score weight. Master-plan §12 reward
defaults (operator Δ equity + dividends; trader risk-adjusted P&L) still report
these three numbers.

## Save / load slots

Two files per named slot under `CampaignSpec.slot_dir`:

| File | Writer | Contents |
|---|---|---|
| `{name}.campaign.json` | `write_campaign_slot` | Overlay: spec, world id, seed, tick (days), `state_hash`, score, narrative, takeovers |
| `{name}.world.json` | `save_engine_slot` → `write_snapshot` | Engine `World.to_state()` (T7.10) |

Load the overlay with `read_campaign_slot`. Restore the engine with
`load_engine_slot` / `load_snapshot`, or with `POST /v1/worlds/load` when that
route is wired (helpers already exist; `rest.py` is not extended by this card).

A live session can save and keep stepping — the overlay is metadata. Continuing
from a cold start needs the engine snapshot (same seed + same inputs → same
hash). Slot names are a single path segment.

## Narrative text

Templates consume only `NewsItem` fields (`id`, `tick`, `category`, `headline`,
`regions`, `sectors`, `severity_hint`, `is_rumour`). Magnitudes never appear
(T4.06). `narrate()` formats:

```
{Rumour — }{headline} [{category} · day {tick} · hint {severity_hint}{ · regions}{ · sectors}]
```

`tick` is days; `severity_hint` is the noisy ordinal 0–3. Rumours stay labelled
only when the item says `is_rumour` (professional debug). The UI must not
decode a size from the hint.

## Takeover notifications

Control transfer is T6.16: ownership > 0.50, effective at the next month-end,
`NewsItem.category == "corporate"`, `id == "control:<firm_id>:<tick>"`,
headline `Operating control of {firm_id} passed to {operator}` (no % and no
digits for a stake). `takeover_notification()` turns that into:

```
Takeover: {headline} ({firm_id}; day {tick})
```

Holdings ≥ 5 % are disclosed in professional mode and hidden in game mode.
Tender offers are out of scope.

## Campaign scenarios

`campaign_scenarios()` ships three short demos (a handful of days, not 50k):

| Id | Role |
|---|---|
| `demo` | Default scripted player (3 days, 100 cr, scales 1.0) |
| `operator` | Firm-first variant (4 days, 50 cr) |
| `storm_watch` | Same API path; difficulty product uses 80 cr × 1.2 × 1.5 |

`run_scripted_player(config_dir)` builds `create_app`, opens `HttpClient`, and
finishes `demo`: register, found, one decision + one limit buy, pause/resume
check, mid-run slot, `duration_ticks` steps, final score.

## UI must not

- Call the engine except through the public API (or T7.10 snapshot helpers).
- Inject a shock into a sector; demand still originates at HOUSEHOLD / GOVT /
  ROW / investment.
- Cap a price level, clip a cash/debt balance, or add noise to dress a score.
- Run a trading strategy inside the engine (background flow is statistical).
- Expose hidden state from `observe()` outside debug mode.
