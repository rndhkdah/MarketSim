# Phase 4 — Events

**Goal.** Real-world-style shocks that fire randomly and *sometimes* cascade (D5), built strictly as compositions of
the seven primitive shocks (design rule 4), with news, data releases, historic templates and scripted scenarios.
Synthetic data; magnitudes are distributions calibrated to history (D6).

**Two propagation mechanisms, kept separate:** (1) **endogenous** — through A, typed edges, regional trade, the central
bank's reaction, bankruptcies — always on; (2) **scripted chains** — follow-ups with probability < 1, a delay, damping
per hop, max depth, category cooldowns and a global concurrency cap.

## 4.1 Gate

1. Schema validates every shipped event; compositions reference only the seven primitives.
2. Cascades bounded: static sub-criticality check passes; in a 10× hazard "event storm" over 20 years the world stays
   finite, SFC holds, |gap| < 25 %, concurrency and depth caps are never exceeded.
3. Each historic template reproduces the historic **directions** in §4.5.
4. Same seed → same event history; event RNG is isolated (adding an agent does not change which events fire).

## 4.2 Schema (`config/events/*.yaml`)

```yaml
id: chip_shortage
category: supply_chain      # energy | supply_chain | pandemic | technology | disaster | policy | financial | trade | firm
hazard:
  base_rate_per_year: 0.06
  multipliers:              # each = clip(exp(coef · feature), 0.2, 5.0)
    - {feature: utilisation_gap, sector: SEMIS, coef: 8.0}
cooldown_days: 756
composition:                # primitive shocks only: demand | supply | cost_push | monetary | risk_appetite | fiscal | catastrophe
  - shock: supply
    magnitude: {dist: lognormal, median: 0.08, sigma: 0.4, sign: -1, min: 0.02, max: 0.30}
    persistence_q: 6
    targets: {sectors: [SEMIS]}           # mask; may be narrow — downstream moves are responses, never inputs
targets: {regions: all}                   # default mask for entries that omit their own
effects_extra: []                         # see whitelist below
followups:
  - {event_id: auto_production_cuts, probability: 0.5, delay: {dist: uniform, low_days: 40, high_days: 120}}
news: {headline: "Chip lead times stretch as fabs run flat out", publication_lag_days: 0, noise: 0.3}
historic_reference: {episode: "2020–23 chip shortage", source: "T4.13", calibrated_params: {verify: true}}
```

`effects_extra` whitelist — `{variable, shape, magnitude, duration}` with shape ∈ step | ramp | pulse | decay:
`want_shift[q]`, `labour_supply[r]`, `link_capacity[a,b]`, `link_cost[a,b]`, `world_demand`, `import_price[s]`,
`bank_equity` (one-off loss, posted on the ledger), `collateral_value`, `vat`, `sentiment` (used from Phase 6),
`capex_preference[s]` (investment demand shifter for the AI-boom type of event). Anything else is rejected.

**Hazard.** `h = base_rate/252 · Π multipliers` per day; fire with `p = 1 − exp(−h)` from stream `events`. Features:
output gap, inflation, policy rate, `utilisation_gap[s]`, bank capital ratio, leverage, price/fundamental gap (stub Q
until Phase 6), inventory cover, unemployment, days since last event of the category.
**Chains.** Effective follow-up probability = `probability × damping^depth` (damping 0.7); `max_depth` 3 (config
allows 4); per-category cooldown; `max_concurrent` 4; every firing logged with its parent → cascade tree.
**Static check:** for every event `Σ followup probabilities ≤ 0.9` and the follow-up graph is a DAG of depth ≤
`max_depth` → expected cascade size is finite.
**Timing.** Primitive shocks enter the real economy at the next month end with weight `(21 − d)/21`; news and
financial effects are immediate.

## 4.3 Visibility

Headlines at event time (optionally lagged); **magnitudes are never published** — they are revealed through data
releases and prices. `NewsItem{id, tick, category, headline, regions, sectors, severity_hint, is_rumour}`;
`severity_hint` is a noisy ordinal (misclassified with probability `noise`). Optional rumours: false headlines at
`rumour_rate` per category. Professional mode enforces publication lags; game mode may show more.

**Data release calendar** (queue items): CPI — day 10 of the following month; unemployment — day 5; sector output —
day 15; GDP — quarterly, day 21 of the month after quarter end, revised at +1 and +2 months (measurement noise sd
0.3 % of level, halving at each revision); trade and government balance — quarterly; policy decision — quarter end.

## 4.4 Historic templates — magnitudes are **indicative seeds, to be verified in T4.13**

| id | composition (primitive → target) | chain (p) | indicative history |
|---|---|---|---|
| `oil_embargo_1973` | cost_push ENERGY, log +1.1…1.4, 8q | monetary_tightening (0.6), confidence_drop (0.5) | crude ≈ ×4, Oct 1973–Mar 1974 |
| `asian_crisis_1997` | world_demand −8 %, risk_appetite +200–400bp, bank_equity loss | credit_crunch (0.5) | no FX in v1 → mapped to ROW + risk |
| `dotcom_2000` | risk_appetite +, `capex_preference` −30 % SOFTWARE/TELECOM/SEMIS, sentiment bust | — | NASDAQ ≈ −78 % peak to trough |
| `gfc_2008` | collateral_value −25…30 %, bank_equity −30…50 %, risk_appetite +400bp | fiscal_stimulus (0.8), world_demand_drop (0.7) | house prices ≈ −27 %, S&P ≈ −57 % |
| `tohoku_2011` | catastrophe, INDUSTRIAL mask, 1–3 % of K in AUTOS/SEMIS/CAPGOODS/UTILITIES; link_capacity −50 % for 60 d | auto_parts_shortage (0.7) | Japanese auto output ≈ −50 % for weeks |
| `thailand_floods_2011` | supply SEMIS −25…30 %, 2q | — | global HDD shipments ≈ −30 % in a quarter |
| `covid_2020` | labour_supply −10…15 % 1–2q; want shifts (EATING_OUT_LEISURE −50 %, MOBILITY −30 %, HEALTH +15 %, COMMUNICATION +15 %, HOUSEHOLD_GOODS +10 %); world_demand −10 % | lockdown → fiscal_stimulus (0.85) → cost_push_inflation (0.5) → monetary_tightening (0.7) | US GDP ≈ −9 % in Q2, unemployment 14.7 % |
| `chip_shortage_2020` | supply SEMIS (above) | auto_production_cuts (0.5) | lead times ≈ 26 weeks |
| `suez_2021` | pulse 6 d: link/import delay, import_price +, TRANSPORT productivity − | — | ≈ 12 % of world trade transits |
| `energy_inflation_2022` | cost_push ENERGY log +0.5…0.9, AGRIFOOD +0.2 | monetary_tightening (0.8), energy_subsidy (0.5) | US CPI peak 9.1 %, policy +525bp in 16 months |
| `ai_boom_2023` | ramp 8q: `capex_preference` +20…40 % SEMIS/SOFTWARE/UTILITIES; risk_appetite −100bp; bubble hazard ↑ | tech_correction (0.35, delay 2–5 y) | data-centre capex surge |

Generic events shipped as well: `regional_disaster`, `strike`, `plant_accident`, `recall`, `large_bankruptcy` (bodies
in Phase 5), `trade_tariff`, `vat_change`, `policy_surprise`, `credit_crunch`, `fiscal_stimulus`, `confidence_drop`,
`monetary_tightening`, `world_demand_drop`. Modes: **random** (training, free play) and **scripted** (evaluation,
tests, campaigns).

## 4.5 Direction tests per template

| template | must hold within 24 months of firing |
|---|---|
| oil_embargo_1973 | CPI level ↑, GDP ↓, ENERGY earnings ↑, AUTOS and TRANSPORT output ↓, policy rate ↑ |
| gfc_2008 | bank capital ratio ↓, lending capacity ↓, CONSTRUCT and AUTOS output ↓ more than STAPLES, unemployment ↑, policy rate ↓ |
| covid_2020 | DISCRET output ↓ > 15 %, HEALTH ↑, saving rate ↑ then ↓, CPI level ↑ after the stimulus link fires |
| chip_shortage_2020 | SEMIS relative price ↑, AUTOS output ↓ with a lag, SOFTWARE barely moves |
| tohoku_2011 | output ↓ in the masked region first, CONSTRUCT ↑ later (reconstruction), INSURANCE equity ↓ |
| energy_inflation_2022 | CPI level ↑, policy rate ↑, real wages ↓, UTILITIES margin squeezed (low pass-through) |
| ai_boom_2023 | SEMIS and UTILITIES utilisation ↑, capex ↑, SEMIS capacity arrives ≈ `build_lag_q` later (cobweb) |
| dotcom_2000 / asian_crisis_1997 / suez_2021 / thailand_floods_2011 | sign of GDP, of the targeted sectors' output and of CPI as implied by the composition |

---

## 4.6 Task cards

### T4.01 — Event schema and loader
**Depends:** T2.18 · **Size:** M · **Files:** `src/marketsim/events/schema.py`, `config/events/_example.yaml`, `tests/unit/events/test_schema.py`
**Read first:** §4.2. **Build:** pydantic models, distribution specs (fixed, normal, lognormal, uniform, triangular —
truncation bounds mandatory), mask validation, follow-up references, whitelist for `effects_extra`. **Tests:** gate 1; bad
primitive name, unknown variable, missing bounds, dangling follow-up each rejected.

### T4.02 — Shock-composition engine
**Depends:** T4.01 · **Size:** M · **Files:** `src/marketsim/events/compose.py`, `tests/unit/events/test_compose.py`
**Build:** sample magnitudes from stream `events`, resolve masks, call `ShockBus.inject` with mid-month weighting.
**Tests:** a narrow SEMIS shock touches no other cell's shock state; sampled magnitudes respect bounds; deterministic.

### T4.03 — `effects_extra` executor
**Depends:** T4.02, T3.13 · **Size:** M · **Files:** `src/marketsim/events/effects.py`, `tests/unit/events/test_effects.py`
**Build:** shape functions and setters for every whitelisted variable; `bank_equity` and `vat` post on the ledger.
**Tests:** each shape's time profile; effects revert; SFC holds for ledger-touching effects.

### T4.04 — Hazard model
**Depends:** T4.01 · **Size:** S · **Files:** `src/marketsim/events/hazard.py`, `tests/unit/events/test_hazard.py`
**Tests:** empirical firing rate over 2,000 simulated years matches `base_rate` (±10 %); multipliers clipped; feature
registry rejects unknown names; cooldown respected.

### T4.05 — Scheduler and chains
**Depends:** T4.02, T4.04, T0.10 · **Size:** M · **Files:** `src/marketsim/events/scheduler.py`, `src/marketsim/events/chains.py`, `tests/unit/events/test_chains.py`
**Read first:** §4.2 chains. **Build:** `Events` World module (phase EVENTS); follow-up sampling with damping, depth cap,
cooldowns, concurrency cap; cascade tree log; static sub-criticality check at load. **Tests:** depth never > max;
concurrency never > cap; a graph with Σp > 0.9 or a cycle is rejected at load; gate 4.

### T4.06 — News feed
**Depends:** T4.05 · **Size:** S · **Files:** `src/marketsim/events/news.py`, `tests/unit/events/test_news.py`
**Read first:** §4.3. **Tests:** magnitudes never appear in any `NewsItem`; lag honoured; severity misclassification rate
≈ `noise`; rumours flagged only in debug mode.

### T4.07 — Data release calendar
**Depends:** T2.17, T0.10 · **Size:** M · **Files:** `src/marketsim/events/releases.py`, `tests/unit/events/test_releases.py`
**Build:** scheduled releases with lags and GDP revisions; `observe()` exposes only published vintages.
**Tests:** an agent cannot see month-*m* CPI before day 10 of *m+1*; revisions converge to truth; vintages stored.

### T4.08 — Historic templates and generic events
**Depends:** T4.03 · **Size:** M · **Files:** `config/events/*.yaml`
**Read first:** §4.4. **Build:** the eleven templates + generic events, every magnitude carrying `verify: true`.
**Tests:** all files validate; static chain check passes.

### T4.09 — Scenario runner
**Depends:** T4.05 · **Size:** S · **Files:** `src/marketsim/scenarios/{loader,runner}.py`, `config/scenarios/*.yaml`, `tests/unit/scenarios/test_runner.py`
**Build:** scripted mode — ordered events at fixed ticks with fixed or sampled magnitudes, hazards off or on.
**Tests:** scripted run reproducible; mixing scripted and random events keeps RNG isolation.

### T4.10 — Cascade and storm tests
**Depends:** T4.08 · **Size:** S · **Files:** `tests/validation/test_event_storm.py`
**Tests:** gate 2 (`slow`); mean cascade size matches the branching-process expectation (±20 %).

### T4.11 — Template direction tests
**Depends:** T4.08, T4.09 · **Size:** M · **Files:** `tests/validation/test_event_templates.py`
**Tests:** §4.5, each template fired alone at median magnitude with follow-ups forced on, then forced off.

### T4.12 — Firm-level event hooks
**Depends:** T4.03 · **Size:** S · **Files:** `src/marketsim/events/firm_hooks.py`
**Build:** interfaces for plant accident, strike, recall, large bankruptcy with `targets.firms`; no-op until Phase 5.
**Tests:** events validate and schedule; hooks called with the right payload.

### T4.13 — Magnitude calibration — **HUMAN GATE**
**Depends:** T4.11 · **Size:** M · **Files:** `config/events/*.yaml`, `claude/plan/reports/event-calibration.md`
**Build:** for every `verify: true` figure, find a primary source (FRED, BLS, EIA, BEA, central-bank publications),
replace the seed with a distribution, fill `historic_reference.source`. **Tests:** no `verify: true` left; §4.5 still green.
**Out of scope:** tuning the economy to match magnitudes (that is T8.04).
