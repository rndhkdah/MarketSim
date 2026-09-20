# marketsim — Master Implementation Plan

Version 1.2 · 2026-09-19 (v1.1 bond market, policy authorities, one borrowing rate; v1.2 adds D15: the monetary policy framework) · Owner: Harry · Audience: coding agents (Cursor) and the human reviewing them.

This plan consolidates every project doc — `claude/engine-design.md`, `claude/simulation-engine-design-brief.md`
(v0.3), `claude/vic3-demand-model.md`, `claude/market-simulation-landscape.md`, `claude/marketsim/` — into a build
sequence of small task cards. Where a doc and this plan disagree, this plan wins and the difference is listed in §14.

| file | contents |
|---|---|
| `AGENTS.md` (repo root) | always-on rules for coding agents |
| `00-MASTER-PLAN.md` | this file: scope, architecture, conventions, phase map, gates, open decisions |
| `01-phase0-foundation.md` | packaging, loaders, Layer-1 bug fixes, core utilities, World skeleton |
| `02-phase2-dynamic-layer.md` | ledger (SFC) + monthly real economy + policy + credit edges — **verified equations** |
| `03-phase3-regions-demand.md` | regions, trade, wealth tiers × wants, NPC entry/exit |
| `04-phase4-events.md` | event schema, hazards, cascades, news, data releases, historic templates |
| `05-phase5-firms.md` | agent-operated firms, heterogeneous-seller markets, financing, bankruptcy, abuse controls |
| `06-phase6-pricing-markets.md` | valuation, mispricing, impact kernel, market maker, CLOB, cap tables, control, margin |
| `07-phase7-api-sdk.md` | REST/WS API, lockstep/real-time, SDK, Gymnasium/PettingZoo, replay |
| `08-phase8-9-realism-game-scale.md` | calibration, realism, game layer, scale-out (coarse; refine after gates) |
| `PROGRESS.md` | checklist of every task ID (the work queue) — 166 cards |
| `README.md`, `DECISIONS.md`, `QUESTIONS.md` | how to start in Cursor · ADRs · questions raised by agents |
| `reference/` | the numerical prototype used to verify Phase 2 — **reference only, not production code** |

---

## 1. What we are building — decisions D1–D11 (2026-09-19)

| # | decision |
|---|---|
| D1 | Dual use: professional agent testing/training environment **and** game-engine backbone, same API. Testing environment sets the fidelity budget. |
| D2 | One country, several **regions** (3–5); rest of world is an exogenous ROW node (export demand, import prices). Multi-country + FX later. |
| D3 | Textbook macro: institutional nodes HOUSEHOLD, GOVT, CENBANK, ROW, LABOUR + 18 sectors on an IO backbone (Layer 1, done). |
| D4 | **Hybrid time**: fixed tick (1 simulated day) + discrete event queue. Real economy monthly, policy quarterly, valuation/market daily. |
| D5 | **Events** are real-world-style shocks that fire randomly and *sometimes* cascade; each is a composition of primitive shocks + target mask + probabilistic follow-ups. |
| D6 | Market data is **synthetic**, calibrated to historic episodes (magnitudes as distributions). |
| D7 | **Moderate execution realism**; agents' own orders must move prices. Tier-2 impact metrics are CI tests. |
| D8 | The engine **prices assets**; strategies live outside and reach it through the API. |
| D9 | Agents can **own and operate firms** (price, produce, hire, invest, finance). |
| D10 | **Any agent may buy shares of any listed firm**, including other agents' firms, funds permitting. Working assumption: control follows >50 % of voting shares. |
| D11 | Storage backend deferred; all state serialisable. |
| D12 | **Bond market.** Government debt in three maturity buckets (bill ≈ 3m, note ≈ 3y, bond ≈ 10y) sold in uniform-price **auctions** and traded on a secondary market; the central bank operates in it (QE/QT); corporate bonds are issued through one **pooled** vehicle. Agents can bid and trade. |
| D13 | **The government and the central bank conduct macroeconomic policy as actors.** Each is a *policy authority* with levers (purchases and their sector/regional mix, tax rates, VAT, excise/subsidies, tariffs, transfers, rescues, fiscal-rule parameters, debt management; policy rate or rule parameters, forward guidance, macroprudential settings, QE/QT). Every lever has an autopilot (the Phase-2 rules); control is `autopilot`, `scripted` or `agent` (API role `policymaker`). |
| D15 | **The central bank sets the rate like a real one.** A committee on an 8-meeting calendar, a **dual mandate** (inflation *and* the unemployment gap), core-vs-headline blending, a neutral rate that tracks trend growth, gradualism, 25bp announcements with a deadband, and decisions taken on **published, lagged, revised** data. Shipped but off by default: makeup strategies (leaky and clipped only), risk-management asymmetry, a financial-conditions term, committee dispersion. At the ELB: guidance → QE → LOLR. See `02-…` §2.14. |
| D14 | **All companies borrow at the same interest rate, everywhere**: policy rate (or the matching government yield) + **one** global corporate spread — no firm, sector, region or rating premium. Leverage is limited by quantity, not price. `credit.pricing: risk_based` remains as an option. |

Assumptions carried as defaults (see §12): commercial use in scope → permissive licences only; tick = 1 day;
generic (BEA/FRED-flavoured) calibration.

## 2. Status: what exists, what is verified, what is design-only

| item | status |
|---|---|
| Layer 1 — 18-sector A matrix, typed-edge config, derived betas, rotation/regime tests | **done**, re-run and passing in a clean sandbox (numpy 2.4, scipy 1.17) |
| BEA replacement of the seed matrix | script exists, **never run** (needs network); also see bug B1 below |
| Phase-2 equations (production modes, rationing, prices, capex, households, fiscal, Taylor) | **prototyped and verified numerically** — `reference/dynamic_core_prototype.py`; results in §13 |
| Ledger/SFC, banks, credit + collateral edges, typed substitution edges, catastrophe & risk-appetite shocks | **design only** — specified here, must pass the Phase-2 stability suite when switched on |
| Phases 3–9 | **design only** |
| One borrowing rate for all firms (D14) | **verified in the prototype** (§13) |
| Monetary framework (D15) | **reaction function verified in the prototype** (§13); committee, communication and ELB toolkit design-only |
| Policy authorities (D13), bond market (D12) | **design only** — autopilot / par-pricing modes must reproduce the verified behaviour bitwise |

Bugs/gaps found in the existing repo (fixed in Phase 0):

- **B1** `validate_io.py` and `derive_betas.py` import `build_A()` and never read `io_table.json` / `io_table_bea.json` →
  after the BEA fetch the real table is neither validated nor used. Fix: one `load_io(path)` loader used everywhere.
- **B2** `derive_betas.py` computes at import time; `FD_BASE` hardcodes 0.62/0.19/0.13/0.06 instead of `fd_weights`;
  unused `earn` in the BANKS block; `vic3_compare.py` prints the trade-off lines twice.
- **B3** `bank_capital_gate` (midpoint 0.085, steepness 90) evaluates to ≈0.63 at baseline if the capital ratio is
  1/`asset_leverage` = 0.0909. Fix: define the ratio on risk-weighted assets (baseline 0.125 → raw gate 0.973) and
  **normalise the gate to 1 at baseline**.
- **B4** `capex.coefficients` have no unit contract; `q_tobin: 0.30` is ≈10× too strong as written. Fix: the unit
  contract in `02-…` §2.6 and `q_scale: 0.1` on a smoothed, clipped log-Q gap.
- **B5** `inv_lag_q` conflates inventory cover with order-book length. Fix: `production_mode` per sector.

## 3. Architecture

```
Clients: game UI | RL harness | scripted bots | human traders / operators
              │  REST + WebSocket (in-process binding for training)
        ┌─────▼──────┐
        │ API        │ auth, accounts, sessions, lockstep / real-time                      Phase 7
        ├────────────┤
        │ L4 Market  │ engine market maker (sector equity, bonds, commodities) + CLOB       Phase 6
        │            │ (agent-firm shares); background flow; impact kernel; margin
        ├────────────┤
        │ L3 Pricing │ cash flows up from the real layer, discount rates down from the      Phase 6
        │            │ financial layer → fundamental value → price = V·exp(ξ)               (stub in Phase 2)
        ├────────────┤
        │ Firms      │ agent-operated firms: levers, autopilot, books, bankruptcy           Phase 5
        ├────────────┤
        │ L1/L2 Real │ NPC mass per region × sector: IO + typed edges, labour, policy       Phase 2–3
        │            │ (Taylor, fiscal), ROW, demand layer (tiers × wants)
        ├────────────┤
        │ L0 Ledger  │ double-entry positions + journal; balance-sheet and transaction-     Phase 2
        │            │ flow matrices; SFC assertion for every entity incl. agents and firms
        ├────────────┤
        │ Events     │ hazards → primitive-shock compositions, follow-up chains, news        Phase 4
        ├────────────┤
        │ Clock      │ tick loop + event queue, deterministic ordering                       Phase 0
        └────────────┘
```

Feedback edges run 4 → 1/2: cost of capital → capex, wealth → consumption, credit spreads → refinancing → hiring.
They are what make the system cycle rather than merely respond; they are switched on one at a time, each behind the
stability suite (T6.19).

Scope note (landscape doc): the survey recommends a separate matching-engine *process* on a real protocol. With
daily ticks and D7's "moderate realism" that is out of scope for v1 — the venue sits behind a `Venue` protocol
(`market/venue.py`) so it can be moved out of process later.

## 4. Target repository layout

```
AGENTS.md  .cursor/rules/marketsim.mdc  pyproject.toml  Makefile
config/        sectors.yaml edges.yaml io_table.json [io_table_bea.json] betas.md
               dynamics.yaml ledger.yaml regions.yaml wants.yaml buy_packages.yaml
               firms.yaml markets.yaml world.yaml events/*.yaml scenarios/*.yaml
src/marketsim/
  core/        calendar, clock (tick + event queue), rng, hashing, state, config, erlang, errors
  layer1/      io (load_io), build_io, betas, checks
  ledger/      entities, instruments, journal, matrices (BSM/TFM), sfc
  real/        steady_state, production, orders, prices, labour, capex, residential, households,
               government, cenbank, row, banks, edges (typed), shocks, aggregates, economy (orchestrator)
  regions/     geometry, trade (MRIO shares), labour pools, migration
  demand/      tiers, wants, packages, calibrate, system (DemandSystem interface)
  events/      schema, hazard, compose, chains, scheduler, news, releases, templates
  firms/       firm, accounts, levers, autopilot, goods_market, labour_market, procurement, financing,
               rating, plants, bankruptcy, reports, controls
  equity/      captable, listing, corporate_actions, control
  pricing/     expectations, curve, discount, fundamentals, mispricing, bonds, commodities, provider
  market/      instruments, venue, mm, clob, background, impact, settlement, margin, shorting, surveillance
  api/         schemas, sessions, rest, ws, lockstep
  sdk/         client, gym_env, pz_env
  scenarios/   loader, runner
  explain/     traces
  world.py     World facade: create / reset / submit / step / observe / state_hash / save / load
scripts/       build_io.py derive_betas.py fetch_bea_io.py vic3_compare.py run_world.py sweep.py bench.py
tests/         unit/ validation/ market/ adversarial/ golden/ integration/
claude/        design docs (unchanged) · claude/plan/ (this plan) · claude/marketsim/ (legacy, frozen → removed at T0.14)
```

Assumption: today's code lives at `claude/marketsim/{config,scripts,tests}`. T0.01 checks the real tree and records
the mapping before anything moves.

## 5. Stack and licence posture

Python ≥ 3.11 · numpy · scipy · pyyaml · pydantic v2 · sortedcontainers · fastapi + uvicorn + websockets ·
gymnasium · pettingzoo · orjson/msgpack (optional) · dev: pytest, pytest-xdist, ruff, hypothesis (MPL-2.0, **test-only**) ·
`[bea]` extra: pandas, openpyxl. Rust core only if profiling demands it (Phase 9).
All dependencies permissive. Own implementation of CLOB, background flow and impact model.

## 6. Conventions

**Units.** Money is `cr`. Baseline national final demand = `100 × world.scale` cr per month. Goods prices are
indices (1.0 at baseline); real quantities are in baseline-price cr per month. Rates are annual decimals
(0.02 = 2 %); `/12` per month, `/252` per day. "Per 100bp" elasticities take the gap in percentage points.

**Time.** Tick = 1 simulated day. 21 days = 1 month, 63 = 1 quarter, 252 = 1 year. The real economy steps on the
last tick of each month, policy on the last tick of each quarter, valuation and markets every tick. Firm *real*
levers take effect at the next month boundary; financial actions (orders, borrowing, dividends) the next tick.
Orders submitted after observing tick *t* are matched in tick *t+1* ("next-tick fills").

**Arrays.** Real-economy state is `(R, S)`; flows between sectors `(R, S, S)` indexed `[region, supplier, buyer]`;
inter-regional flows `(S, R_src, R_dst)`. `R = 1` in Phase 2 but the axis exists from day one. `S = 18` in the
order of `io_table.json`.

**Lags.** Every distributed lag is an Erlang(k) chain implemented by the linear-chain trick: k first-order stages,
per-stage exit probability `a = k/(m + k)` for mean lag `m` months (so mean = k(1−a)/a = m exactly; `m = 0` is a
pass-through). Two forms: **mass-conserving** (pipelines of goods, capacity, money) and **signal smoother** (typed
edges, expectations). Quarter lags in config are converted with `m = 3 × lag_q`.

**Nominal smoothing.** `s ← s·g + (x − s·g)/τ` with `g = exp(π_e/12)`. Beginning-of-month nominal stocks compared
with current-month nominal flows are scaled by `g`. Interest is paid on last month's stock.

**Determinism.** Root seed → `numpy.random.SeedSequence(root, spawn_key=(sha256(name)[:4] as int,))` → `PCG64`
per named stream (`events`, `pricing.noise`, `flow.<instrument>`, `matching`, `demand.noise`, …). No global RNG, no
`hash()`, no wall-clock, sorted iteration. `World.state_hash()` = sha256 over canonical serialisation (sorted keys,
little-endian float64 bytes). Same machine + same versions → identical hashes; across platforms compare with
`state_diff(tol)`.

**Config.** YAML/JSON in `config/`, one pydantic model per file, loaded once into an immutable `Config`. Overrides
by dotted path (`World.create(overrides={"dynamics.prices.kappa_util": 0.9})`). Domain randomisation = sampled
overrides.

**Entities and instruments.** `HH:<r>`, `NPC:<r>:<SECTOR>`, `BANKSYS`, `GOVT`, `CB`, `ROW`, `MM` (NPC investors —
the engine market maker's book, a sub-account of households), later `FIRM:<id>`, `AGENT:<id>`. Instruments: `DEP`,
`LOAN`, government bonds `GB_BILL` / `GB_NOTE` / `GB_BOND`, pooled corporate bonds `CORP_POOL` (issued by the
vehicle `CORPPOOL`, which on-lends to firms through `CLOAN`), `RES`, `EQ:<issuer>`, real assets `CAPITAL`, `INVENTORY`, `HOUSING`. Positions are signed
(+ asset, − liability); every financial instrument sums to zero across entities.

## 7. Tick pipeline (brief §3)

```
for each tick t:
  1. clock.advance(); drain due queue items (key = (timestamp, priority, sequence)):
       events, follow-ups, data releases, reports, policy meetings, order expiries, capacity completions
  2. ingest agent orders and firm decisions submitted since the last publish (lockstep barrier or real-time cut-off)
  3. if month end:  real economy step — demand → plans → orders → rationing/allocation (goods, labour, inputs)
                    → prices → wages → incomes, taxes, interest, dividends → ledger postings
     if quarter end: policy (Taylor rule, fiscal review)
  4. ledger settlement; SFC assertion + invariants (non-negativity, bounded cascades)
  5. valuation (fundamentals, discount rates) → market session (background flow, matching on MM + CLOB, impact,
     margin checks, forced liquidations) → trade postings; SFC assertion again
  6. publish: observations, fills, news, scheduled releases; append to replay log
```

An event firing on day *d* of a month enters that month's real step with weight `(21 − d)/21` and reaches the
financial layer immediately through news → expectations / risk appetite.

## 8. Config registry

| file | owner task | contents |
|---|---|---|
| `sectors.yaml` | Layer 1 (frozen) | elasticities, leverage, duration, lags, pass-through, `financials`, `market_cap_weights` |
| `edges.yaml` | Layer 1 (+T0.13) | capex routing/coefficients, credit, collateral, substitution/complement, labour, policy, 7 primitive shocks |
| `io_table.json` / `io_table_bea.json` | T0.04–T0.09 | A, mu, final-demand vectors, fd weights, source |
| `dynamics.yaml` | T2.01, T3.12 | production modes, inventory/backlog, price formation, capex unit contract, households, `demand.mode` (`scalar_eta` \| `tiers_wants`; default `scalar_eta`), labour, fiscal, ROW, NPC entry/exit (T3.14) |
| `ledger.yaml` | T2.02 | entity and instrument registry, flow tags, opening balance-sheet ratios |
| `regions.yaml` | T3.01 | regions, population, sector location quotients, links (cost, capacity), tradability, Armington σ |
| `wants.yaml`, `buy_packages.yaml` | T3.08–T3.13 | wants → sectors, need shapes, tiers (generated by the calibrator); `shift_persistence_q` (T3.13 want AR) |
| `events/*.yaml`, `scenarios/*.yaml` | T4.01, T4.08–T4.09 | event definitions, historic templates, scripted scenarios |
| `firms.yaml` | T5.01 | `npc_initial_share`; goods-market `ε_s`/`κ`/`τ_loyalty`; lever limits (§5.4: price step/floor, overtime, hire-share, firing cost, cover, premium, capex set-up/growth, ND/EBITDA 4/6); matching `m0`/`ε_w`; rating multipliers AAA…CCC; bankruptcy recoveries + `large_threshold`; report lags; optional cell-share cap |
| `markets.yaml` | T6.01 | `turnover` 0.4 %/day (`ADV = turnover × cap`); `pricing.mode` structural\|factor_lite + curve `λ=0.978`/`horizon_m=120`/`g_lr_anchor`/`τ_ee`; generated `EQ:NPC:<SECTOR>` + `IDX:<SECTOR>`; static `GB_*`/`CORP_POOL`/`CASH`/`OIL`/`METALS`/`GRAINS`; dynamic `EQ:FIRM:<id>` on CLOB; impact `δ,β,Y,τ0`, half-lives; MM `s0,k_σ,k_inv,k_skew,participation_cap`; flow AR components; CLOB tick/lot/halt/`thin_quote_*`; fees; margin 50/25 and 10/7; surveillance wash/circular/pump |
| `world.yaml` | T0.12, T4.05 | scale, seed, mode (game/professional), run mode, enabled modules, randomisation ranges; `events.damping` / `max_depth` / `max_concurrent` / `p_sum_cap` (T4.05 chains) |
| `policy.yaml` | T2.27, T2.32 | per-authority control mode, lever ranges and change limits, legislative / implementation lags, policy-agent reward weights; the `monetary:` block — calendar, reaction function, information set, strategy options, ELB toolkit, committee (§2.14.6) |
| `bonds.yaml` | T6.24 | decaying-coupon buckets `GB_BILL`/`NOTE`/`BOND`/`CORP_POOL` (δ 4 / ⅓ / 0.10 / 0.20); `pricing: par\|market` (par = Phase-2 bitwise); `duration_ref_yield` 4.2 % (GB_BOND duration 7.07 y); issuance mix 20/40/40; κ fixed at SS yield so P=1 at baseline. Auction / tp / QE keys arrive with T6.25–T6.28 |

## 9. Phase map, dependencies, gates

```
P0 foundation ──► P2 dynamic layer + ledger ──► P3 regions + demand ──► P5 firms ──► P6 pricing + markets ──► P7 API/SDK ──► P8 ──► P9
                         │                         └──────► P4 events ───────┘            ▲
                         └── Track B (market microstructure: T6.07–T6.13) may start after T2.03 ─┘
```

| phase | scope | gate ("done when") |
|---|---|---|
| 0 | packaging, loaders, bug fixes B1–B5, core utilities, World skeleton | golden parity with legacy Layer 1; determinism test; CI green |
| 1 ✔ | sector graph + IO + derived betas | kept alive by golden/validation tests |
| 2 | ledger + dynamic real economy + policy + banks + credit/collateral edges | exact stationarity at π\*∈{0, 2 %}; 7 sign-restriction IRFs; hump + timing (housing before capital goods); 100-year stability; SFC every month; policy levers bitwise-neutral on autopilot, lever tests and extreme-policy boundedness (§2.13); one borrowing rate for all firms; monetary-framework parity and validation (§2.14.7) |
| 3 | regions + trade; tiers × wants; NPC entry/exit | R=1 parity; regional shocks propagate, prices converge within band; basket reproduced exactly, η rank-corr ≥ 0.8; aggregate baseline recorded |
| 4 | events, cascades, news, releases, templates, scenarios | cascades bounded (sub-critical branching); templates reproduce historic response directions |
| 5 | firms + heterogeneous-seller markets + financing + bankruptcy | hybrid ≈ aggregate; adversarial + monopoly tests; one agent on 2–3 levers in-process |
| 6 | L3 valuation, ξ, impact, MM, CLOB, cap tables, control, margin | Tier-2 metrics in CI (√-law slope 0.4–0.7, partial reversion); stock–bond correlation flips sign by regime; takeover + liquidation-cascade tests; bond market gates (§6.11): par-mode parity, curve and auction behaviour, QE effect, insurer/bank mark-to-market emerges |
| 7 | REST/WS, lockstep/real-time, SDK, gym/PettingZoo, replay | multi-agent training run end-to-end; ≥ 1,000 ticks/s small world in-process; deterministic replay |
| 8 | realism + game | playable campaign; calibration report |
| 9 | scale-out | — |

Parallel work: within a phase, cards without a dependency path between them can run concurrently. Track B only
needs `core/` + `ledger/`. Human gates (marked **HUMAN GATE** in cards): T0.14, T2.26, T3.16, T4.13, and every
phase gate review.

## 10. Task cards

Format: `### T<phase>.<nn> — title` · **Depends** · **Size** (S ≤ ~150 LOC, M ≤ ~400, L ≤ ~800 incl. tests) ·
**Files** · **Read first** · **Build** · **Tests** (must pass) · **Out of scope**. Cards are written so an agent with no
memory of this conversation can execute them with the phase spec open. Phases 8–9 are coarse on purpose; refine
them into cards of this granularity after the Phase-7 gate.

## 11. Test strategy

- `pytest -m "not slow"` < 2 min: unit, golden, fast validation (single IRFs), determinism, SFC.
- `-m validation`: economic behaviour (IRFs, timing, rotation, regional convergence, demand-system properties).
- `-m slow` (nightly): 100-year stochastic runs, parameter sweeps, event storms, Tier-1/2 market statistics, throughput.
- Invariants asserted every tick in tests: SFC (instrument sums, balanced tx, Δstock = flow), non-negativity of
  physical stocks/prices/employment, finite values, bounded cascade depth/concurrency, determinism hash.
- Validation tiers for markets (landscape doc): T1 distributional facts (necessary, near-worthless alone);
  **T2 response/impact is the bar**; T3 counterfactual validity is a Phase-8 research item.

## 12. Open decisions — defaults the agents must use until the human changes them

| # | question | default |
|---|---|---|
| 1 | Control transfer at > 50 % voting shares? Non-voting shares in v1? | yes; no |
| 2 | Levers live for agents in v1 | price, production target, capex (everything else on autopilot, all levers implemented) |
| 3 | Information policy default | `professional` for training worlds, `game` selectable per world |
| 4 | Tick / cadence | 1 day; real monthly; policy quarterly |
| 5 | v1 asset universe | 18 sector-equity instruments, agent-firm shares, 3 government bond buckets (`GB_BILL`, `GB_NOTE`, `GB_BOND`), the corporate bond pool (`CORP_POOL`), cash, 3 commodities (OIL←ENERGY, METALS←MATERIALS, GRAINS←AGRIFOOD) |
| 6 | Stack | Python + NumPy, FastAPI; Rust later if profiling demands |
| 7 | RL reward | operators: Δ(equity value) + dividends; traders: risk-adjusted P&L; both report net worth |
| 8 | Calibration flavour | generic developed economy (BEA IO, FRED/BLS moments); Korea-flavoured set optional later |
| 9 | Goods layer beneath consumer-facing sectors | **no in v1** (ADR-006 proposed T3.16); T8.12 skipped unless accepted |
| 10 | Commercial use | in scope → licence rule in AGENTS.md |
| 11 | Who runs `GOVT` and `CENBANK` | autopilot; `scripted` for scenarios; `agent` only when a client registers the `policymaker` role. A policymaker cannot also hold a trading account in professional mode |
| 12 | Scope of the common borrowing rate | all firms (NPC cells and agent firms), bank loans and the corporate pool, all regions; households and the government are **not** covered. In a future multi-country world (T9.01) it becomes one rate per currency unless decided otherwise |
| 13 | Sovereign default | none in v1 (own-currency issuer); fiscal stress shows up as a term premium rising with debt/GDP and issuance |
| 14 | Monetary strategy | flexible inflation targeting with a dual mandate; makeup strategies shipped but **off** (see §2.14.3) |
| 15 | Lean vs clean | credit and house prices act on macroprudential levers, not the policy rate (`phi_credit: 0`) |

## 13. What the prototype established (numbers the Phase-2 gate is built on)

All with the unchanged Layer-1 config, single region, monthly step, no ledger/banks:

- **Exact steady state**: no-shock run stationary to 1e-14 in real terms at π\* = 0 and 2 %; inflation = π\*;
  household wealth = Σ liabilities. Needs: closed-form initialiser with leak-adjusted Leontief solve, imports carved
  out of value added and scaled to balanced trade, per-sector payout giving zero real net borrowing (0.50–0.61),
  tax rate and wealth propensity solved as residuals, interest on last month's stock, ROW prices drifting at π\*,
  seeded price histories, drift-compensated nominal smoothing.
- **Capex unit contract**: φ = 1.2, ψ = 0.8, χ = 0.45 work *as written* when read as **percentage points of the
  capital stock per year**, and only together with four stabilisers: supply-line accounting (weight 0.85), anchored
  growth expectations (0.75), income smoothing (12 months), a household-driven residential block (20 % of I).
  Without the supply line the economy is unstable at every accelerator scale tested.
- **Stock-flow norms are required**: a fiscal reaction function (κ_B ≈ 0.04/yr; without it r > g snowballs) and a
  **slow** firm leverage norm through dividends (κ_D ≈ 0.03/yr; 0.10 under-damped, 0.30 explosive).
- **Price tightness κ_u = 1.2** is the main damper (monetary rebound ratio 0.85 → 0.33 when raised from 0.6).
- **Nominal income illusion is a damper**: deflating smoothed income by realised CPI raises the monetary rebound
  ratio from 0.44 to ≈1.0 and doubles inflation volatility. Hence design rule 9.
- **IRFs** (π\* = 2 %): monetary +100bp (4q) → GDP trough −0.30 % at month 16, rebound 0.44, sector troughs AUTOS
  m11 < CONSTRUCT m17 < CAPGOODS m20. Demand +2 % → peak +1.3 %. Cost-push ENERGY +30 % → CPI +2.0 % at 12m, GDP
  trough −2.7 % at month 22 (**large** — calibration item, not to be fixed with caps). All six tested sign
  restrictions pass on price-*level* windows (12-month inflation overshoots later, so level windows are the test).
- **100-year stochastic runs** bounded: gap ∈ [−2.2, +3.0] %, U 3.1–6.4 %, inflation −0.7…+5.0 %, never at the ELB.
  Volatility ranking emerges: SEMIS ≫ CONSTRUCT ≈ CAPGOODS > … > INSURANCE ≈ TELECOM ≈ BANKS.
  Tuning targets still open: sd(I)/sd(GDP) ≈ 7 (target 3–4), sd(C)/sd(GDP) ≈ 1.45 (target < 1).
- **Leak sweep**: with the proportional stock-gap controller the loop is bounded for leak ×0, ×1, ×3. The V3
  experiment diverged because of an integrating controller. Leaks stay (decision), but stability does not rest on them.
- A prototype bug (clipping firm debt at zero) silently leaked money — the practical case for ledger-first.
- **One borrowing rate (D14) re-verified** (`credit_pricing="uniform"` in the reference prototype): exact steady state at
  π\* ∈ {0, 2 %}; all sign tests pass; monetary +100bp → trough −0.29 % at month 16, rebound 0.41, AUTOS m10 < CONSTRUCT m17 <
  CAPGOODS m19; 100-year runs bounded (gap −2.2…+3.0 %, U 3.0–6.4 %). Aggregate dynamics barely change; what is lost is the
  *price* penalty on leverage, so quantity limits carry the discipline. Bond arithmetic checked: the 10-year bucket (decay
  0.10/yr) has duration 7.07 years — equal to `bond_index.duration` — and loses 2.4 % on a +35bp yield move.
- **Monetary framework (D15) tested in the prototype.** Dual mandate (φ_u 1.0), 8 meetings, 25bp grid with a 10bp deadband,
  1-month data lag and r\* on trend growth: steady state still exact, all sign tests pass, monetary trough −0.28 % at month
  15, 100-year runs bounded at ≈ 4.1 rate moves a year averaging 28bp. φ_u 2.0 is too aggressive (sd(π) 1.10 → 1.24).
  Acting on published data instead of the truth costs little and monotonically: sd(π) 1.08 → 1.21 as the lag goes 0 → 3
  months. Reacting to core rather than headline inflation cut the cost-push GDP trough from −2.86 % to −2.56 %.
  **An unbounded makeup term destabilised the economy** (−93 % output gap, 73 % unemployment); leaky (0.98/month) and
  clipped (±2 pp) it is bounded — the same lesson as the undamped inventory stock.

## 14. Deliberate differences from the source docs

| source | says | plan does | why |
|---|---|---|---|
| brief §14 | Phase 2 first | adds Phase 0 | bugs B1–B5, packaging, determinism, golden parity first |
| brief §14 | ledger arrives with firms (Phase 5) | general ledger built in Phase 2; Phase 5 only registers entities | design rule 5; prototype leak |
| `edges.yaml` | credit edges `elasticity` per 1 %/100bp | spreads enter cost of capital structurally; edge elasticities act as **rationing exponents on lending capacity** (=1 at baseline) | avoids double counting with χ and `dem_rate_semi` |
| `sectors.yaml` | `dem_rate_semi` = own final demand per 100bp | applied to the HOUSEHOLD component only; investment responds through the capex function; total simulated sensitivity validated by rank correlation | avoids double counting |
| `engine-design.md` | bullwhip needs leaky inventories | leak kept; stability from proportional controller | §13 leak sweep |
| brief §7 | sector equities include listed agent firms | tradable instrument = NPC sector equity; *published index* is cap-weighted incl. listed agent firms | finite float, SFC-clean |
| landscape | separate matching-engine process | in-process `Venue` interface | D7, daily ticks |
| brief §4.6 | VAT | implemented, default rate 0 (baseline verified without it) | keep verified steady state |
| `edges.yaml` | `credit.spread_scaling: nd_ebitda / 2.5` | unused under `credit.pricing: uniform` (default); active only in `risk_based` | D14 |
| brief §7, §5.5 | corporate debt = rate + spread by rating/leverage | one global spread; ratings drive credit **limits** and disclosure, not price | D14 |
| brief §4.6, `edges.yaml: policy` | government and central bank follow fixed rules | the rules are the *autopilot* of policy authorities with levers | D13 |
| `edges.yaml: policy.taylor` | quarterly rule on inflation and the output gap | 8-meeting committee, dual mandate, core blend, time-varying r\*, 25bp grid, published vintages; `phi_inflation` and `smoothing` still come from this block | D15 |
| brief §7 | government bonds priced from the expected path + term premium | same fair value, plus a real market: maturity buckets, auctions, secondary trading, central-bank operations, supply effects | D12 |

## 15. Risk register

| risk | mitigation |
|---|---|
| Instability when a new feedback is switched on (accelerator, credit, wealth effect, regional trade) | every feedback behind a config switch; stability suite re-run per switch; sweeps document the stable region |
| Money leaks | ledger-first; SFC assertion every posting batch; no clipping |
| Double counting of rate channels | §14 rows 3–4; validation by rank correlation against config |
| Seed IO table used quantitatively | B1 fix; BEA fetch + final-demand concordance (T0.09, T8.04); results labelled "seed" until then |
| Cost-push too strong, I/GDP volatility too high | calibration cards T2.24, T8.04; never patch with level caps |
| Python throughput (≥1,000 ticks/s) | real step monthly (amortised), vectorised market tick, debug journal off in training, profile before Rust |
| RL agents exploiting the simulator | hidden state, noise, domain randomisation, position limits, adversarial tests, surveillance |
| Agent-firm abuse (zero prices, cornering, max leverage, self-pumping, collusion) | hard budget constraint, leverage cap, size limits, NPC entry, share stickiness, bankruptcy in bounded time, fixed-NPC evaluation |
| Moral hazard from one borrowing rate (max leverage is not priced; default losses are mutualised) | leverage caps, collateral limits, gate rationing, bounded bankruptcy; losses raise the common spread for everyone (visible externality); optional regulator levy (T8.08); adversarial max-leverage test stays |
| A policy agent (or player) drives the economy to extremes | lever ranges and per-period change limits, legislative lags, everything financed on the ledger, bounded price steps; adversarial policy test (T2.31) |
| Integrator terms in policy rules (makeup strategies, any cumulative gap) | leak and clip mandatory, rejected at config load otherwise; stability suite per strategy (T2.34) |
| Bond ledger complexity (many issues and coupons) | three fungible decaying-coupon buckets instead of individual issues; par pricing until Phase 6 so Phases 2–5 are unaffected |
| Scope creep | Mizuta's principle; no subsystem without a card; human gates |

## 16. Glossary

**A** technical coefficients (`a_ij` input from i per 1 cr output of j) · **L** = (I−A)⁻¹ demand-pull · **G** = (I−Aᵀ)⁻¹
cost-push · **NPC mass** aggregate producer per (region, sector) cell · **SFC** stock-flow consistent · **BSM/TFM**
balance-sheet / transaction-flow matrix · **ξ** log mispricing (price = V·exp ξ) · **ELB** effective lower bound ·
**cover** inventory / expected monthly sales · **supply line** capacity under construction · **gate** state-dependent
multiplier on an edge · **Tier 1/2/3** distributional / response-impact / counterfactual validation.
