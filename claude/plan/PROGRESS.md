# PROGRESS — the work queue

Rule: take the **lowest-numbered unchecked task whose dependencies are all checked** (or the one the human names).
Tick it as `- [x] T2.07 — … — 2026-10-02, short note`. One task per session. Gates are ticked by the human.
Sizes: S ≤ ~150 LOC · M ≤ ~400 · L ≤ ~800 (incl. tests). Cards live in the phase files; `deps` repeats the card.
v1.1 added T2.27–T2.31, T4.14, T6.24–T6.31, T7.16–T7.17 · v1.2 added T2.32–T2.35 (monetary framework) and T6.32.

## Phase 0 — Foundation (15 tasks) — `01-phase0-foundation.md`

- [x] T0.01 — Repo audit and layout decision record · S · deps: — — 2026-09-19, ADR-000: no legacy tree; layout is `marketsim/`
- [x] T0.02 — Packaging and tooling · M · deps: T0.01 — 2026-09-19, nested package + Makefile
- [x] T0.03 — Typed config loading · M · deps: T0.02 — 2026-09-19
- [x] T0.04 — Single IO loader (fixes B1) · S · deps: T0.03 — 2026-09-19
- [x] T0.05 — Port `build_io` into the library · S · deps: T0.04 — 2026-09-19, reconstructed seed (nnz 219)
- [x] T0.06 — Port `derive_betas` as pure functions (fixes B2) · M · deps: T0.04 — 2026-09-19, invariant + regime windows
- [x] T0.07 — Layer-1 validation suite in pytest · M · deps: T0.05, T0.06 — 2026-09-19
- [x] T0.08 — `vic3_compare` as a regression test · S · deps: T0.02 — 2026-09-19, self-contained
- [x] T0.09 — BEA fetch hardening · M · deps: T0.04, T0.07 — 2026-09-19, script only, no network
- [x] T0.10 — Calendar, clock and event queue · M · deps: T0.02 — 2026-09-19
- [x] T0.11 — RNG streams, state protocol, hashing · M · deps: T0.02 — 2026-09-19
- [x] T0.12 — Erlang kernels · S · deps: T0.02 — 2026-09-19
- [x] T0.13 — World skeleton and module pipeline · M · deps: T0.10, T0.11, T0.12 — 2026-09-19
- [x] T0.14 — Config corrections B3–B5 with ADRs · S · deps: T0.03 — 2026-09-19, ADR-001…003 proposed
- [x] T0.15 — Legacy freeze and removal · S · deps: T0.05, T0.06, T0.07, T0.08 · **HUMAN GATE** — 2026-09-19, N/A (no legacy tree; see QUESTIONS.md)

- [x] **GATE P0** — 2026-09-19, accepted by human; Phase 2 unblocked

## Phase 2 — Ledger + dynamic real economy + policy authorities + monetary framework (35 tasks) — `02-phase2-dynamic-layer.md`

- [x] T2.01 — `dynamics.yaml` and its schema · S · deps: T0.14 — 2026-09-19, §2.11 defaults + pydantic
- [x] T2.02 — Ledger core · M · deps: T0.11 — 2026-09-19, pos matrix + balanced post
- [x] T2.03 — Balance-sheet and transaction-flow matrices, SFC assertion · M · deps: T2.02 — 2026-09-19
- [x] T2.04 — Steady state, real side · M · deps: T2.01, T0.04, T0.12 — 2026-09-19
- [x] T2.05 — Steady state, financial side and opening postings · M · deps: T2.04, T2.03 — 2026-09-19, passthrough opening + α2>0
- [x] T2.06 — Production plans and capacity caps (R1, R4) · M · deps: T2.04 — 2026-09-19, plan=x0; ENERGY→UTILITIES critical
- [x] T2.07 — Orders, rationing, deliveries, input stocks (R5–R6) · M · deps: T2.06 — 2026-09-19, three modes + conservation
- [x] T2.08 — Price formation (R7) · M · deps: T2.04 — 2026-09-19, step bound 0.15; G ranking
- [x] T2.09 — Labour and wages (R8) · S · deps: T2.06 — 2026-09-19, hire/fire + 4× downward stickiness
- [x] T2.10 — Capex, capacity pipeline, supply line (R3, §2.6) · M · deps: T2.04, T0.12, T0.14 — 2026-09-19, −0.45pp/100bp; ENERGY lag > CONSTRUCT
- [x] T2.11 — Residential investment block · S · deps: T2.10 — 2026-09-19, 20% of I; −4%/100bp
- [x] T2.12 — Household sector (aggregate, scalar-η demand system) · M · deps: T2.05 — 2026-09-19, θ=HH FD; AUTOS most rate-sensitive
- [x] T2.13 — Government and fiscal rule · S · deps: T2.05 — 2026-09-19, deficit=grow·B; debt rule mean-reverting
- [x] T2.14 — Central bank and inflation expectations · S · deps: T2.04 — 2026-09-19, quarterly Taylor + seeded π12
- [x] T2.15 — ROW · S · deps: T2.04 — 2026-09-19, balanced TB; +10% p → exports −7.3%
- [x] T2.16 — Income settlement on the ledger (R9) · L · deps: T2.03, T2.07, T2.08, T2.09, T2.10, T2.11, T2.12, T2.13, T2.14, T2.15 — 2026-09-19, 120m SFC; clip-debt regression
- [x] T2.17 — Monthly orchestrator, aggregates, World integration · M · deps: T2.16, T0.13 — 2026-09-19, 360m stationarity; μ=A.sum(0)
- [x] T2.18 — Shock bus and the seven primitives · M · deps: T2.17 — 2026-09-19, AR(1) bus; catastrophe conserves claims
- [x] T2.19 — Banking system (`banks.mode: full`) · M · deps: T2.17 — 2026-09-19, full opening; gate 1; NIM rises with r
- [x] T2.20 — Credit and collateral edges · M · deps: T2.19, T2.21 — 2026-09-19, gate+Λ; crunch |gap| 0.29% y12
- [x] T2.21 — AssetPriceProvider stub · S · deps: T2.17 — 2026-09-19, Q=1; +100bp: RE −6.3%, AUTOS −2.5%
- [x] T2.22 — Typed substitution / complement edges · M · deps: T2.12, T0.12 — 2026-09-19, ENERGY+20%: AUTOS −5.3%, UTIL +6.6%
- [x] T2.23 — Validation suite: stationarity, IRFs, timing · M · deps: T2.18 — 2026-09-19, §2.12 green passthrough/banks; credit+cat see QUESTIONS
- [x] T2.24 — Stability runs, sweeps, moments report · M · deps: T2.23 — 2026-09-19, 48 cells bounded; ranking see QUESTIONS
- [x] T2.25 — Performance baseline · S · deps: T2.17 — 2026-09-19, ~10.6k ticks/s at R=1
- [x] T2.26 — CES substitution on intermediates (optional, off by default) · M · deps: T2.23 · **HUMAN GATE** — 2026-09-20, ADR-009 proposed: skip; Leontief + T2.22 edges suffice
- [x] T2.27 — Policy-authority framework (D13) · M · deps: T2.13, T2.14, T2.18 — 2026-09-19, PolicyDesk + policy.yaml; autopilot hash; clip/lag/arbitration
- [x] T2.28 — Fiscal instruments · M · deps: T2.27, T2.16 — 2026-09-19, VAT/excise/tariff wedges; rescue; GOVT DEP covered
- [x] T2.29 — Monetary and macroprudential instruments · M · deps: T2.27, T2.20 — 2026-09-19, rate at meetings; cap req moves gate; LOLR RES/LOAN; QE stub
- [x] T2.30 — One corporate borrowing rate (D14) · S · deps: T2.20 — 2026-09-19, uniform r+s_t; SS both modes; one-sector default
- [x] T2.31 — Policy validation · M · deps: T2.28, T2.29, T2.23 — 2026-09-19, multiplier 0.66; VAT 2.00pp; extremes finite/SFC
- [x] T2.32 — Monetary policy framework: committee, calendar, reaction function (D15) · M · deps: T2.27, T2.29 — 2026-09-19, 8-meeting dual mandate; SS exact; 100y 3–6 moves; see QUESTIONS
- [x] T2.33 — The committee's information set (published vintages) · S · deps: T2.32, T4.07 — 2026-09-20, CPI/U 1m GDP 1q; poison unpublished; oracle flag
- [x] T2.34 — Strategy options: makeup, risk management, financial conditions · M · deps: T2.32, T2.20 — 2026-09-19, leak+clip reject; Sahm/FCI/ELB; 100y marked slow
- [x] T2.35 — Monetary validation and the policy-rule report · M · deps: T2.33, T2.34, T2.31 — 2026-09-20, §2.14.7; trough m14 −0.41%; 4.65 moves/y 30bp; sacrifice ≈5.1

- [ ] **GATE P2** — human review against the gate in the phase file — 2026-09-20, T2.35 in; CES skipped (ADR-009); not human-accepted

## Phase 3 — Regions and demand layer (16 tasks) — `03-phase3-regions-demand.md`

- [x] T3.01 — `regions.yaml` and geometry · S · deps: T2.23 — 2026-09-20, §3.2 schema; R=1 accepted
- [x] T3.02 — Regionalise the state (R = 1 parity) · M · deps: T3.01 — 2026-09-20, (R,S) live state; Phase-2 golden 1e-12
- [x] T3.03 — Baseline trade shares and regional steady state · M · deps: T3.02 — 2026-09-20, T0 + stacked Leontief; national totals 1e-9
- [x] T3.04 — Regional orders, rationing and link capacity · M · deps: T3.03 — 2026-09-20, spill + source ration; R=1 SFC ok
- [x] T3.05 — Trade-share dynamics and regional prices · M · deps: T3.04 — 2026-09-20, Armington + gate 3 band
- [x] T3.06 — Regional labour pools and migration · S · deps: T3.02 — 2026-09-20, LF conserved; 0.1%/yr/pp
- [x] T3.07 — Regional households and national government · S · deps: T3.02 — 2026-09-20, YD sums; G by pop; HH:r SFC
- [x] T3.08 — Tiers · S · deps: T2.12 — 2026-09-20, ι sum 10; Gini 0.38–0.42
- [x] T3.09 — Want layer and within-want allocation · M · deps: T3.08 — 2026-09-20, two-shape + SEMIS in HOUSEHOLD_GOODS
- [x] T3.10 — Need shapes and budget scaling · S · deps: T3.08 — 2026-09-20, five shapes; survival first
- [x] T3.11 — Calibrator · M · deps: T3.09, T3.10 — 2026-09-20, RAS IPF + frozen-M LS; Spearman 0.816; basket 1e-15
- [x] T3.12 — Swap in `TiersWantsDemand` · M · deps: T3.11, T3.07 — 2026-09-20, demand_mode default scalar_eta; §3.4; see QUESTIONS
- [x] T3.13 — Want shifters (interface for events) · S · deps: T3.12 — 2026-09-20, want_shift[r,q] AR; ShockBus kind=want
- [x] T3.14 — NPC entry / exit · M · deps: T3.02 — 2026-09-20, §3.5; gate 5 AUTOS 8y; SS bitwise
- [x] T3.15 — Regional validation and golden baseline · M · deps: T3.05, T3.12, T3.14 — 2026-09-20, gates 3/6/7; R=3 tick see QUESTIONS
- [x] T3.16 — ADR: goods layer beneath consumer-facing sectors · S · deps: T3.15 · **HUMAN GATE** — 2026-09-20, ADR-006 proposed: no goods layer in v1

- [ ] **GATE P3** — human review against the gate in the phase file — 2026-09-20, ADR-011 proposed (mechanism met; R=3 stepper deferred, QUESTIONS T3.15)

## Phase 4 — Events (14 tasks) — `04-phase4-events.md`

- [x] T4.01 — Event schema and loader · M · deps: T2.18 — 2026-09-20, pydantic DistSpec + extras whitelist; catalog skips `_*.yaml`
- [x] T4.02 — Shock-composition engine · M · deps: T4.01 — 2026-09-20, truncated DistSpec → ShockBus; SEMIS mask isolated
- [x] T4.03 — `effects_extra` executor · M · deps: T4.02, T3.13 — 2026-09-20, shapes revert; bank_equity/vat SFC
- [x] T4.04 — Hazard model · S · deps: T4.01 — 2026-09-20, h=base/252·Πclip(exp); 2000y rate ±10%; cooldown
- [x] T4.05 — Scheduler and chains · M · deps: T4.02, T4.04, T0.10 — 2026-09-20, DAG check; depth/concurrency caps; gate 4 RNG
- [x] T4.06 — News feed · S · deps: T4.05 — 2026-09-20, no magnitudes; lag; noise≈misclass; rumours debug-only
- [x] T4.07 — Data release calendar · M · deps: T2.17, T0.10 — 2026-09-20, CPI d10 / U d5 / GDP q+rev; observe vintages only
- [x] T4.08 — Historic templates and generic events · M · deps: T4.03 — 2026-09-20, 11 templates + generics; DAG Σp≤0.9
- [x] T4.09 — Scenario runner · S · deps: T4.05 — 2026-09-20, scripted ticks; hazards off; RNG isolation
- [x] T4.10 — Cascade and storm tests · S · deps: T4.08 — 2026-09-20, chip cascade ±20%; 20y 10× storm |gap|<25% (oil/energy/covid omitted, QUESTIONS T4.11)
- [x] T4.11 — Template direction tests · M · deps: T4.08, T4.09 — 2026-09-20, §4.5 on/off; oil/energy 12m prefix; covid want-window; see QUESTIONS
- [x] T4.12 — Firm-level event hooks · S · deps: T4.03 — 2026-09-20, FirmHookBus no-op; strike/recall/accident/bankruptcy
- [x] T4.13 — Magnitude calibration · M · deps: T4.11 · **HUMAN GATE** — 2026-09-20, ADR-007 proposed; sources filled; verify cleared; z-mapping is T8.04
- [x] T4.14 — Policy events under scripted or agent-controlled authorities (D13) · S · deps: T4.05, T2.27 — 2026-09-20, follow-up block + pressure news; transfer_oneoff; gate 5 modes

- [ ] **GATE P4** — human review against the gate in the phase file — 2026-09-20, ADR-008 proposed (mechanism met; oil/energy level mapping is T8.04)

## Phase 5 — Agent-operated firms (19 tasks) — `05-phase5-firms.md`

- [x] T5.01 — `firms.yaml` and firm state · M · deps: T3.15 — 2026-09-20, FirmsFile + Plant/Firm registry; unknown cell rejected
- [x] T5.02 — Firm accounts on the ledger · M · deps: T5.01, T2.03 — 2026-09-20, FIRM:/AGENT: + equity; 24m A=L+E SFC
- [x] T5.03 — Founding and entry · S · deps: T5.02 — 2026-09-20, NPC purchase conserves K; greenfield after build_lag_q
- [x] T5.04 — Cell aggregation with firms (incl. zero-NPC cells) · M · deps: T5.03 — 2026-09-20, aggregates = sums; 120m zero-NPC NaN-free
- [x] T5.05 — Heterogeneous-seller goods market · M · deps: T5.04 — 2026-09-20, equal-p ~ K; 5% cut gradual; spill conserves D
- [x] T5.06 — Decision levers and validation · M · deps: T5.04 — 2026-09-20, clip list; partial autopilot; foreign operator rejected
- [x] T5.07 — Autopilot · M · deps: T5.06 — 2026-09-20, R4/R7/R8/§2.6/R9 wrappers; 1e-12 vs Phase-2
- [x] T5.08 — Labour market with matching · M · deps: T5.06 — 2026-09-20, Σhires≤pool; wage ranks; fire cost; V cap
- [x] T5.09 — Procurement and shortage allocation · M · deps: T5.05 — 2026-09-20, pro-rata vs bid/rel; premium posted; order cap
- [x] T5.10 — Financing, rating, tax, hard budget constraint · M · deps: T5.02, T2.20 — 2026-09-20, uniform r+s; rating×Λ limits; seniority; tax carry; cash floor fuzz
- [x] T5.11 — Plants, capex and R&D · M · deps: T5.06 — 2026-09-20, ENERGY 36m vs BIZSVC 6m; routing; firms.rnd
- [x] T5.12 — Bankruptcy resolution · M · deps: T5.10 — 2026-09-20, one-step waterfall; loss≤exposure; SFC; large event
- [x] T5.13 — Reports and information rules · S · deps: T5.02, T4.07 — 2026-09-20, monthly lag 10d; blackout professional-only
- [x] T5.14 — Abuse controls and optional regulator · S · deps: T5.06, T5.10 — 2026-09-20, each control; fine→GOVT
- [x] T5.15 — Firm-level events · S · deps: T4.12, T5.11 — 2026-09-20, accident/strike/recall + firm_*.yaml
- [x] T5.16 — Hybrid ≈ aggregate test · M · deps: T5.07, T5.08, T5.09 — 2026-09-20, N∈{1,5} plan identity 1e-9; World stepper see QUESTIONS
- [x] T5.17 — Adversarial, monopoly and cascade tests · M · deps: T5.12, T5.14 — 2026-09-20, floor/corner/leverage/default; +30% share loss; cascade
- [x] T5.18 — Minimal single-agent loop (in-process) · S · deps: T5.07 — 2026-09-20, World.submit FirmDecision; hash-stable
- [x] T5.19 — Collusion evaluation harness · S · deps: T5.18 — 2026-09-20, persist-above-benchmark flag

- [ ] **GATE P5** — human review against the gate in the phase file — 2026-09-20, ADR-012 proposed (cell hybrid; World wire deferred)

## Phase 6 — Asset pricing, markets and the bond market (32 tasks) — `06-phase6-pricing-markets.md`

- [x] T6.01 — `markets.yaml` and the instrument registry · S · deps: T2.03 — 2026-09-20, ADV=0.4%/day; EQ:NPC×18 + dynamic EQ:FIRM
- [x] T6.02 — Earnings expectations from public information · S · deps: T4.07, T2.21 — 2026-09-20, published-only EMA; exact at π*∈{0,2%}
- [x] T6.03 — Yield curve, term premium, bond pricing · M · deps: T2.14 — 2026-09-20, λ=0.978 → y10 +35bp; −D·Δy + carry
- [x] T6.04 — Discount rates and fundamental value · M · deps: T6.02, T6.03 — 2026-09-20, structural ρ+V behind stub interface; factor_lite; live Δρ=0.35
- [x] T6.05 — Betas-emerge validation · M · deps: T6.04 — 2026-09-20, gate 1 Spearman 0.988/1/1/1; BANKS only +rate
- [x] T6.06 — Mispricing: sentiment, noise, limits to arbitrage · M · deps: T6.04 — 2026-09-20, ξ=I+s+n; calm vol 15–18%; θ∝A
- [x] T6.07 — Impact kernel · M · deps: T6.01 — 2026-09-20, NNLS 5-exp within 10% of G(τ); post-impact fills
- [x] T6.08 — Engine market maker · M · deps: T6.07 — 2026-09-20, next-tick fills; spread/inv/skew; Venue=CLOB
- [x] T6.09 — Background order flow · M · deps: T6.08 — 2026-09-20, Σ AR(1) sign memory; E[|q|]≈ADV; no MM posts
- [x] T6.10 — CLOB · L · deps: T6.01 — 2026-09-20, price-time; stop/IOC/GTC/DAY; STP newest; call auction; halt band
- [x] T6.11 — CLOB liquidity: thin engine quote + queue-reactive-lite · M · deps: T6.10, T6.09 — 2026-09-20, thin quote V·(1±w); 10y book never empty
- [x] T6.12 — Settlement, fees, taxes · S · deps: T6.08, T6.10 — 2026-09-20, cash-for-asset Tx; fees/tax; NPC div; VM nets 0
- [x] T6.13 — Margin, shorting, forced liquidation · M · deps: T6.12 — 2026-09-20, 50/25 & 10/7; borrow fee↑util; waterfall MM→BANKSYS; gate 6
- [x] T6.14 — Cap tables and corporate actions · M · deps: T5.02, T6.12 — 2026-09-20, Σ longs=SO; record+2; buyback cancels; issuer −EQ
- [x] T6.15 — Listing and IPO auction · S · deps: T6.14, T6.10 — 2026-09-20, CLOB call; primary→firm / secondary→founder; reserve; 49% warn
- [ ] T6.16 — Control transfer and the takeover test · M · deps: T6.15, T5.06
- [x] T6.17 — Agent-firm valuation and thin quote feed · S · deps: T6.04, T5.13 — 2026-09-20, published-only V; thin quote ±w
- [x] T6.18 — Commodities · S · deps: T6.08 — 2026-09-20, OIL=ENERGY+carry; 3× cost-push uncapped
- [ ] T6.19 — Feedback edges 4 → 1/2 · M · deps: T6.06, T2.24
- [x] T6.20 — Surveillance and limits · M · deps: T6.12 — 2026-09-20, wash/circular/pump flags; honest MM clean
- [x] T6.21 — Tier-1 / Tier-2 statistics and the regime flip · M · deps: T6.09, T6.06, T6.07 — 2026-09-20, gates 2–4; δ∈[0.4,0.7]; demand corr<−0.15
- [x] T6.22 — Domain randomisation and hidden state · S · deps: T6.06 — 2026-09-20, stream randomise; observe() whitelist
- [x] T6.23 — Market performance · S · deps: T6.21 — 2026-09-20, 25 names 15.1k ticks/s; hash-stable
- [x] T6.24 — Bond buckets: instruments, arithmetic, par ↔ market switch (D12) · M · deps: T6.01, T2.16 — 2026-09-20, P=(κ+δ)/(y+δ); GB_BOND 7.07y; par bitwise; reval ≠ income
- [x] T6.25 — Bucket fair yields and term premium · M · deps: T6.03, T6.24 — 2026-09-20, ω-weighted path; tp debt/QE/FTQ; guidance×credibility; gate 10
- [x] T6.26 — Debt-management office and auctions · M · deps: T6.25, T2.28 — 2026-09-20, uniform-price; gate 12; 20/40/40; buybacks
- [x] T6.27 — Secondary bond market and NPC holders · M · deps: T6.24, T6.08, T6.09 — 2026-09-20, engine MM; ADV∝face; NPC share leak; ROW sell-off; gate 11 hook
- [x] T6.28 — Central-bank operations: QE, QT, open-market operations · M · deps: T6.27, T2.29 — 2026-09-20, 4 postings; gate 13–14; remit; CORP_POOL easing
- [x] T6.29 — Corporate bond pool · M · deps: T6.24, T2.30, T5.10 — 2026-09-20, y_match+s_t; NAV write-down; same terms all firms
- [x] T6.30 — Financial-sector bond holdings and mark-to-market · S · deps: T6.27 — 2026-09-20, market mode off §6.3 overlays; gate 15; BANKS still +rate
- [ ] T6.31 — Bond-market validation report · S · deps: T6.26, T6.28, T6.29, T6.30
- [x] T6.32 — Policy surprises and announcement effects · S · deps: T6.25, T2.32 — 2026-09-20, anticipated flat; surprise hike front>bond; R²<0.2

- [ ] **GATE P6** — human review against the gate in the phase file

## Phase 7 — API and SDK (17 tasks) — `07-phase7-api-sdk.md`

- [x] T7.01 — Versioned API schemas · M · deps: T6.12, T5.06 — 2026-09-20, v1 closed models; golden; Observation no hidden state
- [x] T7.02 — Sessions, worlds, accounts · M · deps: T7.01 — 2026-09-20, WorldManager; SHA-256 tokens; AGENT capital_transfer; isolation/auth/limits
- [ ] T7.03 — In-process client · S · deps: T7.02
- [ ] T7.04 — REST endpoints · L · deps: T7.02
- [ ] T7.05 — WebSocket streams · M · deps: T7.04
- [ ] T7.06 — Lockstep barrier and real-time pacing · M · deps: T7.02
- [ ] T7.07 — Python SDK (HTTP / WS) · M · deps: T7.04, T7.05
- [ ] T7.08 — Gymnasium single-agent environment · M · deps: T7.03
- [ ] T7.09 — PettingZoo parallel environment · M · deps: T7.08
- [ ] T7.10 — Replay, export, save / load · M · deps: T7.06
- [ ] T7.11 — Scenario packs, curricula, evaluation sets · M · deps: T4.09, T7.08
- [ ] T7.12 — Vectorised parallel worlds · M · deps: T7.08
- [ ] T7.13 — Throughput and profiling · S · deps: T7.12
- [ ] T7.14 — End-to-end multi-agent training smoke run · M · deps: T7.09, T7.11
- [ ] T7.15 — API documentation and quickstart · S · deps: T7.07
- [ ] T7.16 — Policy-maker role and endpoints (D13) · M · deps: T7.04, T2.31
- [ ] T7.17 — Bond-market endpoints · S · deps: T7.04, T6.26

- [ ] **GATE P7** — human review against the gate in the phase file

## Phase 8 — Realism and game (coarse) (12 tasks) — `08-phase8-9-realism-game-scale.md`

- [ ] T8.01 — Financing depth · L · deps: T7.14
- [ ] T8.02 — Supply contracts · M · deps: T7.14
- [ ] T8.03 — Quality and brand · M · deps: T7.14
- [ ] T8.04 — Calibration pass · L · deps: T7.14, T0.09 · **HUMAN GATE**
- [ ] T8.05 — Historic replay comparison · M · deps: T8.04
- [ ] T8.06 — Explainability traces · L · deps: T7.04
- [ ] T8.07 — Game layer contract · L · deps: T7.15
- [ ] T8.08 — Regulator · M · deps: T7.14
- [ ] T8.09 — Regional real-estate asset · M · deps: T7.14
- [ ] T8.10 — Tier-3 counterfactual validation (research) · M · deps: T7.14
- [ ] T8.11 — Korea-flavoured calibration set (optional) · M · deps: T8.04
- [ ] T8.12 — Goods layer (only if ADR T3.16 approved it) · L · deps: T3.16, T7.14

- [ ] **GATE P8** — human review against the gate in the phase file

## Phase 9 — Scale-out (coarse) (6 tasks) — `08-phase8-9-realism-game-scale.md`

- [ ] T9.01 — Multi-country and FX · L · deps: T8.04
- [ ] T9.02 — Storage backend · L · deps: T7.10
- [ ] T9.03 — Parallel worlds at scale · M · deps: T7.12
- [ ] T9.04 — M&A depth · L · deps: T6.16
- [ ] T9.05 — Rust core for hot paths (only if profiling demands) · L · deps: T7.13
- [ ] T9.06 — gRPC transport (optional) · M · deps: T7.04

- [ ] **GATE P9** — human review against the gate in the phase file
