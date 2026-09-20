# DECISIONS — architecture decision records

Template: `## ADR-nnn — title` · date · status (proposed | accepted | superseded) · context · decision · consequences.
Agents may *propose*; only the human accepts. Changing a config value in `sectors.yaml` / `edges.yaml`, a validation
tolerance, or anything in AGENTS.md "Non-negotiable design rules" requires an accepted ADR.

Pre-recorded (accepted 2026-09-19, rationale in `00-MASTER-PLAN.md` §13–§14):

- ADR-P1 General double-entry ledger in Phase 2, not Phase 5.
- ADR-P2 Capex coefficients are percentage points of K per year (`unit_scale: 0.01`); supply-line weight 0.85.
- ADR-P3 Bank capital gate on equity / RWA, normalised to 1 at a 0.125 baseline.
- ADR-P4 `dem_rate_semi` applies to the household component only; credit-edge elasticities are rationing exponents.
- ADR-P5 Nominal smoothing is drift-compensated by expected inflation; never deflated by realised CPI.
- ADR-P6 Price steps are bounded (0.15 log-points a month); price levels never are.
- ADR-P7 Tradable sector instrument = NPC sector equity; published sector index also includes listed agent firms.
- ADR-P8 Matching engine in-process behind a `Venue` protocol.

Accepted 2026-09-19 (v1.1):

- ADR-P9 Bond market: three fungible decaying-coupon government buckets + one pooled corporate bond; par pricing until Phase 6.
- ADR-P10 `GOVT` and `CENBANK` are policy authorities with levers; control = autopilot | scripted | agent.
- ADR-P11 One corporate borrowing rate (`credit.pricing: uniform`); `edges.yaml: credit.spread_scaling` unused in this mode;
  ratings drive limits, not price. Implemented T2.30: `CreditBlock.spread` is the single `s_t`; uniform loan rates
  are `r + s_t` with no sector/region/firm index. `risk_based` keeps `s0·max(nd,floor)/2.5` plus a common `(nd/2.5)·Δs`
  bump so a default still moves every name the same direction.

Accepted 2026-09-19 (v1.2):

- ADR-P12 Monetary policy framework (D15): 8-meeting committee, dual mandate (φ_u 1.0, φ_y 0), core weight 0.5, r\* on
  trend growth, 25bp grid + 10bp deadband, decisions on published vintages. Makeup strategies off by default and only
  ever leaky + clipped; risk-management, financial-conditions and credit terms shipped off.

## ADR-000 — target layout

- date: 2026-09-19
- status: proposed (T0.01)
- context: The plan assumed `claude/marketsim/{config,scripts,tests}` and a root-level package. This repository is a Victoria 3 harness (`harness/`, `mod/`, root `pyproject.toml`). There is no legacy Layer-1 tree in the zip or the checkout. A root `AGENTS.md` would hijack Vic3 agents.
- decision: Standalone tree at `marketsim/` with its own `AGENTS.md`, `claude/plan/`, `pyproject.toml`, `Makefile`, `src/marketsim/`, `config/`, `tests/`. Plan pack is copied into `marketsim/claude/plan/`. Zero edits to Vic3 files. Layer 1 is reconstructed from published invariants (appendix).
- consequences: `uv run pytest` at the repo root stays Vic3-only. Engine checks are `cd marketsim && make check`. Nested `.github/workflows/ci.yml` will not auto-run on GitHub unless a root dispatcher is added later.

### Appendix — published Layer-1 goldens (master plan / Phase 0 gate)

Reproduced target (numpy 2.4 / scipy 1.17 sandbox):

| quantity | value |
|---|---|
| A | 18×18 |
| nnz | 219 |
| density | 0.68 |
| ρ(A) | 0.536 |
| output multipliers | 1.58–2.77 |
| regime demo seed 7 | corr(equity, bonds) −0.74 demand / +0.90 supply |
| sign flips | 13/18 |
| rotation early | AUTOS, CONSTRUCT, CAPGOODS, TRANSPORT, SOFTWARE |
| rotation recession | STAPLES, HEALTH, REALESTATE, UTILITIES, TELECOM |
| FD 100 cr | GO 209.3, wages 51.82, GOS 48.18 |

## ADR-001 — bank capital gate on equity / RWA (B3)

- date: 2026-09-19
- status: proposed (T0.14; restates accepted ADR-P3)
- context: The legacy logistic used midpoint 0.085, steepness 90, and a capital ratio of `1/asset_leverage` = 1/11 ≈ 0.0909, so the *raw* gate was ≈ 0.63 at baseline.
- decision: `credit.bank_capital_gate.ratio = equity_over_rwa`, `baseline_capital_ratio = 0.125`, `normalise_at_baseline = true`. `logistic_gate(c) = min(1, σ(k(c−mid)) / σ(k(c0−mid)))`. Then `gate(0.125) = 1` and `gate(0.085) = 0.5/0.9734`.
- consequences: Layer-1 goldens unchanged. Credit edges stay off until Phase 2.

## ADR-002 — capex unit contract (B4)

- date: 2026-09-19
- status: proposed (T0.14; restates accepted ADR-P2)
- context: `q_tobin: 0.30` is ≈10× too strong if coefficients are read as raw rates. Prototype verification used percentage points of K per year plus a supply-line weight.
- decision: `capex.unit = pct_pts_of_K_per_year`, `unit_scale = 0.01`, `q_scale = 0.1`, `supply_line_weight = 0.85`. Coefficients φ=1.2, ψ=0.8, χ=0.45, q_tobin=0.30 stay as written.
- consequences: Used by T2.10. No Layer-1 number changes.

## ADR-003 — production_mode (B5)

- date: 2026-09-19
- status: proposed (T0.14)
- context: `inv_lag_q` conflated finished-goods cover with order-book length.
- decision: `dynamics.yaml` starts with `production.mode` (stock / order / flow) per `02-phase2-dynamic-layer.md` §2.11. `inv_lag_q` remains a sector parameter; cover vs order-book scale are separate (`cover_scale`, `order_book_scale`) in Phase 2.
- consequences: Full `dynamics.yaml` schema is T2.01. Layer-1 goldens untouched.

## ADR-004 — reconstructed etas and regime-demo volumes

- date: 2026-09-19
- status: proposed (T0.06 / T0.07 reconstruction)
- context: No legacy `sectors.yaml` / `betas.md`. Rotation (early AUTOS/CONSTRUCT/CAPGOODS/TRANSPORT/SOFTWARE, recession STAPLES/HEALTH/REALESTATE/UTILITIES/TELECOM) and the seed-7 regime windows are published goldens. Growth betas are `derive-don't-assert` from Leontief cyclicality + eta + capex routing.
- decision: Keep the mechanism; calibrate `eta` (AGRIFOOD 0.95, SEMIS 1.15, UTILITIES 0.12, TRANSPORT 1.95, SOFTWARE 2.35, TELECOM 0.18, INSURANCE 1.05, REALESTATE 0.50) so the top/bottom five `beta_growth` match those sets. Regime Monte-Carlo knobs live on `BetasParams` (oil_own 4.80, demand Taylor 0.085, supply oil vol 0.032) so seed 7 hits −0.74 / +0.90 ±0.02 and 13/18 flips. Not a bit-identical `betas.md`.
- consequences: Changing an eta or a `BetasParams` demo volume to chase a new golden needs a new ADR. Rotation tests assert *sets*, not order.

## ADR-005 — plan upgraded v1.0 → v1.2

- date: 2026-09-19
- status: proposed
- context: The Phase 0 tree was built against the v1.0 pack. A later zip (v1.2 — D15 monetary framework) adds D12–D15 and new cards without changing Phase 0 or Phase 3, or the Layer-1 goldens.
- decision: Adopt the v1.2 pack as the spec. Replace `marketsim/claude/plan/` (and the store `docs/marketsim-plan/` copy) with that pack. Keep reconstruction QUESTIONS (T0.01 / T0.05 / T0.02 / T0.15) and ADRs 000–004. Phase 0 code, World skeleton, and Layer-1 goldens stay unchanged. New cards stay unticked.
- consequences: After GATE P0 the next named card is still T2.01. D14/D15 first affect T2.01 → T2.14 / T2.27+. `policy.yaml` is T2.27/T2.32; `bonds.yaml` is T6.24. AGENTS.md rules 15–17 apply from this ADR.

## ADR-006 — no goods layer under consumer-facing sectors (v1)

- date: 2026-09-20
- status: proposed (T3.16 HUMAN GATE; higher-reasoning recommendation)
- context: Victoria 3 splits consumer goods (grain, clothes, furniture…) under pop needs, while MarketSim’s financial
  unit is the 18-sector IO table. Open decision 9 in the master plan asked whether to add a goods layer under
  STAPLES / DISCRET / AGRIFOOD for game SKUs, with sectors remaining the listed equity. Phase 3 now has a two-shape
  want layer (FOOD_HOME survival + BASIC_GOODS vanish on STAPLES; EATING_OUT / LUXURY on DISCRET) RAS-fitted to the
  HOUSEHOLD basket (Spearman 0.816, basket 1e-9). T8.12 is the only implementation card and is gated on this ADR.
- decision: **Do not add a goods layer in v1.** Keep the 18 sectors as the real *and* financial unit. Within-want
  allocation (`share ∝ M (p/P)^{-σ} avail^κ`) is the substitution surface. Game legibility is names, icons and
  flavour text on existing sectors and wants — not a second production graph. T8.12 stays skipped unless a later
  accepted ADR reverses this.
- where it would plug (if reversed): below the want layer only. Wants allocate to goods; goods map many-to-one onto
  STAPLES / DISCRET / AGRIFOOD for IO, prices, inventories and listed equity. No goods-level listed instrument.
  RAS and SFC stay on the 18-sector table; goods quantities would be a pure split of those three columns.
- cost if reversed: new `config/goods.yaml`, `(R, G)` inventories and prices, a mapping matrix, a second RAS or
  nested CES, calibration against no public goods-level IO, and a risk of breaking gate 4 (HOUSEHOLD basket identity)
  plus every `(R, S)` kernel. T8.12 is sized L. Mizuta: the Phase-3 question (non-homothetic composition + regional
  prices) is already answered without it.
- consequences: T8.12 is not in the v1 path. Phase 4 events target wants and sectors, not SKUs. GATE P3 does **not**
  include a goods layer. Reversal needs a new accepted ADR before any goods state is added.

## ADR-007 — T4.13 event-magnitude sources; engine mapping deferred

- date: 2026-09-20
- status: proposed (T4.13 HUMAN GATE; higher-reasoning recommendation)
- context: Every shipped event carried `verify: true` seeds from §4.4. T4.13 asks for a primary source (FRED, BLS,
  EIA, BEA, IMF, UNCTAD, METI) and to clear the flag. Putting the raw historic *price* move on `z_cost` (oil
  `ln(4) ≈ 1.25`) is not SFC-finite at 24 months because the Phase-2 IRF of `z = 0.30` already yields ~+72 % CPI.
- decision: **Accept the published figures as the calibration targets and clear `verify`.** Keep the §4.4 seed
  *distributions* (sign and order of magnitude of the composition). Do **not** retune `edges.yaml` / `sectors.yaml`
  or shrink seeds to chase 24-month SFC — that mapping is T8.04. Report: `claude/plan/reports/event-calibration.md`.
- consequences: T4.11 direction tests stay on the SFC-finite prefix for oil/energy. T4.10 10× storm omits those
  three ids until T8.04. GATE P4 can be reviewed on schema / cascade-boundedness / directions, not level match.

## ADR-008 — GATE P4 review against master plan §9 / phase §4.1

- date: 2026-09-20
- status: proposed (GATE P4; higher-reasoning recommendation)
- context: Master plan §9 Phase 4 is done when “cascades bounded (sub-critical branching); templates reproduce
  historic response directions.” Phase file §4.1 adds schema-only primitives, seed→history isolation, and D13
  policy-follow-up control.
- decision: **Propose GATE P4 as met for the mechanism, with the T4.11/T4.13 caveats.** Do not mark the PROGRESS
  checkbox accepted until a human confirms. Evidence:
  1. Schema: seven primitives only; extras whitelist; catalog load + DAG Σp ≤ 0.9 (T4.01, T4.08).
  2. Cascades: `expected_cascade_size` ±20 % on chip_shortage; 20y 10× storm SFC-finite, |gap|<25 %, depth ≤ 3,
     concurrency ≤ 4, after omitting oil/energy/covid seeds (T4.10 + QUESTIONS T4.11).
  3. Directions: T4.11 §4.5 on/off-follow-up; oil/energy on the 12-month SFC-finite prefix; covid DISCRET < −15 %
     and HEALTH > DISCRET in the want-shift window under `tiers_wants`.
  4. RNG isolation: same seed → same history; adding RandomWalkModule does not change fires (T4.05, T4.09, T4.14).
  5. D13: policy follow-ups become “policy pressure” news when the authority is not autopilot (T4.14).
- not claimed: level match to FRED/BLS paths (T8.04); 10× storm that includes uncalibrated oil/energy/covid seeds.
- consequences: Phase 5 (T5.01) is unblocked on mechanism. Human may reject the storm omission or the 12-month
  oil/energy prefix.

## ADR-009 — T2.26 CES on intermediates stays off

- date: 2026-09-20
- status: proposed (T2.26 HUMAN GATE; higher-reasoning recommendation)
- context: T2.26 is optional. Default production is Leontief (`A` fixed). CES would be
  `a_ij,t = a_ij·(p_i/p̄_j)^(−σ_ij)` renormalised, σ from ENERGY↔UTILITIES and MATERIALS→CONSTRUCT
  substitution pairs. σ = 0 must reproduce Leontief bitwise and keep §2.12 green.
- decision: **Do not implement CES in v1.** The Leontief `A` plus typed substitution edges (T2.22)
  already move AUTOS/UTILITIES under an ENERGY shock. Adding a second, lagged CES layer on
  intermediates is a new mechanism without a failed test that needs it (Mizuta). `ces.py` is not
  added. The card remains skippable; σ = 0 is the current engine.
- consequences: T2.26 is recorded as skipped. A later accepted ADR can add `src/marketsim/real/ces.py`
  behind a default-off config flag. No `edges.yaml` / `sectors.yaml` change.

## ADR-010 — GATE P2 review (proposed)

- date: 2026-09-20
- status: proposed (GATE P2; higher-reasoning recommendation)
- context: Master plan §9 Phase 2 requires stationarity at π\* ∈ {0, 2 %}, the seven IRF
  signs, housing-before-capex timing, 100-year stability, monthly SFC, autopilot
  bitwise-neutral policy, one corporate rate, and §2.14.7 monetary validation.
- decision: **Propose GATE P2 as met for the shipped mechanism.** Do not tick the
  PROGRESS checkbox as human-accepted. Known, already-answered xfails stay:
  credit-on IRF windows (T2.23 A), stochastic ranking (T2.24 A), dual-mandate
  supply price window (T2.32). CES is off (ADR-009). T2.35 report records the
  monetary numbers (trough m14 −0.41 %, 4.65 moves/y, 30 bp).
- consequences: Phase 3–4 already landed on this branch. Human may still reject
  the credit-on xfails or ask for a `phi_y_mult` ADR (pre-authorised ADR-P14 if U
  falls under a supply shock).

## ADR-011 — GATE P3 review against master plan §9 / phase §3.1

- date: 2026-09-20
- status: proposed (GATE P3; higher-reasoning recommendation)
- context: Master plan §9 Phase 3 is done when “R=1 parity; regional shocks propagate,
  prices converge within band; basket reproduced exactly, η rank-corr ≥ 0.8; aggregate
  baseline recorded.” Phase file §3.1 adds regional SS, entry/exit, and §2.12 at R=3
  with `tiers_wants`.
- decision: **Propose GATE P3 as met for the shipped mechanism, with the T3.15 caveats.**
  Do not tick the PROGRESS checkbox as human-accepted. Evidence
  (`claude/plan/reports/phase3-validation.md`):
  1. R=1 parity vs Phase-2 golden to 1e-12 (T3.02).
  2. Regional SS national totals 1e-9; RESOURCE ENERGY / CAPITAL SOFTWARE exporters (T3.03).
  3. −10 % MATERIALS in RESOURCE and an INDUSTRIAL AUTOS strike: tradable `|ln p_r − ln p̄|`
     inside max link cost + 0.05 after month 24 (T3.05 / gate 3).
  4. HOUSEHOLD basket `V @ M` vs θ to 1e-9; Spearman(implied η, config η) = 0.816;
     Engel signs (T3.11 / T3.12). mean |Δη| 0.33 is the 2026-09-20 rank-only waiver.
  5. +10 % AUTOS demand: excess profit below θ_e within 8 years (T3.14).
  6. Golden national series recorded for T5.16 (gate 7).
- not claimed: live `step_month` at R=3 (QUESTIONS T3.15 — orchestrator stays R=1);
  §2.12 monetary level window under `tiers_wants` (xfail); default `demand.mode` remains
  `scalar_eta` so Phase 2 stays bitwise. Goods layer is out (ADR-006).
- consequences: Phase 5 (T5.01) is unblocked. Human may reject the R=3 stepper deferral.

## ADR-012 — GATE P5 review against master plan §9 / phase §5.1

- date: 2026-09-20
- status: proposed (GATE P5; higher-reasoning recommendation)
- context: Master plan §9 Phase 5 is done when “hybrid ≈ aggregate; adversarial + monopoly
  tests; one agent on 2–3 levers in-process.” Phase file §5.1 adds books identity, cascade,
  and a 10,000-tick smoke run.
- decision: **Propose GATE P5 as met for the shipped mechanism, with the T5.16 caveat.**
  Do not tick the PROGRESS checkbox as human-accepted. Evidence:
  1. Hybrid identity: N∈{1,5} autopilot firms reproduce the NPC plan to 1e-9 (T5.16).
     Live `step_month` still NPC-only (QUESTIONS T5.16).
  2. Adversarial: floor pricing, cornering, max leverage, strategic default stay bounded
     (T5.17). Monopoly +30 % loses share to an NPC residual. Large bankruptcy is one step,
     SFC-green, emits `large_bankruptcy`.
  3. One agent: `World.submit(agent, FirmDecision)` for price / production / capex;
     same seed → same hash (T5.18).
  4. Books: 24-month A=L+E and tag-tied P&L (T5.02). Uniform corporate rate (T5.10 / D14).
- not claimed: 10-year World GDP within 1 % after splitting every cell (needs a hybrid
  orchestrator card).
- consequences: Phase 6 (T6.01) is unblocked on mechanism.

## ADR-013 — GATE P6 review against master plan §9 / phase §6.1

- date: 2026-09-20
- status: proposed (GATE P6; higher-reasoning recommendation)
- context: Master plan §9 Phase 6 requires Tier-2 impact, stock–bond regime flip, takeover +
  liquidation cascade, and §6.11 bond gates. T6.19 World feedback hook is still in QUESTIONS.
- decision: **Propose GATE P6 as met for the shipped market mechanism, with T6.19 deferred.**
  Do not tick the PROGRESS checkbox as human-accepted. Evidence:
  1. Betas-emerge Spearman 0.988/1/1/1; BANKS only +rate (T6.05 / gate 1).
  2. Regime flip and Tier-1/2 impact on the 25-name book (T6.21 / T6.07 / T6.31).
  3. Takeover at next strictly-later month-end; liquidation waterfall; SFC (T6.16 / T6.13).
  4. Bond gates 9–16 collected (T6.31); live `values()` stub Δρ = 0.35 so credit IRFs stay put.
- not claimed: T6.19 maps exist but are not on the monthly `World` stepper; credit-crunch 12y
  |gap| remains a pre-T6 xfail (do not loosen).
- consequences: Phase 7 API work already landed on this branch.

## ADR-014 — GATE P7 review against master plan §9 / phase §7.1

- date: 2026-09-20
- status: proposed (GATE P7; higher-reasoning recommendation)
- context: Master plan §9 Phase 7 is done when “multi-agent training run end-to-end;
  ≥ 1,000 ticks/s small world in-process; deterministic replay.”
- decision: **Propose GATE P7 as met for the shipped API/SDK.** Do not tick the PROGRESS
  checkbox as human-accepted. Evidence:
  1. Gate 1: 4 random + 2 scripted PettingZoo agents, 50,000 ticks, invariant counters zero (T7.14).
  2. Gate 2: lockstep `(agent, seq)` apply; replay reproduces per-tick hashes (T7.06 / T7.10).
  3. Gate 3: 5,488 ticks/s small world; 4,786 ticks/s with 8 scripted agents
     (`claude/plan/reports/phase7-throughput.md`). REST/WS overhead documented. No Rust (T9.05).
  4. Schemas `v1.1`; OpenAPI; SDK + Gymnasium/PettingZoo; examples in CI (T7.01 / T7.07–T7.15).
  5. Professional-mode information rules; policymaker D13 + bond desk (T7.16 / T7.17).
- not claimed: T6.19 still off the World hook; R=3 monthly stepper still QUESTIONS T3.15.
- consequences: Phase 8 cards that depend on T7.14 are unblocked. Human may still reject
  the empty-module World used for the 50k smoke (no RealEconomy).

## ADR-015 — T8.04 calibration pass (propose only; do not apply)

- date: 2026-09-20
- status: proposed (T8.04 HUMAN GATE; higher-reasoning recommendation)
- context: Seed `io_table.json` still carries hand-set `final_demand` and
  `fd_weights` 0.62 / 0.19 / 0.13 / 0.06. T0.09 aggregated the Use table onto
  18 sectors but left PCE / I / G / X seeded (`reports/bea-howto.md`). Historic
  event *levels* overflow SFC when raw price moves land on `z_cost` (QUESTIONS
  T4.11; `reports/event-calibration.md`). T2.24 cost-push IRF `z = 0.30` already
  yields ~+72 % CPI @12m. Open decision: map published FRED/BLS/EIA/BEA figures
  onto primitives through A and the typed edges, without retuning Layer-1
  elasticities by fiat.
- decision: **Accept the concordance and z-mapping *method*; do not write the
  proposed numbers into `edges.yaml`, `sectors.yaml`, or the seed IO table until
  the human ticks T8.04.** Specifically:
  1. **Final-demand concordance.** Map BEA NIPA PCE, private fixed investment,
     government consumption+investment, and exports of goods and services onto
     the 18 `CODES` so `final_demand` stops being a seed. `fd_weights` become
     the observed C/I/G/X shares of that year. Keep `world.io_source: seed`
     until the human switches it. Coverage floor stays 0.85.
  2. **Event magnitudes.** Historic *real* moves are calibration *targets*, not
     ShockBus values. Map 1973 crude ×4 onto `z_cost` in the IRF-safe band
     (~0.30–0.40, not `ln 4`). Keep YAML seed *distributions* (sign / order of
     magnitude) until the accepted map is written. T4.10 10× storm may then
     re-include oil/energy/covid.
  3. **Business-cycle moments.** Targets stay the card’s: `sd(I)/sd(GDP)` 3–4,
     `sd(C)/sd(GDP) < 1`, GDP persistence. Propose `dynamics` / household /
     capex gains only in `reports/calibration.md` before/after tables. Do not
     edit those yaml files in this card.
  4. **Cost-push / I.** T2.24 reopen stands: if the proposed set still ranks
     cost-push I response wrong, record it; do not invent a new elasticity.
- not claimed: a live BEA download in CI (script stays local-xlsx); moment match
  on the *current* seed (the proposed set is not loaded); human acceptance.
- consequences: T8.04 stays **unticked**. T8.05 / T8.11 may use the
  *method* (directions, optional KR overlay) against the seed
  economy. Applying the table is a human commit. T9.01 is deferred (ADR-017).

## ADR-016 — skip T9.05 Rust core

- date: 2026-09-20
- status: proposed
- context: T9.05 is “only if profiling demands.” T7.13 gate 3 measured 5,488
  ticks/s small world and 4,786 ticks/s with 8 agents, both well above the
  1,000 / 300 floors (`reports/phase7-throughput.md`).
- decision: **Do not start a Rust CLOB/monthly-step port.** Revisit only if a
  later profile shows a hot path below those floors on the intended hardware.
- consequences: T9.05 is ticked skipped. Python interfaces stay the ABI.

## ADR-017 — defer T9.01 multi-country/FX and T9.03 cluster eval

- date: 2026-09-20
- status: accepted (human: remove from v1)
- context: T9.01 (second economy + FX instrument) and T9.03 (process-pool
  `batch_eval`) were started on this branch. The human asked to drop
  multi-country / FX for now and to remove both cards from the v1 path.
- decision: **Do not ship T9.01 or T9.03.** Delete `countries.py`, `fx.py`,
  `sdk/cluster.py` and their tests. T7.12 `VectorEnv` stays (in-process
  parallel worlds). A later card may reintroduce FX or a cluster runner.
- consequences: PROGRESS T9.01 / T9.03 stay unchecked and marked deferred.
  One-country, one-currency remains the v1 scope.


