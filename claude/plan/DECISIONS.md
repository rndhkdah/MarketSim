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

