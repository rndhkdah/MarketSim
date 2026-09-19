# Phase 3 — Regions and the demand layer

**Goal.** (a) Replicate the economy over R = 3–5 regions with sticky, price-sensitive inter-regional trade so that
events can hit one area and prices converge within a band; (b) replace the scalar income elasticity by **wealth-tier
buy packages over a want layer** — non-homothetic demand where needs appear, plateau and vanish — keeping
price-sensitive substitution; (c) add NPC entry/exit so cells stay contestable. All design-only: every step must keep
the Phase-2 suite green.

From the Victoria 3 research — **borrow:** wealth-tier packages, the want layer, the two-regime price handoff (already
in Phase 2, with the bound on the step), leaky inventories. **Do not borrow:** the ±75 % price cap, price-blind
substitution, the absence of asset prices.

## 3.1 Gate

1. **R = 1 parity:** the regional code with one region reproduces Phase 2 (series equal to 1e-12).
2. **Regional steady state:** national totals equal the Phase-2 baseline to 1e-9; every cell clears; no-shock run stationary at π\* ∈ {0, 0.02}.
3. **Propagation and convergence:** a −10 % productivity shock to MATERIALS in one region raises MATERIALS prices everywhere, most at the source; for tradables `|ln p_r − ln p̄|` stays inside `max link cost + 0.05` after month 24; non-tradables (CONSTRUCT, REALESTATE) need not converge.
4. **Demand system:** baseline basket = `final_demand.HOUSEHOLD` exactly (1e-9); Spearman(implied η, config `eta`) ≥ 0.8 and mean |Δη| ≤ 0.25; non-homothetic regression (3.4) passes.
5. **Entry/exit:** a permanent +10 % demand shift to one cell → excess profit rate below the entry threshold within 8 years.
6. Phase-2 validation suite green at R = 3 with `demand.mode: tiers_wants`.
7. Golden national series recorded for the Phase-5 hybrid ≈ aggregate test.

## 3.2 Regions and trade

`config/regions.yaml` (starting values; `location_quotient` omitted entries = 1.0):

```yaml
regions:
  - {code: CAPITAL,    population_share: 0.45, wage_level: 1.10}   # services, finance, software
  - {code: INDUSTRIAL, population_share: 0.35, wage_level: 1.00}   # manufacturing, ports, logistics
  - {code: RESOURCE,   population_share: 0.20, wage_level: 0.92}   # energy, mining, agriculture
location_quotient:
  CAPITAL:    {SOFTWARE: 1.6, BANKS: 1.6, INSURANCE: 1.5, BIZSVC: 1.4, TELECOM: 1.3, REALESTATE: 1.3}
  INDUSTRIAL: {AUTOS: 1.7, CAPGOODS: 1.6, SEMIS: 1.5, TRANSPORT: 1.4, STAPLES: 1.2, MATERIALS: 1.1}
  RESOURCE:   {ENERGY: 2.4, AGRIFOOD: 2.0, MATERIALS: 1.6, UTILITIES: 1.3}
tradability: {ENERGY: 0.9, UTILITIES: 0.3, MATERIALS: 0.9, AGRIFOOD: 0.8, SEMIS: 1.0, CAPGOODS: 0.9, CONSTRUCT: 0.1,
              TRANSPORT: 0.5, AUTOS: 0.9, DISCRET: 0.2, STAPLES: 0.8, HEALTH: 0.15, SOFTWARE: 0.9, TELECOM: 0.5,
              BIZSVC: 0.5, BANKS: 0.6, INSURANCE: 0.7, REALESTATE: 0.0}
armington_sigma: {default: 2.0, ENERGY: 4.0, MATERIALS: 3.0, AGRIFOOD: 3.0, SEMIS: 3.0}
links:
  - {a: CAPITAL,    b: INDUSTRIAL, cost: 0.04, capacity_mult: 2.0}
  - {a: INDUSTRIAL, b: RESOURCE,   cost: 0.06, capacity_mult: 2.0}
  - {a: CAPITAL,    b: RESOURCE,   cost: 0.09, capacity_mult: 1.5}
trade: {tau_share_m: 6, home_bias: 1.5, gravity_theta: 8.0, availability_kappa: 1.0}
migration: {enabled: true, rate_per_pp_per_year: 0.001}
```

**Baseline trade shares.** `T0[i, s→d] = (1 − trad_i)·1[s = d] + trad_i·Gr[i, s→d]`,
`Gr ∝ target_capacity_share[s, i] · exp(−θ·cost_sd) · (home_bias if s = d)`, normalised over source regions `s`.
`target_capacity_share[s,i] ∝ population_share_s × LQ[s,i]`.

**Regional steady state.** Stack cells into a vector of length R·S. With `d` the stacked regional final demand,
`x0 = (I − Λ 𝒯 𝒜)⁻¹ Λ 𝒯 d`, where `𝒜` applies A within each buying region and `𝒯` routes each product's orders from
buying region to source regions by `T0`. Because `Σ_s T0[i, s→d] = 1` and technology is identical across regions,
national totals equal the Phase-2 solution exactly. Regional household demand depends on regional income, which
depends on regional production → solve the fixed point by iteration (start from population shares; converge to 1e-12).
Regional wage levels are offset by productivity (`ell_rs = ell_s / wage_level_r`), so unit costs — and baseline
prices — are 1 in every region while wage income per worker differs.

**Dynamics.** Delivered input price for buyers in `d`: `P_in[i,d] = Σ_s T[i,s→d]·p[s,i]`. Transport cost is a
**preference friction only** (no payment wedge — transport services are already bought through A), which keeps the
ledger exact. Targets: `T*[i,s→d] ∝ T0 · (p[s,i](1 + cost_sd)/P̄[i,d])^(−σ_i) · avail[s,i]^κ`, renormalised;
`T += (T* − T)/τ_share`. Link capacity = `capacity_mult ×` baseline flow; flows above it are scaled back pro-rata,
spill to other sources in proportion to their shares, and the remainder meets R6 rationing. Supplier cells ration
pro-rata across buying regions.

**Labour.** `LF_r`, `U_r`, `w_r` with a regional Phillips curve and national `π_e`. Migration:
`ΔLF_r = LF_r · rate · (Ū − U_r)·100 / 12` per month, renormalised to conserve national LF.
**Households** are per region (`HH:<r>`): income = regional wages + dividends (by wealth share) + transfers − taxes.
**Government** is national; purchases are allocated to regions by population share.

## 3.3 Tiers × wants

**Tiers.** Ten income deciles per region. Relative income from a lognormal with σ = 0.75 (Gini ≈ 0.40):
`ι_k = 10·[Φ(z_k − σ) − Φ(z_{k−1} − σ)]`, `z_k = Φ⁻¹(k/10)`. Regional nominal consumption comes from the **unchanged
Phase-2 macro consumption function**; tiers decide only its *composition*. Tier budget share `b_k ∝ ι_k^0.9` (the rich
save more). Tier real income `y_k = ι_k · y_r` (`y_r` = regional real disposable income index, 1 at baseline).

**Need shapes** (package value before budget scaling; all parameters in `buy_packages.yaml`):

| shape | formula | V3 analogue |
|---|---|---|
| survival | `v_max·(1 − exp(−y/y_s))` | lower strata, wealth 1–9 |
| plateau | `v_max·min(1, y/y_p)` | middle strata: grow linearly then plateau |
| vanish | `v_pk·(y/y_pk)·exp(1 − y/y_pk)` | needs that disappear as pops get richer |
| normal | `b·y` | — |
| luxury | `b·max(0, y − y_th)^γ`, γ ∈ [1.3, 2] | upper strata: threshold then acceleration |

Budget scaling: survival wants are served first (scaled only if they alone exceed the budget); the rest pro-rata; any
surplus is spread pro-rata over non-survival wants. Want-level own-price response:
`v_q ← v_q·(P_q/Pc)^(1 + ε_q)`, ε_q = spending-weighted `eps_own` of its sectors.

**Wants → sectors** (prior weights `M0`; the calibrator RAS-fits them):

| want | shape | sectors (prior weights) |
|---|---|---|
| FOOD_HOME | survival | STAPLES 0.75, AGRIFOOD 0.25 |
| BASIC_GOODS | vanish | STAPLES 0.7, AGRIFOOD 0.3 |
| SHELTER | survival → plateau | REALESTATE 0.9, CONSTRUCT 0.06, MATERIALS 0.04 |
| HEAT_POWER | survival | UTILITIES 0.7, ENERGY 0.3 |
| HEALTH | normal | HEALTH 0.85, INSURANCE 0.15 |
| MOBILITY | plateau | AUTOS 0.5, ENERGY 0.25, TRANSPORT 0.25 |
| COMMUNICATION | plateau | TELECOM 0.7, SOFTWARE 0.3 |
| HOUSEHOLD_GOODS | plateau | DISCRET 0.75, CAPGOODS 0.15, AUTOS 0.10 |
| EATING_OUT_LEISURE | luxury (low threshold) | DISCRET 0.85, TRANSPORT 0.15 |
| FINANCIAL_PROTECTION | normal | BANKS 0.45, INSURANCE 0.55 |
| PERSONAL_SERVICES | luxury | BIZSVC 0.7, SOFTWARE 0.3 |
| LUXURY | luxury (high threshold, γ 2) | DISCRET 0.5, AUTOS 0.2, REALESTATE 0.2, BIZSVC 0.1 |

**Within-want allocation — price-sensitive, availability-aware:**
`share_{i|q} ∝ M[q,i] · (p_i/P_q)^(−σ_q) · avail_i^κ`, clipped to `[min_share, max_share]` and renormalised;
`avail` = 3-month EMA of the fill rate; σ_q default 0.8 (FOOD 0.6, MOBILITY 1.0). Price keeps the cross-price channel
alive (what V3 deleted); availability prevents "starving beside a mountain of fish". Sector-level rate shifters
(`dem_rate_semi`) and typed-edge shifters from Phase 2 still apply afterwards.

**Calibrator** (`demand/calibrate.py` → generated `buy_packages.yaml`, `wants.yaml`, report):
(1) start from priors; (2) RAS-fit `M` so that `Σ_q V_q·M[q,i]` equals the national baseline basket **exactly**;
(3) compute implied sector income elasticities by a +1 % uniform real-income bump; (4) `scipy.optimize.least_squares`
on shape parameters (deterministic start, bounded) to match config `eta`; (5) repeat 2–4 to convergence.

## 3.4 Non-homothetic regression (tests/validation/test_demand_nonhomothetic.py)

As mean real income rises 50 % at fixed prices: FOOD_HOME budget share falls; EATING_OUT_LEISURE and LUXURY shares rise;
BASIC_GOODS falls in **absolute** real terms for the top 5 deciles (a need vanishing — impossible with a constant
elasticity); in a −10 % income downturn DISCRET falls more than STAPLES ("eating out vs eating in") *without* any
explicit edge. With one good's fill rate at 5 %, its want is still ≥ 95 % satisfied through substitutes while the good's
relative price has risen (both channels active).

## 3.5 NPC entry / exit

`excess_rs` = 12-month smoothed (profit rate / baseline profit rate) − 1. Entry adds
`κ_entry·max(excess − θ_e, 0)` to the capex start rate (same pipeline, same financing); exit scraps
`κ_exit·max(−excess − θ_x, 0)` of K per year (real-asset write-off; debt stays and feeds loan losses).
Defaults: θ_e 0.15, κ_entry 0.10, θ_x 0.25, κ_exit 0.05. This caps what an agent monopolist can extract (Phase 5).

---

## 3.6 Task cards

### T3.01 — `regions.yaml` and geometry
**Depends:** T2.23 · **Size:** S · **Files:** `config/regions.yaml`, `src/marketsim/regions/geometry.py`, `tests/unit/regions/test_geometry.py`
**Read first:** §3.2. **Build:** schema, capacity-share targets, link matrix (cost, capacity), region masks
(`mask(regions=..., sectors=...) -> bool (R,S)`). **Tests:** shares sum to 1 per sector; symmetric links; masks correct;
R = 1 file accepted.

### T3.02 — Regionalise the state (R = 1 parity)
**Depends:** T3.01 · **Size:** M · **Files:** `src/marketsim/real/*.py`, `tests/golden/test_r1_parity.py`
**Build:** remove any remaining single-region assumption; every array `(R,S)`; per-region wages, labour force,
households. **Tests:** gate 1 (golden series from Phase 2 equal to 1e-12).

### T3.03 — Baseline trade shares and regional steady state
**Depends:** T3.02 · **Size:** M · **Files:** `src/marketsim/regions/trade.py`, `src/marketsim/real/steady_state.py`, `tests/unit/regions/test_regional_steady_state.py`
**Read first:** §3.2. **Build:** `T0`, stacked Leontief solve, income fixed point, productivity offset for wage levels.
**Tests:** gate 2; non-tradables produced where consumed; RESOURCE is a net exporter of ENERGY, CAPITAL of SOFTWARE.

### T3.04 — Regional orders, rationing and link capacity
**Depends:** T3.03 · **Size:** M · **Files:** `src/marketsim/regions/trade.py`, `src/marketsim/real/orders.py`, `tests/unit/regions/test_regional_rationing.py`
**Build:** route orders by `T`, pro-rata supplier rationing across buying regions, link caps with spill-over.
**Tests:** conservation (Σ deliveries = sales per cell); a link cut to 10 % capacity forces spill to the other source and
then rationing; ledger SFC unaffected.

### T3.05 — Trade-share dynamics and regional prices
**Depends:** T3.04 · **Size:** M · **Files:** `src/marketsim/regions/trade.py`, `src/marketsim/real/prices.py`, `tests/validation/test_regional_convergence.py`
**Build:** Armington targets with stickiness and availability; delivered input prices in unit costs. **Tests:** gate 3;
shares stay in the simplex; no oscillation (share path monotone after a step change in relative price).

### T3.06 — Regional labour pools and migration
**Depends:** T3.02 · **Size:** S · **Files:** `src/marketsim/regions/labour.py`, `tests/unit/regions/test_migration.py`
**Tests:** national LF conserved to 1e-12; a regional slump loses labour force slowly (≈0.1 % of LF per year per pp gap).

### T3.07 — Regional households and national government
**Depends:** T3.02 · **Size:** S · **Files:** `src/marketsim/real/households.py`, `src/marketsim/real/government.py`, `tests/unit/real/test_regional_households.py`
**Tests:** regional incomes sum to national; SFC per region entity; government purchases split by population.

### T3.08 — Tiers
**Depends:** T2.12 · **Size:** S · **Files:** `src/marketsim/demand/tiers.py`, `tests/unit/demand/test_tiers.py`
**Read first:** §3.3. **Tests:** ι sums to 10 (mean 1); Gini of the discretised distribution 0.38–0.42; budget shares sum to 1.

### T3.09 — Want layer and within-want allocation
**Depends:** T3.08 · **Size:** M · **Files:** `src/marketsim/demand/wants.py`, `config/wants.yaml`, `tests/unit/demand/test_wants.py`
**Tests:** shares in simplex and inside min/max; cheaper good gains share; availability 5 % → share collapses but the want's
total is preserved; every sector with positive household demand appears in ≥ 1 want.

### T3.10 — Need shapes and budget scaling
**Depends:** T3.08 · **Size:** S · **Files:** `src/marketsim/demand/packages.py`, `tests/unit/demand/test_packages.py`
**Tests:** each shape's formula at three points; vanish peaks at `y_pk` and → 0; budget exhausted exactly; survival served first.

### T3.11 — Calibrator
**Depends:** T3.09, T3.10 · **Size:** M · **Files:** `src/marketsim/demand/calibrate.py`, `scripts/calibrate_demand.py`, `config/buy_packages.yaml` (generated), `claude/plan/reports/demand-calibration.md`
**Tests:** gate 4; deterministic output (same file twice); report lists implied vs config η per sector.

### T3.12 — Swap in `TiersWantsDemand`
**Depends:** T3.11, T3.07 · **Size:** M · **Files:** `src/marketsim/demand/system.py`, `tests/validation/test_demand_nonhomothetic.py`
**Build:** second `DemandSystem` implementation; `demand.mode: scalar_eta | tiers_wants`. **Tests:** §3.4; aggregate
response to ±5 % income within 20 % of the scalar-η system; gate 6.

### T3.13 — Want shifters (interface for events)
**Depends:** T3.12 · **Size:** S · **Files:** `src/marketsim/demand/wants.py`, `tests/unit/demand/test_want_shifters.py`
**Build:** `want_shift[r, q]` multiplicative states with AR decay, driven through `ShockBus`. **Tests:** −50 % on
EATING_OUT_LEISURE cuts DISCRET demand, budget re-spreads over other wants, saving unchanged.

### T3.14 — NPC entry / exit
**Depends:** T3.02 · **Size:** M · **Files:** `src/marketsim/real/entry_exit.py`, `tests/validation/test_entry_exit.py`
**Read first:** §3.5. **Tests:** gate 5; no entry/exit at baseline (bitwise); exit write-offs pass SFC.

### T3.15 — Regional validation and golden baseline
**Depends:** T3.05, T3.12, T3.14 · **Size:** M · **Files:** `tests/validation/test_regional_shocks.py`, `tests/golden/data/aggregate_baseline.npz`, `claude/plan/reports/phase3-validation.md`
**Build:** regional disaster / strike scenarios via masks; rerun §2.12 at R = 3; record golden national series (10 years,
fixed shock script) for T5.16. **Tests:** gates 3, 6, 7.

### T3.16 — ADR: goods layer beneath consumer-facing sectors — **HUMAN GATE**
**Depends:** T3.15 · **Size:** S · **Files:** `claude/plan/DECISIONS.md`
**Build:** Write the decision memo for the V3 structural fork (sectors as the financial unit + optional goods layer
under STAPLES / DISCRET / AGRIFOOD for game legibility): cost, where it plugs in (below the want layer), recommendation.
