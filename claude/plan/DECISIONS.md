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
