# Demand calibration (T3.11)

RAS-fitted `M` so `Σ_q V_q M[q,i]` matches `final_demand.HOUSEHOLD` exactly, then
`least_squares` on shape parameters to match config `eta`. Rank-only Spearman
target is ≥ 0.7 (2026-09-20); master-plan gate 4 still cites 0.8.

- Spearman(implied η, config η) = **0.8164**
- mean |Δη| = 0.3287
- max |basket − θ| = 9.992e-16

| sector | θ | config η | implied η | Δη |
|---|---:|---:|---:|---:|
| ENERGY | 0.0249 | 0.85 | 0.698 | -0.152 |
| MATERIALS | 0.0100 | 1.15 | 0.396 | -0.754 |
| AGRIFOOD | 0.0547 | 0.95 | 0.667 | -0.283 |
| SEMIS | 0.0050 | 1.15 | 1.535 | +0.385 |
| AUTOS | 0.0547 | 2.40 | 1.657 | -0.743 |
| STAPLES | 0.1443 | 0.45 | 0.693 | +0.243 |
| DISCRET | 0.1194 | 1.50 | 1.505 | +0.005 |
| HEALTH | 0.1443 | 0.65 | 0.987 | +0.337 |
| CAPGOODS | 0.0149 | 1.90 | 1.382 | -0.518 |
| CONSTRUCT | 0.0100 | 2.10 | 1.485 | -0.615 |
| UTILITIES | 0.0348 | 0.12 | 0.414 | +0.294 |
| TRANSPORT | 0.0348 | 1.95 | 1.677 | -0.273 |
| SOFTWARE | 0.0398 | 2.35 | 1.663 | -0.687 |
| TELECOM | 0.0348 | 0.18 | 0.502 | +0.322 |
| BIZSVC | 0.0398 | 1.15 | 1.124 | -0.026 |
| BANKS | 0.0448 | 1.05 | 1.025 | -0.025 |
| INSURANCE | 0.0448 | 1.05 | 0.962 | -0.088 |
| REALESTATE | 0.1443 | 0.50 | 0.666 | +0.166 |
