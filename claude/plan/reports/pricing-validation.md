# T6.05 — Pricing vs Layer-1 betas

Measured 2026-09-20. Impulses run through `AssetPriceProvider` (`structural`,
`apply_financials`, `banks.mode: passthrough`) so BANKS NIM is the earnings
channel the ledger does not yet produce on this path. Earnings shocks use the
same IO / operating-leverage / oil-cost / credit-elasticity ingredients as
`derive_betas`; the test is that `ln V = ln E^e + ln PE0 − D·Δρ + financials`
transmits those channels (gate 1). Full World IRF valuations wait on T6.19.

## Gate 1

| factor | impulse | Spearman vs `derive_betas` | threshold |
|---|---|---|---|
| rate | +100 bp + refi/revenue earnings + financials | **0.988** | ≥ 0.7 |
| growth | household/investment FD + η + capex routing | **1.000** | ≥ 0.7 |
| oil | ENERGY cost share via `G` + own-price | **1.000** | ≥ 0.7 |
| credit | `edges.credit` elasticities × `nd_ebitda` | **1.000** | ≥ 0.6 |

Ranking (rate hike): BANKS is the only positive sector; REALESTATE, SOFTWARE
and UTILITIES sit in the four most rate-negative names (TELECOM is the other
long-duration name).

Live `values(r, π_e, z)` without `ee` / `apply_financials` stays on the Phase-2
stub `Δρ = 0.35·(r − π_e − r_n) + z_risk` (T6.04 / credit IRFs).
