# T6.31 — Bond-market validation

Measured 2026-09-20. Collector: `tests/validation/test_bond_market.py`.
Coefficients `κ_debt = 0.03` and `κ_qe = 0.05` are §6.11 / T8.04 calibration
flags, recovered from `term_premium_bucket` at GB_BOND duration 7.07 y.

## Gates 9–16

| gate | claim | result | source |
|---|---|---|---|
| 9 | `bonds.pricing: par` is Phase-2 bitwise; market mode keeps SS `P = 1` | pass | `test_par_opening_bitwise_phase2` |
| 10 | Policy +100 bp (4q): GB_BOND yield +35 bp (±10), bill quiet | pass | `test_gate10_plus_100bp_gb_bond_35bp_and_bill_quiet` |
| 11 | Regime flip uses actual GB_BOND total returns | pass | `test_gate_11_regime_flip_standin` |
| 12 | Calm auction bid-to-cover > 1 and \|tail\| < 5 bp | pass | `test_gate12_calm_cover_and_tail` |
| 13 | QE 10 % of GDP / 12 m: GB_BOND −30–80 bp; reserves +1:1; QT unwinds | pass | `test_gate13_qe_then_qt_reserves_and_yield` |
| 14 | Bond-financed Δb raises `tp` and lowers capex vs QE; QE injects reserves | pass | `test_gate14_bond_financed_raises_tp_and_lowers_capex` |
| 15 | +100 bp: INSURANCE MTM = holdings × D; BANKS less | pass | `test_gate15_insurance_mtm_from_ledger_exceeds_banks` |
| 16 | A large default raises every firm's borrowing rate by the same amount | pass | `test_gate16_default_raises_all_firm_rates_equally` |

## Measured `κ` effects (GB_BOND, D = 7.07 y)

| coefficient | value | effect |
|---|---|---|
| `κ_debt` | 0.03 | +1 pp of debt/GDP → **+3.03 bp** on `tp_b` |
| `κ_qe` | 0.05 | QE of 10 % of GDP → **−50.5 bp** on `tp_b` |

Secondary marks add the §6.4 kernel on top of this stock term (gate 13 band).

## Gate 3 repeated on `GB_BOND`

Same size grid (0.1–30 % of ADV) and horizons (1–20 days) as T6.21, stepped
through the live `GB_BOND` impact kernel (`σ = 0.01`).

| metric | measured | band |
|---|---|---|
| `δ` (log-log peak vs Q/ADV, 5-day) | **0.50** | [0.4, 0.7] |
| horizon slope at 5 % ADV | **0.20** | [0, 0.25] |
| revert 20d after 5 / 10 / 20-day 10 % ADV | **0.67 / 0.60 / 0.52** | [0.40, 0.75] |
| round-trip P&L | ≤ 0 | no free lunch |
