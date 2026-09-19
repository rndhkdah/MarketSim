# Policy-authority validation (T2.31)

Measured 2026-09-19. Autopilot, π\* = 0, `R = 1`. Levers applied with `lag_m = 0`.
Defaults were not retuned.

## Autopilot parity

Two worlds, seed 3, 24 months: bitwise-identical `state_hash`. Idle `PolicyDesk` does
not change the real path.

## Fiscal multiplier (bond-financed G +1 % of GDP, 8 quarters)

| quantity | value |
|---|---|
| extra G | 0.01 × GDP0 per month for 24 months |
| cumulative output multiplier `ΣΔY / ΣΔG` | **0.66** (gate [0.5, 2.0]) |
| CPI level vs baseline @ 24m | 1.022 vs 1.000 (up) |

## Other §2.13 lever signs

| experiment | result |
|---|---|
| income-tax cut −3 pp, debt rule off, 18m | real GDP **+0.89 %** vs baseline |
| VAT +2 pp | consumer CPI = 1.02 × producer index at m1 (exact 2.00 pp wedge); π12 @ 24m = −1.1 % (no permanent inflation) |
| tariff +10 % | import *duty* 10 % of cif; CPI 1.0014 vs 1.000 @ 12m (up) |
| bank recap after 40 % equity write-off | gate < 0.6 then rises after `rescue_banksys` |
| capital requirement +2 pp (0.125 → 0.145) | gate = Λ = **0.882** |

VAT pass-through onto the *consumer* index is 1. Producer prices dip slightly on the
weaker real basket (m1 producer CPI ≈ 0.998).

## Adversarial extremes (10 years, SFC on)

G +10 % of GDP, `τ_y = τ_c = 0`, policy rate pinned at the ELB after the first
quarter-end meeting. All months finite; SFC holds; GOVT DEP never negative.

| month | real GDP | CPI | U | B |
|---|---|---|---|---|
| 0 | 94.0 | 1.00 | 5.0 % | 677 |
| 12 | 106.6 | 1.29 | floor | 1 002 |
| 24 | 40.9 | 1.09 | 1.6 % | 1 344 |
| 120 | 13.9 | 0.28 | 52 % | 2 719 |

Year-1 inflation and debt move in the expected (loose-fiscal) direction. Later the
run boom-busts: production/expenditure GDP identity opens (rationing vs still-high
nominal G) and the price level falls. Reported, not retuned.

## Notes

Credit-on IRF xfails (QUESTIONS T2.23 A, 2026-09-19) are unchanged. CES (T2.26) stays
off. T2.33 / T2.35 not started.
