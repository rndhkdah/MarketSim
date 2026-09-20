# T2.35 — Monetary-policy framework report

Measured 2026-09-20. Default D15 autopilot unless noted. `R = 1`. Stochastic
cells use the T2.24 `STOCH` volumes, seed 11, π\* = 2 %.

## §2.14.7 checklist

| item | result |
|---|---|
| Autopilot parity (`phi_u: 0`, 4 meetings, `rate_step: 0`, lags 0, oracle) | bitwise vs the old quarterly rule (T2.32 / T2.33 PARITY) |
| Steady state at π\* ∈ {0, 2 %} | exact (`max ‖x/x0−1‖ < 1e-9`, π12 = π\*) |
| Monetary IRF timing (passthrough) | trough **month 14**, depth **−0.41 %**, rebound **0.24** (gate month 9–24, depth 0.15–0.60 %, rebound < 0.6) |
| Demand +2 % | rate rises (max 2.5 %, back to 1.0 % by m36) |
| Cost-push ENERGY +10 % | dual-mandate max r **14.5 %**; pure-headline (`core_weight: 0`) **14.75 %** |
| Labour supply −2 % (LF) | U falls to **3.5 %**; rate rises to **2.25 %** (sign of `u\* − u`) |
| 100-year grid / deadband | 3–6 moves/year, 20–40 bp (existing `test_grid_deadband_move_frequency`) |
| Makeup `decay: 1.0` / `clip: 0` | rejected at config load (prototype −93 % / −73 % messages) |
| Sahm | triggers in the scripted recession, not the soft landing (T2.34) |
| ELB | `shadow_rate` ≤ announced r; demand −8 % trough then recovers (T2.35) |

Credit-on monetary timing remains xfail (QUESTIONS T2.23 A). Dual-mandate supply
price-window xfail remains (QUESTIONS T2.32).

## Move frequency (20-year STOCH sample)

| rule | moves / year | mean |Δr| (bp) | sd(π12) | mean r |
|---|---|---|---|---|
| default dual (φ_u 1, core 0.5, vintages on) | 4.65 | 30.1 | 1.29 pp | 3.13 % |
| output-gap (φ_u 0, φ_y on) | 4.30 | 27.0 | 1.09 pp | 2.98 % |
| headline (`core_weight: 0`) | 4.80 | 31.2 | 1.32 pp | 3.13 % |
| core (`core_weight: 1`) | 4.80 | 29.2 | 1.25 pp | 3.13 % |
| lag-0 oracle | 4.55 | 29.7 | 1.18 pp | 3.23 % |
| CPI/U lag 3 months | 4.65 | 32.5 | 1.65 pp | 2.89 % |

`sd(π)` rises with the information lag (lag 0 < default 1m < lag 3), matching T2.33.

## Sacrifice ratio (π\* 2 % → 0, 48 months after 12m SS)

Cumulative monthly output-gap sum = −0.50; Δπ12 = −0.81 pp. Year-equivalent
sacrifice `−(Σ gap)/12 / Δπ` ≈ **5.1**. Raw month-sum / Δπ ≈ 62. The 2 % → 0
step is a large, permanent target change; reported, not retuned.

## ELB episode share

At π\* = 0 the ELB is 0 and the SS rate is 1 %, so the no-shock path never
touches the floor. A demand −8 % impulse records `shadow_rate` and a lower
trough, then recovers. A 100-year ELB *share* under STOCH volumes is left to
the marked-`slow` 100-year strategy suite (T2.34); this report does not add a
second 100-year cell.

## Notes

CES (T2.26) stays off (ADR-009). Event-seed mapping of crude ×4 onto `z_cost`
is T8.04 (ADR-007). No `edges.yaml` / `policy.yaml` retune.
