# Phase 3 validation (T3.15)

Regional geometry is R = 3 (`CAPITAL`, `INDUSTRIAL`, `RESOURCE`). The monthly
orchestrator remains R = 1 (Phase-2 parity); regional disaster / strike paths
use the T3.03–T3.05 trade and price layer. Live IRFs below are R = 1 with
`demand.mode: tiers_wants`. See QUESTIONS.md T3.15 for the R = 3 stepper.

## Gate 1 — R = 1 parity

`tests/golden/test_r1_parity.py` vs `tests/golden/data/r1_phase2.npz`: no-shock
and demand-IRF series match Phase 2 to 1e-12 (T3.02).

## Gate 2 — regional steady state

National totals of `x0` and `C0` at R = 3 equal the Phase-2 baseline to 1e-9.
RESOURCE is a net ENERGY exporter; CAPITAL a SOFTWARE exporter (T3.03).

## Gate 3 — propagation and the price band

A −10 % MATERIALS productivity shock in RESOURCE raises MATERIALS prices in
every region, most at the source. After month 24, `|ln p_r − ln p̄|` for
tradables stays inside `max link cost + 0.05` (0.09 + 0.05). The same band
holds for an INDUSTRIAL AUTOS strike. Non-tradables (CONSTRUCT, REALESTATE)
are not required to converge.

## Gate 4 — demand system

| check | result |
|---|---|
| HOUSEHOLD basket `V @ M` vs θ | max abs 9.99e-16 in-calibrator; reload 1e-11 (< 1e-9) |
| Spearman(implied η, config η) | **0.8164** (rank-only ≥ 0.7; master-plan cites 0.8) |
| mean \|Δη\| | 0.3287 (master-plan ≤ 0.25 waived 2026-09-20) |
| §3.4 Engel | STAPLES share falls with y; EATING_OUT / LUXURY rise; DISCRET drops more than STAPLES in a downturn |
| default `demand.mode` | `scalar_eta` (Phase 2 unchanged) |

## Gate 5 — entry / exit

Permanent +10 % AUTOS demand: excess profit below θ_e within 8 years (T3.14).
No entry/exit at baseline (bitwise). Exit write-offs pass SFC.

## Gate 6 — §2.12 with `tiers_wants` (R = 1, passthrough)

| primitive | gap window | level window | note |
|---|---|---|---|
| demand +2 % | + | + | stronger than scalar-η |
| fiscal +5 % | + | + | |
| cost_push ENERGY +30 % | gap[5..48] − | lvl[5..24] + | |
| row −10 % | gap[0..24] − | | |
| risk +1 pp | gap[0..24] − | | |
| monetary +100 bp | gap[0..36] − | lvl[17..36] **+** | xfail; QUESTIONS T3.15 |

No-shock 24-month max `|x/x0 − 1|` with `tiers_wants` is ~1e-10 (stationary).

## Gate 7 — golden national series

`tests/golden/data/aggregate_baseline.npz` — 120 months, passthrough, π\* = 0,
`demand.mode: scalar_eta`, no primitive shocks, SFC on. Replay matches to 1e-10.
Script tag: `passthrough,pi_star=0,demand_mode=scalar_eta,no_shocks,120m`.
Intended as the T5.16 hybrid ≈ aggregate reference.

## Master plan §9 (Phase 3)

| required | status |
|---|---|
| R = 1 parity | yes (1e-12) |
| regional shocks propagate; prices in band | yes (gate 3) |
| basket reproduced exactly | yes (1e-9) |
| η rank-corr ≥ 0.8 | 0.816 |
| aggregate baseline recorded | yes (gate 7) |
