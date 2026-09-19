# QUESTIONS — raised by agents, answered by the human

Format: `## <task ID> — <one-line question>` · what blocks · option A · option B · recommendation.

## T0.01 — Five legacy Layer-1 commands do not exist

- what blocks: There is no `claude/marketsim/{config,scripts,tests}` and no `INDEX.md` in this repo or the plan zip. T0.01's "five legacy commands exit 0" cannot be run.
- option A: Skip the commands; record the reconstruction layout (ADR-000) and treat published goldens as the baseline.
- option B: Stop Phase 0 until a legacy dump is provided.
- recommendation: A (done). Layer 1 is reconstructed from master-plan / Phase 0 gate numbers.

## T0.05 / T0.06 — No bit-identical `io_table.json` or `betas.md`

- what blocks: Golden-vs-legacy bit-match tests have no fixture.
- option A: Golden tests = published invariants (nnz 219, ρ 0.536, multiplier band, FD identity, regime seed 7 windows) and a regenerated `config/betas.md`.
- option B: Invent a `betas.md` and claim parity.
- recommendation: A (done).

## T0.02 — Nested GitHub Actions workflow will not auto-run

- what blocks: GitHub only loads `.github/workflows` at the **repository** root. The card's workflow lives at `marketsim/.github/workflows/ci.yml`. Adding a root workflow would edit a Vic3-adjacent path.
- option A: Keep the nested file; run `make check` locally / in a later root dispatcher.
- option B: Add a one-file root workflow that only `cd marketsim && make check`.
- recommendation: A until the human wants CI on origin.

## T0.15 — No legacy tree to freeze

- what blocks: HUMAN GATE asked to replace `claude/marketsim/scripts|tests|config` with pointers. Those paths never existed here.
- option A: Treat T0.15 as N/A; gate = `marketsim/ make check`, invariants, determinism, Vic3 pytest still green.
- option B: Wait for a legacy dump before ticking T0.15.
- recommendation: A. Left **GATE P0** for the human.

## T2.23 — Catastrophe months 1–6 are not an output drop

- what blocks: §2.12 wants `gap[1..6] < 0` and CONSTRUCT above baseline within 24m. T2.18 posts the full replacement-cost claim and adds it as same-month final demand, so months 1–5 are `+0.5%…+2.8%` GDP (CONSTRUCT `+11%…+17%`) and the GDP trough is month 14. No extra lag is named in §2.9.
- option A: Keep same-month recon FD; treat the month-1–6 sign as a report item and keep the CONSTRUCT-within-24m / CPI-up checks.
- option B: Delay or Erlang-spread `claims_to` demand so months 1–6 go negative (invents a lag not in the spec).
- recommendation: A until the human names a recon lag. Tests xfail with this id; do not invent a kernel.
- **answered 2026-09-19: A.** Keep same-month recon FD and the xfail. No invented catastrophe lag.

## T2.23 — Credit-on demand / monetary IRFs miss the §2.12 windows

- what blocks: Gate 6 says banks+credit+edges must leave §2.12 green. With `credit.gate_enabled: true` the stub `V_RE` (earnings frozen, only `Δρ`) makes every rate rise tighten `Λ_coll`. Measured: demand +2% 24-month gap sum `−2.8%` (price window also slightly negative); monetary trough month 47 at `−0.86%` (passthrough is month 13, `−0.36%`, rebound 0.21, AUTOS 9 < CONSTRUCT 14 < CAPGOODS 15). Retuning `edges.yaml` needs an ADR.
- option A: Keep specified defaults; xfail the two credit-on rows; report in T2.24. Live `E^e` (still §2.10) might restore demand but would deepen the hike.
- option B: Human accepts a config / ADR change (weaker collateral down-factor, or earnings-updated `V_RE`).
- recommendation: A. Passthrough and `banks.mode: full` (gate off) already meet the table and the monetary timing block.
- **answered 2026-09-19: A.** Keep specified defaults and the credit-on xfails. No `edges.yaml` / `sectors.yaml` retune.

## T2.24 — Stochastic volatility ranking misses the §2.12 order

- what blocks: 1,200 months × 3 seeds at π\* = 2 % are finite, `|gap| < 2.2 %`, U ∈ (3 %, 7 %), envelope ratio < 1.5. Ranking is not: SEMIS is 5th–6th (AGRIFOOD / CAPGOODS lead); CONSTRUCT is 7th–8th; TELECOM is mid-pack on seeds 1–2. Soft `sd(I)/sd(GDP)` is 0.88–1.03 (target 3–4; prototype ≈ 7). Cost-push ENERGY +30 %: GDP trough `−13 %` m42, CPI `+72 %` @12m (prototype `−2.7 %` / `+2.0 %`). Changing `edges.yaml` / `sectors.yaml` needs an ADR.
- option A: Keep defaults; report the gaps and tuning ideas; xfail the ranking test.
- option B: Human picks a retune (inventory leak, bullwhip, `kappa_util`, cost-push pass-through).
- recommendation: A (card: out of scope to change defaults).
- **answered 2026-09-19: A.** Keep defaults; ranking stays xfail; gaps stay in the T2.24 report. No retune.
