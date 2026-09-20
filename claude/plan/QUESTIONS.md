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

## T2.32 — Dual-mandate calendar flips two §2.12 price windows

- what blocks: T2.32 attaches the D15 autopilot (8 meetings, φ_u 1.0, 25bp grid). Supply −3 % 12q: output window still negative, but mean `lvl[6..36]` is slightly negative (passthrough −0.006 %, banks −0.49 %, credit −3.7 %). Credit-on fiscal +5 %: output window still positive, mean `lvl[18..36]` −0.09 %. T2.35 wants §2.12 unchanged within tolerance; retuning `policy.yaml` / `edges.yaml` needs an ADR.
- option A: Keep D15 defaults; xfail those two price rows; report in T2.35.
- option B: Human accepts a config / ADR change (φ_u 0, more core weight, or keep the quarterly skeleton as the default autopilot).
- recommendation: A. Steady state stays exact; other §2.12 signs and the monetary timing block still pass.

## T3.12 — ±5 % income L2-within-20 % vs rank-only η

- what blocks: T3.12 wants the tiers×wants composition response to ±5 % income within 20 % of scalar-η. T3.11 (2026-09-20) is rank-only Spearman ≥ 0.7; committed mean |Δη| = 0.33. Measured: cosine(Δs_tw, Δs_se) ≈ 0.91, L2 relative residual ≈ 0.49, ||Δs_tw||/||Δs_se|| ≈ 0.63. Hitting 20 % L2 needs η magnitudes, which that decision waived.
- option A: Keep rank-only; test cosine ≥ 0.8 and matching STAPLES/DISCRET signs (implemented).
- option B: Re-open T3.11 to fit η magnitudes (risks Spearman / basket / two-shape).
- recommendation: A. Do not retune `edges.yaml` / `sectors.yaml`.

## T3.12 — BASIC_GOODS vanish on the top 5 deciles

- what blocks: §3.4 wants BASIC_GOODS **absolute** real spend to fall for the top 5 deciles as mean y rises 50 %. Fitted `y_pk` ≈ 1.80 (upper bound) so only the top 2 deciles are past the vanish peak on that experiment. National BASIC want share does fall.
- option A: Treat "past the peak" (top 2 with this fit) plus national share-down as the mechanism test (implemented).
- option B: Add an Engel residual and re-run T3.11 so all five rich deciles are past `y_pk`.
- recommendation: A until a magnitude re-fit is authorised. Survival-first + scale-to-1 is required for the HOUSEHOLD basket (actual-budget V is 60 % FOOD vs a 20 % STAPLES+AGRIFOOD θ, RAS-infeasible).

## T3.15 — R = 3 live monthly stepper

- what blocks: Gate 6 asks to rerun §2.12 at R = 3. `RealEconomy.step_month` is the R = 1 orchestrator (`compute_real_baseline` always sets `R = 1`). Regional SS, Armington and the gate-3 price path exist (T3.03–T3.05) but orders, labour, the ledger and demand.allocate are national.
- option A: Keep the live tick at R = 1; run gate-3 disasters on the trade/price layer and §2.12 with `tiers_wants` at R = 1 (implemented). Wire the R = 3 tick under a later card.
- option B: Regionalise `step_month` now (large, not in the T3.15 file list).
- recommendation: A. Do not invent a second orchestrator without a card.

## T3.15 — monetary level window under `tiers_wants`

- what blocks: Passthrough monetary +100 bp still has `gap[0..36] < 0`, but `lvl[17..36]` is slightly positive (~+0.07 pp) with `demand.mode: tiers_wants` (scalar-η is negative). Composition is more rate-sensitive (AUTOS/DISCRET) so the CPI window flips. Retuning `edges.yaml` / `sectors.yaml` needs an ADR.
- option A: xfail the level window; keep the gap sign (implemented).
- option B: Human accepts a config / ADR change.
- recommendation: A.

## T4.11 — historic cost-push seeds are not 24-month SFC-finite

- what blocks: §4.4 oil seed is lognormal median 1.25 (crude ×4 as `z_cost`); energy_inflation ENERGY uniform 0.5–0.9. The Phase-2 cost-push IRF is **+0.30** and already delivers ~+72 % CPI @12m (T2.24). At the historic seeds, `step_max` 0.15 compounds, the dual-mandate rule ratchets `r` above 90 %, ENERGY output hits 0 around month 12, and SFC / IEEE overflow appear by month 17–18. Covid labour_supply −10…−15 % plus want shifts is SFC-finite but HEALTH never rises vs baseline (labour cap dominates); DISCRET < −15 % holds in the want-shift window.
- option A: Keep the YAML seeds for T4.13; assert §4.5 directions on the finite prefix (12m oil/energy; covid relative HEALTH vs DISCRET in months 1–6). 10× storm (T4.10) omits those three ids until T4.13 maps crude ×4 onto a `z_cost` the engine can digest (likely ~0.3–0.4, not ln 4).
- option B: Rewrite `config/events/*.yaml` now so medians match the IRF-safe band (changes the shipped historic table before the calibration card).
- recommendation: A. T4.13 is the magnitude card; do not retune `edges.yaml` / `sectors.yaml`.

## T5.16 — hybrid World stepper vs cell-level identity

- what blocks: Gate 1 wants a 10-year GDP path after splitting every NPC cell into N autopilot firms. `RealEconomy.step_month` is still the NPC-only orchestrator; wiring firms into orders/prices/labour is a new monthly kernel (not in the T5.16 file list).
- option A: Assert the linear identity — N equal autopilot firms with `(se,inv,k,backlog)/N` reproduce the NPC plan to 1e-9 (implemented). Leave the live hybrid tick for a later card.
- option B: Regionalise `step_month` now so firms enter R4–R8 (large, out of file list).
- recommendation: A. Same reason as QUESTIONS T3.15.


