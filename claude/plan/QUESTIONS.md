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
