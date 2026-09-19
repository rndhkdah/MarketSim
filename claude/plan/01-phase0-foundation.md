# Phase 0 — Foundation

**Goal.** Turn the Layer-1 scripts into an installable, tested package; fix bugs B1–B5 (master plan §2); add the
core utilities every later phase needs (calendar, event queue, RNG streams, state hashing, Erlang kernels) and a
`World` walking skeleton. No economics changes: Layer-1 numbers must come out identical.

**Gate.** `make check` green · golden parity with legacy outputs (`io_table.json`, `betas.md`, all structural and
rotation checks) · determinism test (same seed → same hash, save/load mid-run) · CI on 3.11 and 3.12.

Reference numbers reproduced in a clean sandbox (numpy 2.4.4, scipy 1.17.1, pyyaml 6.0.3): A is 18×18, nnz 219,
density 0.68; spectral radius 0.536; output multipliers 1.58–2.77; regime demo (seed 7) corr(equity, bonds) −0.74
demand / +0.90 supply; 13/18 sectors flip sign; rotation tops early AUTOS, CONSTRUCT, CAPGOODS, TRANSPORT, SOFTWARE ·
recession STAPLES, HEALTH, REALESTATE, UTILITIES, TELECOM. Per 100 cr of final demand: gross output 209.3, wages
51.82, gross operating surplus 48.18.

---

### T0.01 — Repo audit and layout decision record
**Depends:** — · **Size:** S · **Files:** `claude/plan/DECISIONS.md`, `claude/plan/QUESTIONS.md`
**Build:** Inspect the real tree. Confirm `claude/marketsim/{config,scripts,tests}` exists and the five commands in
`claude/marketsim/INDEX.md` exit 0. Write ADR-000 "target layout" (old path → new path per master plan §4; adapt if
the real tree differs). Paste the baseline outputs above as an appendix. `DECISIONS.md` and `QUESTIONS.md` ship as stubs — fill ADR-000.
**Tests:** the five legacy commands exit 0. **Out of scope:** moving or editing any file.

### T0.02 — Packaging and tooling
**Depends:** T0.01 · **Size:** M · **Files:** `pyproject.toml`, `Makefile`, `src/marketsim/__init__.py`, `tests/conftest.py`, `.github/workflows/ci.yml`, `.gitignore`, `config/*` (copied)
**Build:** `src/` layout; dependencies per master plan §5 with extras `[api]`, `[rl]`, `[bea]`, `[dev]`; ruff
(line length 110); pytest markers `validation`, `slow`; `make check` = ruff + `pytest -m "not slow"`, `make test-all`,
`make format`. Copy `claude/marketsim/config/*` to `config/` (legacy copy stays frozen until T0.15).
**Tests:** `pip install -e .[dev]`; `python -c "import marketsim"`; `make check` green with one placeholder test.

### T0.03 — Typed config loading
**Depends:** T0.02 · **Size:** M · **Files:** `src/marketsim/core/config.py`, `tests/unit/core/test_config.py`
**Build:** pydantic v2 models for `sectors.yaml` (`SectorParams` with the `defaults:` block merged, `financials`,
`market_cap_weights`, `bond_index`) and `edges.yaml` (capex, credit, collateral, substitution, complement, labour,
policy, shocks). `load_config(config_dir, overrides=None) -> Config` (frozen); overrides by dotted path.
Validation: all 18 codes in every per-sector map; `capex.routing` and `market_cap_weights` sum to 1 (±1e-6);
`eps_own < 0`; `0 < pass_through ≤ 1`; lags ≥ 0; `erlang(k)` strings parsed to int k; edge endpoints ∈ sector codes ∪
institutions ∪ {`ALL_SECTORS`, `INVESTMENT`, `DISCOUNT_RATE`}.
**Tests:** shipped config loads; one failing fixture per rule; override works; mutation raises.

### T0.04 — Single IO loader (fixes B1)
**Depends:** T0.03 · **Size:** S · **Files:** `src/marketsim/layer1/io.py`, `tests/unit/layer1/test_io.py`
**Build:** `IOTable` (codes, names, tiers, `A` ndarray, `mu`, `final_demand` vectors, `fd_weights`, institutions,
source, coverage) with cached `L = (I−A)⁻¹`, `G = (I−Aᵀ)⁻¹`, `baseline_final_demand()`. `load_io(path)`,
`save_io(table, path)`, `resolve_io_path(cfg)` (returns `io_table_bea.json` when `world.io_source == "bea"` and the
file exists, else the seed). Load-time checks: column sums = `mu` (±5e-5, JSON is rounded to 6 dp), ρ(A) < 1,
final-demand vectors sum to 1. **From now on nothing else may read the IO table or call `build_A()` directly.**
**Tests:** round trip; loaded A equals `build_A()` within 1e-6; ρ ≥ 1 rejected.

### T0.05 — Port `build_io` into the library
**Depends:** T0.04 · **Size:** S · **Files:** `src/marketsim/layer1/build_io.py`, `config/io_seed.yaml`, `scripts/build_io.py`, `tests/golden/test_build_io_parity.py`
**Build:** Move `SECTORS`, `MU`, `MIX`, `FINAL_DEMAND`, `FD_WEIGHTS` to `config/io_seed.yaml` (flat config so the game
layer can retune without code). `build_A(seed)`, `build_final_demand(seed)`; CLI writes `config/io_table.json`.
**Tests:** output identical to the legacy JSON (±1e-9); nnz 219; each MIX column sums to 1.

### T0.06 — Port `derive_betas` as pure functions (fixes B2)
**Depends:** T0.04 · **Size:** M · **Files:** `src/marketsim/layer1/betas.py`, `scripts/derive_betas.py`, `tests/golden/test_betas_parity.py`
**Build:** `derive_betas(io, cfg, params=BetasParams()) -> Betas` with fields `beta_growth`, `beta_rate` (+ the three
channels), `beta_oil`, `beta_credit`, `idio_vol`, `mcap`. No import-time computation. Final-demand base weights from
`io.fd_weights` (not literals). Remove the unused `earn`. Constants (`DISC_PASS 0.35`, `REFI_SHARE 0.25`,
`REVENUE_LINK`, the 0.70 D&A/tax factor, cyclical mix 0.55/0.35/0.10 …) become `BetasParams` defaults.
`regime_demo(betas, seed)`. CLI writes `betas.npz` and regenerates `config/betas.md`.
**Tests:** all 18×6 values match `config/betas.md` within 5e-4; regime demo seed 7 → −0.74 / +0.90 (±0.02) and
13/18 sign flips; importing the module performs no computation.

### T0.07 — Layer-1 validation suite in pytest
**Depends:** T0.05, T0.06 · **Size:** M · **Files:** `src/marketsim/layer1/checks.py`, `tests/validation/test_io_structure.py`, `tests/validation/test_rotation.py`
**Build:** Port every `check(...)` of `validate_io.py` and `test_rotation.py` to asserts, parametrised over the IO
tables available (`seed` always, `bea` if the file exists). For `bea`, a failing *prior* is reported as
`xfail(strict=False)` with the numbers printed — "if a prior fails against real data, the prior was wrong; read it
before editing".
**Tests:** all legacy checks pass on the seed; suite provably reads through `load_io` (monkeypatch `build_A` to raise).

### T0.08 — `vic3_compare` as a regression test
**Depends:** T0.02 · **Size:** S · **Files:** `scripts/vic3_compare.py`, `tests/validation/test_vic3_mechanics.py`
**Build:** Move the script, expose `vic3_price`, `allocate`, `stock_loop` as functions, delete the duplicated print.
**Tests:** the script's checks A–D as asserts (price rule 25–175 %, saturation at 2×, availability substitution,
undamped stock diverges at every gain, 10 % decay stabilises).

### T0.09 — BEA fetch hardening
**Depends:** T0.04, T0.07 · **Size:** M · **Files:** `scripts/fetch_bea_io.py`, `tests/unit/layer1/test_bea_aggregation.py`, `claude/plan/reports/bea-howto.md`
**Build:** Extract `aggregate_use_table(df, concordance, splits) -> (Z, gross_output, coverage)`. Add `--validate`
(runs T0.07 against the written file through `load_io`). Friendly failure without network. Write the how-to: run
locally (needs `apps.bea.gov`), expected output, and the known follow-up — final-demand vectors stay seeded (T8.04).
**Tests:** synthetic 4-industry xlsx fixture → known Z, A, coverage; split codes distribute rows **and** columns;
coverage below the floor exits non-zero. Never runs in CI against the network.

### T0.10 — Calendar, clock and event queue
**Depends:** T0.02 · **Size:** M · **Files:** `src/marketsim/core/calendar.py`, `src/marketsim/core/clock.py`, `tests/unit/core/test_clock.py`
**Build:** `Calendar(days_per_month=21)` → `(year, month, day)`, `is_month_end`, `is_quarter_end`, `months_elapsed`.
`EventQueue` on `heapq` keyed `(timestamp, priority, sequence)`, sequence strictly increasing; `schedule`, `pop_due(t)`,
`to_state/from_state`.
**Tests:** ties break by priority then sequence; `pop_due` deterministic; save/load preserves order; boundaries for 3 years.

### T0.11 — RNG streams, state protocol, hashing
**Depends:** T0.02 · **Size:** M · **Files:** `src/marketsim/core/rng.py`, `core/state.py`, `core/hashing.py`, `tests/unit/core/test_determinism_utils.py`
**Build:** `RngHub(root_seed).stream(name)` → cached `Generator(PCG64(SeedSequence(root, spawn_key=(k,))))`,
`k = int.from_bytes(sha256(name)[:4], "little")`; stream states serialisable. `Stateful` protocol. `canonical_bytes`
(sorted keys; ndarray → dtype, shape, C-order little-endian bytes; float → `struct.pack("<d")`), `state_hash`,
`state_diff(a, b, tol)`.
**Tests:** streams independent of creation order; hash invariant to dict insertion order; save → load → continue
equals an uninterrupted run.

### T0.12 — Erlang kernels
**Depends:** T0.02 · **Size:** S · **Files:** `src/marketsim/core/erlang.py`, `tests/unit/core/test_erlang.py`
**Build:** `ErlangChain(k, mean_m, shape)` — mass-conserving: `push(inflow) -> outflow`; per stage `q += x; x = a·q;
q −= x` with `a = k/(mean_m + k)` (array-valued means allowed); `seed(flow)` fills each stage with `flow·(1−a)/a`;
`content()` = Σ stages. `ErlangSmoother(k, mean_m, init)` — signal form `s_i += a·(x − s_i)`. `mean_m = 0` → pass-through.
**Tests:** impulse response sums to 1 and has mean `mean_m` (±1e-6); `k=1` geometric; after `seed(f)` a constant
inflow `f` returns exactly `f` at the first push; mass conserved (Σ in = Σ out + content); shapes `(R,S)`; mixed means.

### T0.13 — World skeleton and module pipeline
**Depends:** T0.10, T0.11, T0.12 · **Size:** M · **Files:** `src/marketsim/world.py`, `src/marketsim/core/module.py`, `config/world.yaml`, `tests/integration/test_world_skeleton.py`
**Build:** `Module` protocol (`name`, `reset(ctx)`, `on_phase(ctx, phase)`, `to_state/from_state`); `Phase` enum
`EVENTS, INGEST, REAL, SETTLE, VALUE, MARKET, PUBLISH` (master plan §7). `World.create(config_dir, seed, overrides,
scenario)`, `reset()`, `submit(agent_id, actions)`, `step(n=1)`, `observe(agent_id)`, `state_hash()`, `save/load`.
`world.yaml`: `scale`, `seed`, `mode` (game|professional), `run_mode` (lockstep|realtime), `io_source`, `modules`.
**Tests:** empty world runs 1,000 ticks; equal seeds → equal hash every tick; a dummy random module makes different
seeds diverge; save/load mid-run continues identically.

### T0.14 — Config corrections B3–B5 with ADRs
**Depends:** T0.03 · **Size:** S · **Files:** `config/edges.yaml`, `config/dynamics.yaml` (stub), `src/marketsim/core/gates.py`, `claude/plan/DECISIONS.md`, `tests/unit/core/test_gates.py`
**Build:** `credit.bank_capital_gate` gains `ratio: equity_over_rwa`, `baseline_capital_ratio: 0.125`,
`normalise_at_baseline: true`. `capex` gains `unit: pct_pts_of_K_per_year`, `unit_scale: 0.01`, `q_scale: 0.1`,
`supply_line_weight: 0.85`. `dynamics.yaml` starts with `production.mode` (see `02-…` §2.11).
`logistic_gate(c) = min(1, σ(k(c−mid)) / σ(k(c0−mid)))`; `asymmetric_gate(gap) = exp(e·(up·max(gap,0) + down·min(gap,0)))`
(level-based, so no ratchet). ADR-001…003 record what changed and why.
**Tests:** gate(0.125) = 1; gate(0.085) = 0.5/0.9734; monotone; the legacy value at 1/11 (≈0.63) documented in a
comment; Layer-1 golden tests untouched.

### T0.15 — Legacy freeze and removal — **HUMAN GATE**
**Depends:** T0.05, T0.06, T0.07, T0.08 · **Size:** S · **Files:** `claude/marketsim/*`, `claude/engine-design.md` (pointers only)
**Build:** After parity is green and the human approves: replace `claude/marketsim/scripts|tests|config` with a README
pointing to the new locations; update `INDEX.md`, `README.md` and doc pointers. Docs keep their content.
**Tests:** `make check` green; no file under `src/` or `tests/` imports from `claude/`.
