# Phase 10 — Integration: make the engine one working World

**v1.1 — 2026-09-22.** Cards T10.00–T10.17. Written against `main` at `c7f9e95`; v1.1 adds the recorder and dashboard cards (T10.16 / T10.17, shipped) and the verified test-suite notes in §10.2.

**Goal.** MarketSim has ~31k LOC of tested components and no top-level assembly. Every public entry point
creates an **empty** `World`: `WorldManager.create_world` passes no modules (`api/sessions.py:133`), the SDK
envs attach only the inbox sink (`sdk/gym_env.py:272`), `cfg.world.modules` is declared (`core/config.py:225`,
`config/world.yaml:8`) but never read, and nothing in `src/` composes economy + events + releases + valuation +
market on one ledger. Consequences, all verified: orders submitted through the API are swallowed by
`FirmAgentModule` and dropped when `World.step` clears `_inbox` at PUBLISH (`world.py:70-71`), `observe()`
returns `{tick, agent_id}` plus whatever events/releases published, gym rewards are identically zero (they read
`portfolio.net_worth` out of an empty observation, `sdk/gym_env.py:103-107`), firms founded through REST are
dict rows (`api/rest.py:1002`), `GET …/goods` and `GET …/labour/{region}` return empty models
(`api/rest.py:1088`, `:1103`), and the Phase-7 "50,000-tick multi-agent" gate ran on an empty world.

This phase delivers the system-level features the mission needs — a training environment **and** a game
backbone through one API — as independently mergeable units: one card, one worktree, one PR. It follows the
repo's own workflow: tests first, `make check` green, no tolerance loosened, no `config/sectors.yaml` /
`config/edges.yaml` value changes, everything on the ledger, no new subsystem without a card.

**Out of scope on purpose.** Multi-country/FX and cluster eval (deferred, ADR-017); the goods layer (ADR-006);
CES on intermediates (ADR-009); a Rust core (ADR-016); T8.04 calibration (human gate); T8.05 / T8.11 / T9.06
(open PR #2 awaiting review); the documented `xfail`s and the cost-push magnitude (human decisions);
wiring `MonetaryFramework.qe` to the T6.28 QE desk (economics decision — see §10.2 and `QUESTIONS.md`).

---

## 10.1 Gate

Phase 10 is done when, on the assembled world (`config/world.yaml` → `modules: [events, real_economy,
data_releases, valuation, market]`):

1. **Deterministic.** `World.create("config")` steps 3 simulated years (756 ticks); same seed → identical
   `state_hash()`; no NaN in any published array; iteration order stable (no unordered set/dict in the core).
2. **SFC-clean.** The monthly SFC assertion runs on the one ledger for every month of that run, including the
   months in which an agent posts capital, an order fills, and a firm is founded.
3. **Save carries the world.** `World.save(p)` → `World.load(p, "config")` **with no `modules` argument**
   rebuilds economy + market + valuation + news + releases and reproduces both `state_hash()` and the hash
   after one further `step()`.
4. **Orders fill on a venue.** A marketable limit order submitted through REST (and through `LocalClient`)
   reaches a venue, fills on the next tick (master plan §6 "next-tick fills"), settles with `settle_fill` on
   the shared ledger, and appears in `observe(agent)["fills"]`; `assert_consistent` green with an `AGENT:`
   counterparty.
5. **Firms live through the API.** `POST …/firms` founds a firm in the economy's `FirmRegistry` with books on
   the shared ledger; `GET …/firms/{fid}/state|financials|reports` return live numbers; a submitted price cut
   changes that firm's own share the next month; a strategic default resolves in one step, SFC-green.
6. **Rewards are ledger-backed.** Gym / PettingZoo rewards move with the agent's ledger net worth and change
   sign with a price move on a held position; bankruptcy also terminates on non-positive ledger net worth.
7. **Earlier gates re-run on the assembled world.** Gate P5-1 (hybrid ≈ aggregate: GDP within 1 %, sector
   outputs within 2 %, CPI within 0.5 % over 120 months) and gate P7-1 (`examples/train_smoke.py --ticks 50000`,
   invariants all zero) pass with the real modules attached, not on an empty world; `scripts/bench.py --gate3`
   still meets its floors (≥ 1,000 ticks/s small world in-process).
8. **Suite honest and green.** `make check` from a clean checkout: `ruff` clean and `pytest -m "not slow"`
   green with **no** `--deselect`, no tolerance changed, no test rewritten to pass, and the three tests that
   are red today either fixed or documented in `DECISIONS.md` with an accepted ADR.

Human gate: the coordinator reports 1–8 with measured numbers; the human accepts (or sends items back).

---

## 10.2 Process

### Waves

| wave | cards | how |
|---|---|---|
| 0 | T10.00 | Coordinator lands it on `main` **before** any worker starts. Everything else forks from it. |
| 1 | T10.01–T10.12 | Twelve worktrees in parallel, one PR each, all targeting `main`. |
| 1b | T10.13 (parent T10.04), T10.14 (parent T10.07) | Spawned when the parent PR **merges**. |
| 2 | T10.15 | One sequential card after every other card has merged. |
| 3 | — | Coordinator wiring PR: flip `config/world.yaml`, re-run the gates, tick `PROGRESS.md`. |

### Rules every worker follows

1. **One card.** Stay inside your card's **Files**. If you must touch another file, say why in the PR body.
2. **Never edit `claude/plan/PROGRESS.md` or `claude/plan/QUESTIONS.md`.** Every PR would conflict on adjacent
   lines. Questions go in the PR body under a `## QUESTIONS` heading (task id + your best two options); the
   coordinator ticks cards and consolidates questions after merge.
3. **Never edit `claude/plan/00-MASTER-PLAN.md`.** T10.00 pre-registers this batch's config keys in §8. A key
   that is not there goes in the PR body under `## QUESTIONS`.
4. **Known-red tests.** Three tests fail on `main` at `c7f9e95` before any unit starts:
   - `tests/golden/test_r1_parity.py::test_r1_demand_irf_matches_phase2_golden`
   - `tests/validation/test_credit_crunch.py::test_credit_crunch_gate_and_recovery`
   - `tests/validation/test_policy_levers.py::test_vat_step_and_no_permanent_inflation`

   Confirm they are red before you start. Every card except T10.02 deselects exactly these three (§10.4) and
   must not touch them or their goldens. If one is *green* for you, say so in the PR body — do not silently
   adopt it. **Verified red** on Python 3.12.14 / numpy 2.x at `c7f9e95`;
   `test_vat_step_and_no_permanent_inflation` misses by a hair (|π12| = 0.015031 against a 0.015 bound).

   Two further tests are **performance floors, not correctness**, and fail under `-n auto` on a loaded
   machine while passing serially:
   - `tests/integration/test_perf_smoke.py::test_r1_throughput_5000_ticks_per_s` (measured 3,591 t/s under
     `-n auto`, passes alone)
   - `tests/integration/test_real_economy.py::test_monthly_step_under_5ms` (5.2 ms under load, ~4.9 ms alone)

   Deselect them too when you run with `-n auto`, and do not "fix" them by loosening the floor. **T10.01 owns
   the decision** — adopting `-n auto` in `make check`/CI without marking these two makes CI red at random.
5. **Git hygiene.** Stage by path only (`git add <path>` — never `git add -A`). Never commit `.venv/`,
   `.claude/`, `scratch_tmp/`, `__pycache__/`, or `*.npz` outside `tests/golden/data/` (`.gitignore` ignores
   `*.npz`, so a new golden needs `git add -f tests/golden/data/<name>.npz`).
6. **Stay out of open PR #2's files:** `config/kr/*`, `scripts/fetch_bok_io.py`, `src/marketsim/api/grpc/*`,
   and the `grpc` extra in `pyproject.toml`.
7. **No new wire models.** `tests/unit/api/test_schemas.py:221-229` freezes every model in
   `src/marketsim/api/schemas.py` against `tests/golden/data/api_schema_v1.1.json`, so any new or changed model
   forces a version bump. Only **T10.10** may add one (and it bumps `API_SCHEMA_VERSION` to `v1.2` with a new
   golden, leaving `v1` and `v1.1` snapshots in place). Everyone else uses path/query params, existing models,
   or a plain `dict` response. Response models defined **inside `api/rest.py`** (e.g. `FirmStateView`) are not
   part of that golden — extending those is allowed.
8. **Commit message** `T10.xx: <title>`, ending with the repo's attribution trailer. PR title: the card title.

### Coordinator checklist

1. Land **T10.00** on `main` and push, so worker PRs target it.
2. Spawn wave 1 (12 workers). Merge order when PRs go green: **T10.02, T10.03, T10.01** first (they own the
   harness), then the rest as they arrive. Expected conflicts are line-local: `real/economy.py` (T10.06 /
   T10.07 / T10.08), `api/rest.py` and `api/sessions.py` (T10.09 / T10.11 / T10.12),
   `src/marketsim/assembly.py` (one registry line per card).
3. Spawn **T10.13** when T10.04 merges and **T10.14** when T10.07 merges.
4. Wiring PR: flip `config/world.yaml` to `modules: [events, real_economy, data_releases, valuation, market]`,
   run `make check`, `python scripts/bench.py --gate3`, `python examples/train_smoke.py --ticks 50000`
   (gate P7-1) and T10.07's gate P5-1 on the assembled world; tick the Phase 10 rows in `PROGRESS.md`; append
   the consolidated `## QUESTIONS` sections to `QUESTIONS.md`.
5. Wave 2: **T10.15** alone; after it lands re-verify the R=1 parity golden (1e-12) and T10.07's
   zero-firms-bitwise test.
6. Report the §10.1 gate numbers to the human. Open decisions that stay with the human: GATE P2–P8 acceptance,
   the documented `xfail`s, the cost-push magnitude, T8.04, PR #2 review, and whether
   `MonetaryFramework.qe` should route to the T6.28 `cb_operations` desk (economics decision — `DeferredQE`
   still raises `NotImplementedError` at `real/policy/monetary.py:60,64` and is the live `qe` field).

---

## 10.3 Shared contracts

These are copied verbatim into every worker prompt. They are what keeps twelve parallel PRs mergeable.

**Module names.** `events`, `real_economy`, `data_releases`, `valuation`, `market`, `firm_agent`.
**Canonical order** (the intra-phase determinism tie-break; modules run in list order inside every phase):
`events, real_economy, data_releases, valuation, market, firm_agent`.
**Registry:** `marketsim.assembly.MODULE_FACTORIES` — one line per module name; each card replaces exactly its
own line (T10.00 ships the skeleton, every factory raising `ConfigError("<name> is not shipped yet")`).

**Lookup.** Find other modules only through `marketsim.core.lookup.find_module(world, name)`. Never scan
`world.modules` inline (T10.03 removes the five existing copies of that idiom).

**Ledger.** There is one ledger. Get it with `marketsim.core.lookup.world_ledger(world)`: the live
`RealEconomy.ledger` when an economy is attached, else a lazily created `Ledger.empty()` cached on the
`World`. **Resolve it at the point of use, never cache it across ticks** — `RealEconomy.reset` replaces
`self.ledger` (`real/economy.py:697`).

**Observations.** Re-emit everything you publish **every tick**: `World.step` clears `_observations` at INGEST
(`world.py:68-69`). Write only through
`ctx.world._observations.setdefault(key, {}).update({...})` — never assign the whole dict (the bug T10.00
fixes at `events/releases.py:246`, where `_observations["*"] = {"releases": …}` clobbers the news the events
module published at `events/scheduler.py:179-180`). Key `"*"` is public, `agent_id` is private.
Shapes are the frozen, `extra="forbid"` models in `src/marketsim/api/schemas.py` (`ApiModel`, `:62-65`):

| key | shape | source model |
|---|---|---|
| `prices` | `{code: float}` — goods price index per sector code | `Observation.prices` |
| `quotes` | `{symbol: float}` — mid per instrument | `Observation.quotes` |
| `markets` | `{symbol: {bid, ask, last, adv, halted}}` | `Observation.markets` |
| `goods` | `{"reference_prices": [{region, sector, value}], "demand": [{region, sector, value}], "competitor_prices": [{seller_id, region, sector, posted_price}]}` | `GoodsView` `:503-508`, `CellQty` `:87-92` |
| `labour` | `[{region, labour_force, unemployed, wage_bar, vacancies}]`, one row per configured region code | `LabourView` `:511-518` |
| `regions` | `{code: {...}}` | `Observation.regions` |
| `bonds` | `{...}` | `Observation.bonds` |
| `news` | `[NewsItem dict]` | `NewsItem` `:113-126` |
| `releases` | `{...}` visible vintages only | `Observation.releases` |
| `macro` | `{gdp, cpi, core_cpi, u, infl, policy_rate}` from **published** vintages | new in T10.10 |
| `fills` | `[Fill dict]` (private) | `Fill` `:174-185` |
| `orders` | `[OpenOrder dict]` (private) | `OpenOrder` `:305-315` |
| `portfolio` | `{cash, positions: [{instrument, qty, market_value}], net_worth}` (private) | `PortfolioView` `:648-666` |
| `reports` | `[{firm_id, live, books, tick}]` (private) | `Report` `:669-675` |

`macro` is invisible to REST / `LocalClient` until T10.10 adds it to `OBSERVATION_FIELDS`,
`OBSERVE_PUBLIC_KEYS` and the `LocalClient` whitelist — before that, test it on `world.observe()` directly.
**Never publish** a nested key named `xi`, `xi_j`, `mispricing`, `sentiment`, `noise`, `capital`,
`arb_capital`, `A`, `last_impact`, `truth`, `month_truth`, `_month_truth`, `unpublished`, `noise_scale`,
`idio_vol`, `vintages_unpublished`, or anything `_`-prefixed: `sanitise_observe`
(`scenarios/randomise.py:56-75`) strips them silently, so a test that asserts on them passes for the wrong
reason and a real leak would be invisible.

**Valuation ↔ market.** The valuation module exposes an **in-process** `fair_values() -> dict[str, float]`
(one entry per tradable instrument symbol; bond buckets = 1.0 at baseline, `EQ:NPC:*` = baseline `V`) and
publishes nothing public. The market module reads it through `find_module(world, "valuation")` and falls back
to baseline constants when valuation is absent. Nothing else couples the two.

**Orders.** The market module owns order ids (world-level, monotonic — venue ids are per-book:
`market/clob.py:253`, `market/mm.py:194`), resting / cancel state, fills and idempotency.
`accept(agent_id, OrderRequest) -> OrderAck` parks an order; INGEST drains parked orders into venues.
Inbox consumers **must ignore payloads that are not theirs** — the inbox carries `OrderRequest`s,
`FirmDecision`s and raw dicts (`FirmAgentModule` currently swallows all three, `world.py:164-171`).

**RNG.** Randomness only through `ctx.world.rng.stream("<name>")`, and every stream you use must be touched in
`reset()` so the stream order is seed-stable. In use today: `events`, `flow`, `noise`, `sentiment`
(`Mispricing` defaults), `releases`, `firms.rnd`, `randomise`. Keep those names; add a new one only if your
card says so (`monetary_rule` in T10.06).

**State.** Every module implements `to_state()` / `from_state()`; `World.save` → `World.load` must hash-match.
Arrays go through the existing numpy-friendly serialisation (`World._json_default`, `world.py:123-126`).

**Config.** Every parameter lives in `config/*.yaml` and is validated by a pydantic model; no magic numbers
(a literal other than 0/1/2 needs a named config key or a comment citing the spec equation). Docstrings state
units (cr/month, index, annual decimal, 100bp, months).

---

## 10.4 E2E recipe

Every worker runs this. Setup, inside your worktree (any Python ≥ 3.11 — check `python3 -V`; the project
requires `>=3.11` and the CI matrix is 3.11 / 3.12):

```bash
python3.12 -m venv .venv          # or python3.11 / any interpreter >= 3.11
.venv/bin/pip install -q -e ".[dev]"
export PATH="$PWD/.venv/bin:$PATH"
```

Unit tests (the three deselected tests are red on `main` before your change — §10.2 rule 4):

```bash
ruff check src tests scripts
pytest -m "not slow" -n auto -p no:cacheprovider \
  --deselect tests/golden/test_r1_parity.py::test_r1_demand_irf_matches_phase2_golden \
  --deselect tests/validation/test_credit_crunch.py::test_credit_crunch_gate_and_recovery \
  --deselect tests/validation/test_policy_levers.py::test_vat_step_and_no_permanent_inflation
```

End-to-end — all five must exit 0:

```bash
python scripts/bench.py --small-world --ticks 500 --warmup 21 --config config
#   market + economy on one World; prints ticks_per_s, n_instruments, n_regions_live
python examples/quickstart_trader.py --config config
#   REST stack; tick_after == tick_before + 1, 64-hex state_hash
python examples/quickstart_operator.py --config config
#   REST stack; firm live == True
python examples/train_smoke.py --ticks 16 --seed 0
#   PettingZoo; invariants all zero
python -c "from marketsim.scenarios.runner import run_scenario; \
w = run_scenario('config', 'config/scenarios/chip_script.yaml', n_ticks=63); \
print(w.clock.tick, w.state_hash(), sorted(w.observe('x').keys()))"
```

Then your unit-specific observable:

| card | prove it with |
|---|---|
| T10.00 | `pytest -m "not slow" -n auto` collects with **zero** collection errors; `git status` clean of `scratch_tmp/` and `.claude/` |
| T10.01 | `time pytest -m "not slow" -n auto` under 120 s; `marketsim-run --ticks 63 --save out.json` prints three monthly lines and a hash equal to `World.load('out.json', 'config').state_hash()` |
| T10.02 | the three deselected tests pass **without** `--deselect`, or the PR body names exactly which stays red, why, and the proposed ADR |
| T10.03 | `[m.name for m in World.create('config').modules] == ['events','real_economy','data_releases']`; after 63 ticks `observe('x')` has `news` and `releases`; `World.save` → `World.load` (no `modules` arg) → equal hashes |
| T10.04 | small-world bench prints `n_instruments=25` and ≥ 1,000 ticks/s; `observe('x')['quotes']` has 25 symbols |
| T10.05 | at tick 0 `fair_values()['GB_BOND'] == 1.0` and every `EQ:NPC:*` equals its baseline `V`; after a scripted +100 bp REALESTATE falls more than STAPLES |
| T10.06 | after 21 ticks `world.observe('x')['macro']['cpi']` exists; `GET …/goods` returns 18 reference prices |
| T10.07 | gate P5-1 numbers printed in the PR body; `len(eco.firm_registry) == N` |
| T10.08 | `run_irf('monetary', 0.01, config_dir=…, overrides={'dynamics.feedbacks.wealth_consumption': True})` finite, sign-correct, and different from the all-off path |
| T10.09 | after `POST …/agents` with capital 100: `world_ledger(world).position('AGENT:<id>', 'DEP') == 100`, GOVT `DEP` fell by 100, SFC holds |
| T10.10 | `tests/golden/data/api_schema_v1.2.json` committed; gym reward non-zero after a held position and a price move |
| T10.11 | `POST …/save` then `POST /v1/worlds/load` → same `state_hash`; a WS client receives one `tick` event per step |
| T10.12 | IPO through REST: cap table shows the operator at 100 % then dilutes after `…/issue`; ledger consistent |
| T10.13 | a marketable limit buy fills next tick; `observe(agent)['fills']` non-empty; `assert_consistent(world_ledger(world))` passes |
| T10.14 | a submitted price cut lowers that firm's share next month; `observe(agent)['reports']` non-empty |
| T10.15 | bench prints `n_regions_live=3`; R=1 parity golden still 1e-12 |
| T10.16 | `record_run` on a seeded world leaves `state_hash` unchanged; a 2520-tick recording reproduces `aggregate_baseline.npz` to 0.0 on every shared series |
| T10.17 | `scripts/plot_run.py --run … --golden …` writes an HTML page with no `http` reference; two renders are byte-identical |

---

## 10.5 Worker prompt template

```
You are implementing ONE task card of the MarketSim engine in an isolated worktree. Read `AGENTS.md`
first, then `claude/plan/00-MASTER-PLAN.md` §6–§8, then your card in
`claude/plan/09-phase10-integration.md`.

Follow AGENTS.md exactly: tests first, no magic numbers (config + pydantic), numpy over (R, S), randomness
only via `core.rng.stream`, state via `to_state/from_state`, never loosen a tolerance or rewrite a test to
get green, never change `config/sectors.yaml` / `config/edges.yaml` values without an approved ADR, never
expose hidden state through `observe()`.

Overall goal: <user instruction>.
Your unit: <card id, title, Files, Build, Tests, Out of scope — verbatim from §10.6>.
Shared contracts: <verbatim §10.3>.
Process rules: <verbatim §10.2 "Rules every worker follows">.
E2E recipe: <verbatim §10.4> plus your unit-specific observable.
Commit message: `T10.xx: <title>`, ending with the attribution trailer.

After you finish implementing the change:
1. **Code review** — invoke the `Skill` tool with `skill: "code-review"` to find correctness bugs (it
   reports findings; it does not edit code). Fix any findings it surfaces before continuing.
2. **Run unit tests** — `make check`, or the recipe's pytest line if you need the deselects. If tests fail,
   fix them.
3. **Test end-to-end** — follow the e2e recipe above.
4. **Commit and push** — commit all changes with a clear message, push the branch, and open a PR with
   `gh pr create`. Use a descriptive title. If `gh` is not available or the push fails, note it in your
   final message.
5. **Report** — end with a single line: `PR: <url>`. If no PR was created, end with
   `PR: none — <reason>`.
```

---

## 10.6 Task cards

Sizes: **S** ≤ ~150 LOC · **M** ≤ ~400 · **L** ≤ ~800, tests included.

### T10.00 — Prep: test path, lookup helper, registry skeleton, hygiene
**Depends:** — · **Size:** M · **Files:** `pyproject.toml`, new `src/marketsim/core/lookup.py`, new
`src/marketsim/assembly.py`, `src/marketsim/events/releases.py`, `src/marketsim/api/replay.py` (docstring),
`.gitignore`, `scratch_tmp/` (delete), `.claude/worktrees/wf_168f11b0-163-3` (gitlink), `claude/plan/PROGRESS.md`,
`claude/plan/00-MASTER-PLAN.md` (§8, §9), new `tests/unit/core/test_lookup.py`, new
`tests/unit/events/test_publish_merge.py`
**Read first:** §10.2, §10.3. **Build:**
1. `[tool.pytest.ini_options] pythonpath = ["src", "scripts", "."]`. `tests/validation/test_financials_mtm.py:7`
   imports `tests.validation.test_pricing_vs_betas`, which needs the repo root on `sys.path`. **Version-dependent:**
   under pytest 9.1.1 this collects fine (verified — the module yields 3 tests and the suite runs clean), so
   the fix is hardening, not a live outage; under pytest 8's `prepend` import mode the same line raises a
   collection error, and pytest interrupts the whole session on one. Add `"."` so the import is correct under
   both, without an `__init__.py`. Do not claim in the PR that it fixes a current break unless you reproduce one.
2. `core/lookup.py`: `find_module(world, name) -> Module | None` (single pass over `world.modules`, first
   match by `name`, no exception) and `world_ledger(world) -> Ledger` — the live `RealEconomy.ledger` when an
   economy is attached (resolved **on every call**, because `RealEconomy.reset` replaces it at
   `real/economy.py:697`), otherwise one `Ledger.empty()` created lazily and cached on the `World`
   (a private attribute, not in `to_state`).
3. `assembly.py` skeleton: `CANONICAL_ORDER: tuple[str, ...]` = `("events", "real_economy", "data_releases",
   "valuation", "market", "firm_agent")` and `MODULE_FACTORIES: dict[str, Callable[..., Module]]` with **one
   line per name**, each pointing at `_not_shipped("<name>")` which raises
   `ConfigError("<name> module is not shipped yet")`. Every later card replaces exactly its own line, so the
   twelve PRs do not collide. `build_modules` itself is T10.03.
4. `events/releases.py:246`: `ctx.world._observations[PUBLIC_OBS_KEY] = {"releases": vis}` →
   `ctx.world._observations.setdefault(PUBLIC_OBS_KEY, {}).update({"releases": vis})`. As written it wipes the
   `news` the events module published at `events/scheduler.py:179-180` whenever both are attached.
5. Hygiene: add `.claude/` and `scratch_tmp/` to `.gitignore`; `git rm -r --cached
   .claude/worktrees/wf_168f11b0-163-3` (a committed gitlink, mode 160000); `git rm -r scratch_tmp/` (four
   throwaway repro scripts). Reword the `api/replay.py:1-6` docstring, which documents four routes that do not
   exist, to say they arrive with T10.11.
6. Plan bookkeeping: a `## Phase 10 — Integration (16 tasks) — 09-phase10-integration.md` section in
   `PROGRESS.md` with one unticked row per card (id · title · size · deps) plus an unticked `**GATE P10**`;
   in `00-MASTER-PLAN.md` add Phase 10 to the §9 phase map and table, and pre-register this batch's new config
   keys in the §8 registry (`world.yaml`: `check_sfc`, the read `modules` list; `dynamics.yaml`: the
   `feedbacks` block) so no worker has to edit the master plan.

**Tests:** `tests/unit/core/test_lookup.py` — `find_module` hit and miss; `world_ledger` on a world with an
economy is that economy's ledger and follows it across `reset()`; without an economy it returns the same empty
ledger on repeated calls and is not in `to_state()`. `tests/unit/events/test_publish_merge.py` — a world with
both `events` and `data_releases` publishes `news` **and** `releases` in the same tick, in either module order.
`pytest -m "not slow" -n auto` collects with zero errors.
**Out of scope:** any factory body, `World.create` changes, CI / Makefile (T10.01), the three red tests
(T10.02).

### T10.01 — CI, suite budget, CLI and docs
**Depends:** T10.00 · **Size:** M · **Files:** `Makefile`, `.github/workflows/ci.yml`, `tests/conftest.py`,
`pyproject.toml` (`[project.scripts]`, `[tool.ruff]`), new `src/marketsim/cli.py`,
`src/marketsim/real/policy/monetary.py` (two messages), new `docs/architecture.md`, `README.md`, `AGENTS.md`,
`claude/plan/README.md`, new `tests/unit/scripts/test_cli.py`
**Read first:** master plan §6–§7, §10.3. **Build:**
1. `make check` and CI run `pytest -m "not slow" -n auto` (`pytest-xdist>=3.6` is already a dev dependency and
   unused); CI gains `timeout-minutes: 30`; `ruff` covers `examples/` (`ruff check src tests scripts examples`
   and `[tool.ruff] src = [..., "examples"]`).
2. `tests/conftest.py:15-20`: generate `config/io_table.json` once in `pytest_sessionstart`, controller only
   (`if not hasattr(session.config, "workerinput")`), instead of inside a session fixture — under `-n auto`
   every worker would race on the write. The file is tracked today, so this is belt-and-braces, not a fix.
3. `real/policy/monetary.py:60,64`: make the messages truthful — the QE/QT desk **does** exist
   (`real/policy/cb_operations.py`, T6.28) and `DeferredQE` is the live `MonetaryFramework.qe` field, so the
   text should name the desk and say the wiring is an open question. **No behaviour change.**
4. `marketsim-run` console script → `src/marketsim/cli.py` (`--config --seed --ticks --save --modules`):
   builds through `World.create`, prints one macro line per month-end and the final `state_hash()`. Pass
   explicit modules until T10.03 lands, then `modules=None`.
5. `docs/architecture.md`: the tick pipeline (§7), the module registry and canonical order, the §10.3
   contracts, the one-ledger seam, where state lives. Refresh `README.md` (layout, Python ≥ 3.11,
   `make check`, the CLI). Reconcile the stale nested-layout text in `AGENTS.md:3-4` and
   `claude/plan/README.md:1-2` with the real layout (the checkout root **is** the package root —
   `.github/workflows/ci.yml:1-2` already records this).
6. Report the measured `pytest -m "not slow" -n auto` wall time in the PR body; target < 2 min.

**Tests:** `tests/unit/scripts/test_cli.py` — `main(["--ticks", "63", "--save", tmp])` returns 0, prints three
monthly lines, and the printed hash equals `World.load(tmp, "config").state_hash()`; `ruff` clean over
`examples/`.
**Out of scope:** the three red tests; module factories; touching `monetary.py` behaviour.

### T10.02 — Bisect and resolve the three red validation tests
**Depends:** T10.00 · **Size:** M · **Files:** `tests/golden/data/` (only a golden the bisect justifies),
`src/marketsim/real/*` (only what the bisect implicates), `claude/plan/DECISIONS.md` (proposed ADR only),
`claude/plan/reports/*.md` (the matching report)
**Read first:** §10.2 rule 4, AGENTS.md "Never", `00-MASTER-PLAN.md` §13. **Build:** for each of
`test_r1_demand_irf_matches_phase2_golden`, `test_credit_crunch_gate_and_recovery` and
`test_vat_step_and_no_permanent_inflation`, bisect from the golden anchor `26254b2`
(*T3.02: Regionalise live state (R=1 parity)*) to `c7f9e95` running **only** that test
(`git bisect run .venv/bin/pytest -x -p no:cacheprovider <nodeid>`). Prime suspects from the log:
`ded52b8` (T3.14 NPC entry/exit — `dynamics.entry_exit.enabled` defaults to `True` and fires under shocks,
`real/economy.py:309-315`, `:323-324`), `2c36ebf` (T3.13 want shifters with AR decay on the ShockBus),
`a4b7640` (T2.33 committee information set from published vintages). Then, per test:
- **regression** → fix the code; the golden stays untouched;
- **intended dynamics change** → regenerate that one golden, with a proposed ADR (next free number:
  **ADR-018**) in `DECISIONS.md` and an updated line in the matching report under `claude/plan/reports/`;
- **needs an economics decision** → leave it red and put the full question text in the PR body under
  `## QUESTIONS`, with your best two options.

**Never change a tolerance** (`TOL` in `tests/golden/test_r1_parity.py`, the bounds at
`tests/validation/test_credit_crunch.py:104-108`, the VAT bounds in `test_policy_levers.py`) and never rewrite
a test to pass.
**Tests:** the three named tests run **without** `--deselect`, plus any regression test a fix needs.
**Out of scope:** the documented `xfail`s, the cost-push magnitude, any other validation bound, and every file
outside the list above.

### T10.03 — Module registry and default World assembly
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/assembly.py`, `src/marketsim/world.py`
(`:26-48`, `:85-120`), `src/marketsim/core/config.py` (`WorldSettings`, `:219-226`), `config/world.yaml`,
`src/marketsim/api/sessions.py` (`:127-152`, `:373-428`), `src/marketsim/sdk/gym_env.py` (`:272`),
`src/marketsim/sdk/pz_env.py`, `src/marketsim/sdk/vector_env.py`, `src/marketsim/sdk/curriculum.py` (`:333`),
`src/marketsim/scenarios/runner.py` (`:42`), `scripts/run_world.py`, `examples/train_smoke.py` (docstring),
the four scanner sites (`events/scheduler.py:255-259`, `events/releases.py:275-279`, `api/rest.py:698-702`,
`sdk/policy_env.py:153-158`), new `tests/integration/test_assembled_world.py`,
`tests/integration/test_world_skeleton.py`, `tests/validation/test_intervention_consistency.py`
**Read first:** §10.3, master plan §7. **Build:**
1. Fill the `events`, `real_economy` and `data_releases` factory lines. Bodies come from
   `Events.from_config` (`events/scheduler.py:52-63`), `make_real_world` (`real/economy.py:853-862`) and
   `attach_releases` (`events/releases.py:282-290`) — but as constructors returning a module, not as
   appenders to a live world.
2. `build_modules(cfg, names=None)`: `names` defaults to `cfg.world.modules`; an unknown name raises
   `ConfigError` naming the valid set; the result is always sorted into `CANONICAL_ORDER`; `load_io` and
   `load_catalog` are cached per resolved config path so N worlds do not re-read the IO table and the event
   catalog.
3. `World.__init__`: `list(modules) if modules is not None else []` — stop conflating `None` and `[]`
   (`world.py:31`). `World.create(..., modules=None)` → `build_modules(cfg)`; `modules=[]` → an empty world.
4. `World.to_state()` records `"module_names"`; `World.load(path, config_dir, modules=None)` rebuilds from
   those names before `load_state`; `load_state` raises `StateError` naming a saved module that is not
   attached instead of a bare `KeyError` (`world.py:99-101`).
5. New config key `WorldSettings.check_sfc: bool = True`, passed by the `real_economy` factory to
   `RealEconomy(..., check_sfc=cfg.world.check_sfc)`. **The default stays `True`** — AGENTS.md rule 5 and the
   definition of done require the SFC assertion on for anything that steps the world; throughput runs turn it
   off with `overrides={"world.check_sfc": False}` (what `scripts/bench.py:166` does today by passing the
   flag directly).
6. `config/world.yaml`: `modules: [events, real_economy, data_releases]`. The coordinator appends
   `valuation, market` in the wiring PR once T10.04 / T10.05 have merged.
7. SDK and scenario builders take `[*build_modules(cfg), FirmAgentModule()]` (`gym_env.py:272`, `pz_env.py`,
   `vector_env.py`, `curriculum.py:333`). `scenarios/runner.py:42` keeps its own `Events` instance (it patches
   the catalog with `_zero_hazards`): build the rest with
   `[ev, *(m for m in build_modules(cfg) if m.name != "events"), *(extra_modules or [])]` and re-sort
   canonically — never two `events` modules. Update `scripts/run_world.py:13` and the
   `examples/train_smoke.py` docstring.
8. Replace the four inline `real_economy` scans with `find_module`. `attach_policy_desk`
   (`api/sessions.py:106-112`) keeps its `isinstance(PolicyDesk)` fallback but tries
   `find_module(world, "real_economy")` first.
9. Fallout: every `World.create(config_dir)` without `modules` now builds a real economy. Run
   `tests/integration/api/test_policy_api.py` and `tests/integration/api/test_explain.py` first, then the
   whole suite. Tests that genuinely need an empty world pass `modules=[]`
   (`tests/integration/test_world_skeleton.py:6-10`, `tests/validation/test_intervention_consistency.py:61-81`)
   — never weaken an assertion to accommodate the new default.

**Tests:** new `tests/integration/test_assembled_world.py` — three simulated years (756 ticks) on the
assembled world is hash-reproducible across two builds; the SFC assertion is active (it runs monthly);
`save` → `load` with **no** `modules` argument reproduces the hash and the next-step hash; `observe("x")`
carries `news` and `releases`; `build_modules` rejects an unknown name; module order is canonical whatever the
YAML order.
**Out of scope:** the `valuation` / `market` factory lines (T10.04 / T10.05); publishing `macro` / `goods` /
`labour` (T10.06); flipping `world.yaml` to the full five (coordinator).

### T10.04 — Market module core
**Depends:** T10.00 · **Size:** L · **Files:** new `src/marketsim/market/module.py`,
`src/marketsim/assembly.py` (the `market` line), `scripts/bench.py` (this card owns the whole file),
new `tests/market/test_market_module.py`
**Read first:** `06-phase6-pricing-markets.md` §6.2, §6.6–§6.8, §6.10–§6.12; §10.3. **Build:**
1. `MarketModule(name="market")`: the book from `InstrumentRegistry.from_markets(cfg.markets, codes)`
   (`market/instruments.py:407`); `EngineMM` venues (`market/mm.py:183`, `add_book` / `set_mark` / `ingest`)
   for `EQ:NPC:*`, commodities and `RE:<REGION>`; `OrderBook.from_clob_cfg` (`market/clob.py:262`) for
   `EQ:FIRM:*`; `BackgroundFlow` (`market/background.py:57`), `ImpactKernel` (`market/impact.py:199`),
   `ClobLiquidity` thin quotes (`market/clob_liquidity.py:61`), `Mispricing` (`pricing/mispricing.py:154`)
   with the news impulse from `find_module(world, "events")`.
2. Bond buckets are **quoted** from `fair_values()` (`GB_BILL` / `GB_NOTE` / `GB_BOND` / `CORP_POOL`). Do not
   build a second bond venue — `BondDesk` (`api/rest.py:363`) and `market/bond_market.py` stay as they are.
3. On `Phase.MARKET`: background flow → matching → impact → publish `quotes` and `markets` per §10.3.
   `Phase.VALUE` belongs to valuation; draining agent orders at `Phase.INGEST` is T10.13.
4. `fair_values()` comes from `find_module(world, "valuation")`, with a documented fallback to the baseline
   marks (`StubAssetPriceProvider.from_baseline`, `config/markets.yaml`) when valuation is absent.
5. `to_state()` / `from_state()` cover venues, flow, kernel, mispricing and marks; the streams `flow`,
   `noise` and `sentiment` are touched in `reset` in a fixed order.
6. Replace the toy `MarketBookModule` in `scripts/bench.py` (`:100-153`) with the library module, keeping
   `make_small_world` (`:156-167`), `small_world_bench` (`:170-198`), `scripted_agents_bench` and the
   `--gate3` report working with unchanged printed keys.

**Tests:** determinism (same seed → same hash over 200 ticks; state round trip); the small-world bench holds
≥ 1,000 ticks/s with the measured number in the PR body; `fair_values` fallback with no valuation module;
`observe("x")["quotes"]` has 25 symbols and the published `quotes` / `markets` validate as `Observation`.
**Out of scope:** accepting agent orders, settlement and per-agent views (T10.13); a second bond venue; new
wire models.

### T10.05 — Valuation module
**Depends:** T10.00 · **Size:** M · **Files:** new `src/marketsim/pricing/module.py`,
`src/marketsim/assembly.py` (the `valuation` line), new `tests/unit/pricing/test_valuation_module.py`
**Read first:** `06-phase6-pricing-markets.md` §6.3–§6.5, §6.11; `config/betas.md`. **Build:**
`ValuationModule(name="valuation")` on `Phase.VALUE`. With an economy (`find_module`), read **published**
state only — `eco.pub` (`Published.published`, `real/aggregates.py:53-67`), `eco.cb.r`, `eco.cb.pi_e`,
`z_risk = eco.sh["risk"]` — and run `EarningsExpectations.update_from_vintages`
(`pricing/expectations.py:150`) → `TermPremium.step` (`pricing/curve.py:80`) → `y10` (`:50`) →
`delta_rho` (`pricing/discount.py:30`) → `fundamental_values` (`pricing/fundamentals.py:30`);
`bucket_fair_yield` (`curve.py:141`) → `bucket_price` (`pricing/bond_buckets.py:92`) per bucket;
`commodity_prices` (`pricing/commodities.py:93`) on `eco.p`; `regional_prices`
(`pricing/realestate.py:88`) for `RE:<REGION>`. Without an economy, baseline constants. Expose
`fair_values() -> dict[str, float]`; publish **nothing** public; `to_state` / `from_state`; no RNG.
**Tests:** at steady state every fair value equals baseline to 1e-9 (bond `P = 1.0`, `EQ:NPC:*` = `V0`);
deterministic across two builds; under a scripted +100 bp path REALESTATE, UTILITIES and TELECOM fall most and
BANKS is the only riser, matching the `beta_rate` column of `config/betas.md` (−8.03, −6.77, −5.91 vs +3.08);
REALESTATE falls more than STAPLES.
**Out of scope:** publishing anything on `observe()`; the market wiring (T10.04 reads `fair_values`).

### T10.06 — Economy as a World citizen
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/real/economy.py` (`:680-682` `on_phase`,
`:695-698` `reset`, `to_state` / `from_state`), `src/marketsim/api/rest.py` (`:1077-1103`),
new `tests/unit/real/test_economy_publish.py`, `tests/integration/api/test_rest.py`
**Read first:** §10.3, `sdk/policy_env.py:169-185` (`_macro` is the template). **Build:**
1. At `Phase.PUBLISH`, **every tick**, publish `macro`, `prices`, `goods`, `labour` and `regions` in the
   §10.3 shapes, from published or last-closed state only (`Published.published(series, lag)`;
   `self.last_agg`; `self.p`). `on_phase` today handles only `Phase.REAL` at month end
   (`real/economy.py:680-682`), and `_observations` is cleared every tick, so the publish must be
   unconditional. An R=1 economy broadcasts the national labour row to every configured region code.
2. `reset(ctx)` re-seeds the monetary rule from the **World** seed:
   `MonetaryRule.from_config(cfg, r0=self.cb.r, seed=<derived from ctx.world.rng.stream("monetary_rule")>)`,
   touching that stream so it is seed-stable. Today it is seeded from `cfg.world.seed`
   (`real/economy.py:161`), so two worlds with different `World` seeds share one committee / noise path.
   **Keep the constructor as it is** — `RealEconomy(...)` built directly (`scenarios/irf.py`, the goldens)
   must stay bit-identical; only `reset(ctx)` re-seeds.
3. `GET …/goods` and `GET …/labour/{region}` return the live views instead of empty models
   (`api/rest.py:1088`, `:1103`), through `find_module`.

**Tests:** new `tests/unit/real/test_economy_publish.py` asserts on `world.observe("x")` directly — `macro`
has `gdp/cpi/core_cpi/u/infl/policy_rate` after 21 ticks, `prices` has 18 sector codes, `goods.reference_prices`
has 18 rows, `labour` has one row per configured region code, and the keys are re-emitted on the next tick;
publishing is read-only, so `tests/golden/test_r1_parity.py` and the IRF goldens are unchanged; the REST goods
and labour routes return live numbers.
**Out of scope:** adding `macro` to `OBSERVATION_FIELDS` (T10.10); firm views and reports (T10.14).

### T10.07 — Hybrid firms core
**Depends:** T10.00 · **Size:** L · **Files:** new `src/marketsim/firms/hybrid.py`,
`src/marketsim/real/economy.py` (`:214-217`, one guarded call after `:325`, one guarded call after `:656`),
`tests/validation/test_hybrid_vs_aggregate.py` (extend)
**Read first:** `05-phase5-firms.md` §5.2–§5.6; `QUESTIONS.md` "T5.16 — hybrid World stepper vs cell-level
identity". **Build:** `HybridBlock(cfg, registry, book, ledger)` with two entry points:
- `pre_month(economy, k_eff)` — advance construction (`firms/founding.py:203`) and plants
  (`firms/plants.py:74`), autopilot plan / price / starts per firm (`firms/autopilot.py:22`), labour matching
  on `lf − Σ n` (`firms/labour_market.py:23`); returns the firm contributions to `k_eff`, employment,
  inventories, backlog and posted prices;
- `post_month(economy, supplied, dshare, rev, wages, ebitda, interest, ctax, tick)` — goods-market
  de-aggregation through `firms/goods_market.py:106 allocate`, procurement
  (`firms/procurement.py:30`), financing (`firms/financing.py`), `FirmBooks.close_month`
  (`firms/accounts.py:202`) on the **shared** ledger, then `assert_consistent`.

Two guarded call sites in `economy.py` (`if self.hybrid is not None:`) — one where `k_eff` is formed
(`:325`) and one after settlement (`:652-656`) — so an economy with no firms is bitwise unchanged.
`attach_firms` (`:214-217`) keeps its signature and builds the block.
**Tests:** with zero firms, `tests/golden/data/r1_phase2.npz` and `tests/golden/data/aggregate_baseline.npz`
match bitwise; with N ∈ {1, 5} equal autopilot firms per cell, 120-month GDP within 1 %, sector outputs within
2 % and CPI within 0.5 % of `aggregate_baseline.npz` (gate P5-1 — mark `slow` if it exceeds the suite budget,
and print the deviations in the PR body); SFC green every month; `len(eco.firm_registry) == N`.
**Out of scope:** routing `FirmDecision`s, bankruptcy and reports (T10.14); REST founding (T10.09).

### T10.08 — Feedback edges on the World (closes T6.19)
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/core/config.py` (new `FeedbacksCfg`,
`DynamicsConfig` field), `src/marketsim/real/economy.py` (`:268`, `:276-302`, `:404-416`, state),
`src/marketsim/real/labour.py` (`:75-97`), `config/dynamics.yaml`,
new `tests/validation/test_feedback_stability.py`, new `claude/plan/reports/feedback-stability.md`
**Read first:** `06-phase6-pricing-markets.md` §6.9; `src/marketsim/real/feedbacks.py` (the kernel is already
complete and tested: `FeedbackState`, `capex_q_term`, `derp_to_cc_gap`, `smooth_wealth`,
`wealth_consumption`, `icr_hire_scale`, `scale_employment_target`); `QUESTIONS.md` "T6.19". **Build:**
1. `FeedbacksCfg` with **every switch off by default** — `capex_q: bool = False`,
   `wealth_consumption: bool = False`, `icr_hiring: bool = False` plus the map parameters — on
   `DynamicsConfig`, following the `EntryExitCfg` precedent (`core/config.py:482-497`, `:515`).
2. Three guarded edits in `economy.py`: (a) the `cost_of_capital_gap` / `start_rate` block (`:276-302`) adds
   `derp_to_cc_gap` and `capex_q_term` from the live asset-price provider; (b) `consumption_nominal` (`:268`)
   adds `wealth_consumption(state.step_wealth(mark))` on the live equity mark; (c) `step_labour`
   (`:404-416`) passes an ICR scale — `real/labour.py:75-97` gains an optional `icr_scale` keyword applied to
   `n*` through `scale_employment_target`, with the ICR computed as at `:453`.
3. `FeedbackState` joins `to_state` / `from_state`. Append a documented all-off `feedbacks:` block to
   `config/dynamics.yaml`.

**Tests:** with every switch off, the r1 parity golden and the IRF suite are bitwise unchanged (this is the
first assertion to write); per switch,
`run_irf("monetary", 0.01, config_dir=…, overrides={"dynamics.feedbacks.<switch>": True})` is finite,
sign-correct and measurably different from the all-off path; one combined run stays bounded over 240 months
(mark `slow` if needed); `reports/feedback-stability.md` records the stable gain ranges and what diverges
beyond them (AGENTS.md rule 15).
**Out of scope:** turning any switch on by default — that needs an ADR and the human.

### T10.09 — Shared ledger and firm founding through the API
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/api/sessions.py` (`:83-94`, `:127-152`,
`:154-199`, `:276-278`, `:344-371`, `:373-428`), `src/marketsim/api/rest.py` (`:556-570`, `:986-1075`),
new `tests/integration/api/test_shared_ledger.py`
**Read first:** §10.3 (Ledger), `05-phase5-firms.md` §5.2, `firms/accounts.py`, `firms/founding.py`.
**Build:**
1. `_ManagedWorld.ledger` becomes a **property** resolving `world_ledger(self.world)` at use; the
   `Ledger.empty()` fallback (`sessions.py:143`) moves to a private field created lazily and used only when no
   economy is attached. `WorldManager.ledger(wid)` (`:276-278`) follows. Serialise **only** the fallback in
   `to_state` (`:366`) — the economy's ledger is already inside
   `world.to_state()["modules"]["real_economy"]` — and restore it in `from_state` (`:406`) only when there is
   no economy, so a round trip neither doubles the ledger nor resurrects a stale copy.
2. **Agent starting capital is a government grant** (decision taken): with an economy attached, post
   `GOVT.DEP −= capital` and `AGENT:<id>.DEP += capital` on the shared ledger with the **registered**
   `capital_transfer` flow tag (`config/ledger.yaml:49`, already used at `sessions.py:33`) — do not invent a
   tag. The grant then reaches the debt path by itself: `cover_govt_shortfall`
   (`real/settlement.py:282`) funds the resulting GOVT deposit shortfall with bond issuance at the next
   monthly close. `assert_sector_balances` (`real/settlement.py:329-349`) is unaffected — the posting is
   balanced and falls outside the settlement window — but assert it anyway in the test. `BANKSYS` stays the
   fallback with no economy (today's behaviour, `sessions.py:179-191`); `assert_consistent` after every
   posting (already at `:192`).
3. `POST …/firms` (`rest.py:989-1003`) calls `found_firm` (`firms/founding.py:91`) plus `register_firm`
   (`firms/accounts.py:41`) and `post_founding_capital` (`:51`) on the shared ledger, creates `FirmBooks`
   (`:202`) on the `_WorldRest` record, and registers the firm with the economy through
   `RealEconomy.attach_firms(registry, book)` using a `FirmRegistry` held on the managed world.
4. `GET …/firms/{fid}/state|financials` return real `balance_sheet()` / `income_statement()` numbers
   (`rest.py:1042`, `:1056`); `…/reports` stays empty until T10.14.

**Tests:** after `POST …/agents` with capital 100 — `world_ledger(world).position("AGENT:<id>", "DEP") == 100`,
GOVT `DEP` down 100, SFC green; founding through REST puts `FIRM:<id>` on the shared ledger with its founding
capital, the economy's registry contains it, and `GET …/state` shows non-empty books; a `save` → `load`
round trip through `WorldManager` keeps one ledger and the same positions. Existing REST tests stay green —
in particular the bond-desk postings at `rest.py:810` now land on the shared ledger, so `assert_consistent`
must hold after `POST …/step` on a world with bond activity. If the two books turn out to be economically
incompatible, **stop** and put it in the PR body under `## QUESTIONS` rather than papering over it.
**Out of scope:** new wire models; firm decisions and reports (T10.14); equity routes (T10.12).

### T10.10 — Observation schema v1.2 and ledger-backed rewards
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/api/schemas.py` (`:23`, `:27-47`, `:759-778`),
`src/marketsim/scenarios/randomise.py` (`OBSERVE_PUBLIC_KEYS`), `src/marketsim/sdk/local.py`,
`src/marketsim/api/rest.py` (`:683-691`), new `src/marketsim/sdk/portfolio.py`,
`src/marketsim/sdk/gym_env.py` (`:103-170`, `:519-553`), `src/marketsim/sdk/pz_env.py`,
new `tests/golden/data/api_schema_v1.2.json`, `tests/unit/api/test_schemas.py`, `docs/api.md`
**Read first:** §10.3, `07-phase7-api-sdk.md` §7.1–§7.2. **Build:**
1. Add a `MacroView` model (`gdp`, `cpi`, `core_cpi`, `u`, `infl`, `policy_rate`; units in the docstring) and
   an `Observation.macro` field; add `"macro"` to `OBSERVATION_FIELDS` (`schemas.py:27-47`), to
   `OBSERVE_PUBLIC_KEYS` (`scenarios/randomise.py:34-54`) and to the `LocalClient` whitelist. Bump
   `API_SCHEMA_VERSION` to `"v1.2"` and commit a **new** golden
   (`git add -f tests/golden/data/api_schema_v1.2.json`); leave the `v1` and `v1.1` snapshots untouched
   (`tests/unit/api/test_schemas.py:221-229`).
2. New `sdk/portfolio.py`: `net_worth(ledger, agent_id, prices) -> float` — cash plus Σ position × mark in cr,
   using `ledger.position` and the `net_financial_assets` convention (`ledger/sfc.py:16`), with real assets
   excluded and units documented.
3. Gym / PettingZoo rewards use it when the observation carries no `portfolio`
   (`sdk/gym_env.py:103-117`, `:519-544`); `is_bankrupt` (`:159-170`) also fires on non-positive ledger net
   worth; dividends come from the ledger's `dividends` tag rather than a missing observation field.

**Tests:** the golden schema test passes at v1.2; a held position plus a price move flips the reward sign; an
agent with no ledger row still gets a finite reward and is not spuriously bankrupt; `Observation` accepts a
published `macro` and still rejects an unknown key; the `LocalClient` / REST observation surfaces stay
identical to each other.
**Out of scope:** publishing `macro` (T10.06 does that); any other schema change.

### T10.11 — World lifecycle routes and WebSocket wiring
**Depends:** T10.00 · **Size:** M/L · **Files:** `src/marketsim/api/rest.py` (new `lifecycle_router`,
`RestState` `:573-579`, `:799-811`, `:1440-1461`), `src/marketsim/api/replay.py` (docstring),
`src/marketsim/api/ws.py` (`:314-330`), `src/marketsim/sdk/client.py`,
new `tests/integration/api/test_lifecycle.py`, `docs/api.md`
**Read first:** `07-phase7-api-sdk.md` §7.2, `src/marketsim/api/replay.py`, `src/marketsim/storage/`.
**Build:**
1. `POST …/save`, `POST /v1/worlds/load`, `GET /v1/worlds`, `GET …/replay`, `POST …/export` as thin wrappers
   over `api/replay.py` (`write_snapshot`, `load_snapshot`, `read_replay_log`, `write_replay_log`,
   `write_export_npz`) and `storage/` (`FileStore`, `WorldRegistry`), rooted at a storage path carried on
   `RestState` (a `create_app` argument, **not** a new wire model).
2. `_WorldRest` is serialised **from `rest.py`**: `RestState.to_state()` wraps `manager.to_state()` and adds
   the per-world REST records. Do not touch `WorldManager.to_state` — serialising `_WorldRest` from
   `sessions.py` would be circular.
3. `create_app` calls `include_ws(app, manager=…)` (`api/ws.py:314`); it never has, so the WS route is
   currently unreachable in the assembled app. `POST …/step` (`:799-811`) publishes one `tick` event, the
   public news, and each agent's own fills to the `StreamHub`.
4. SDK client methods mirroring the routes; `docs/api.md` rows; note in the docs that these are REST-only, so
   the `LOCAL_METHODS` contract (`sdk/local.py:21-32`) is unchanged. Restore the `api/replay.py:1-6` docstring
   to the present tense now that the routes exist.

**Tests:** `save` → `load` gives an identical `state_hash` and an identical next-step hash; `GET /v1/worlds`
lists the ids; `GET …/replay` returns the log; `POST …/export` writes an `.npz` with the T7.10 keys (to a
tmp path, never into `tests/golden/data/`); a WS client receives exactly one tick event per step and one news
item when an event fires; resume with `since` has no gaps or duplicates (T7.05 behaviour preserved).
**Out of scope:** new wire models; equity routes (T10.12); order plumbing (T10.13).

### T10.12 — Equity and financing routes
**Depends:** T10.00 · **Size:** M · **Files:** `src/marketsim/api/rest.py` (new `equity_router`; `_WorldRest`
gains fields only), `src/marketsim/sdk/client.py`, new `tests/integration/api/test_equity_routes.py`,
`docs/api.md`
**Read first:** `06-phase6-pricing-markets.md` §6.10 (listing / cap table / control), `equity/`. **Build:**
`POST …/firms/{fid}/listing` (IPO call auction through `equity/listing.py:207 list_firm`; with no market
module attached, a `CLOB.from_clob_cfg(cfg.markets.clob)` held on the `_WorldRest` record),
`POST …/firms/{fid}/issue`, `…/buyback`, `…/dividends` (`equity/corporate_actions.py:64`, `:104`, `:141`),
`GET …/firms/{fid}/captable` (`equity/captable.py:53`), `POST …/firms/{fid}/plants`
(`firms/plants.py:39 start_expansion`), `POST …/firms/{fid}/financing` (`firms/financing.py`). Every posting
goes on `world_ledger(world)`. Reuse `ListingRequest` (`schemas.py:437`) and `CorporateAction` (`:465`);
responses use models defined in `rest.py` or plain dicts.
**Tests:** Σ cap-table longs `== shares_outstanding` after every action; the ledger is consistent after each;
the operator holds 100 % after the IPO and dilutes after `…/issue`; 403 for a non-operator, 404 for an unknown
firm.
**Out of scope:** new wire models in `api/schemas.py`; secondary-market matching (T10.04 / T10.13).

### T10.13 — Order seam, settlement and per-agent views
**Depends:** T10.04 (merged) · **Size:** M · **Files:** `src/marketsim/market/module.py`,
`src/marketsim/api/sessions.py` (`:220-234`), `src/marketsim/api/rest.py` (`:884-983`),
new `tests/market/test_order_seam.py`, `tests/validation/test_intervention_consistency.py`
**Read first:** §10.3 (Orders), master plan §6 ("next-tick fills"), `market/settlement.py`. **Build:**
1. `MarketModule.accept(agent_id, OrderRequest) -> OrderAck` allocates a **world-level** order id and parks
   the order (venue ids are per-book: `market/clob.py:253`, `market/mm.py:194`). At `Phase.INGEST` parked
   orders drain into their venues, keeping `{oid: (venue, symbol, venue_order_id, agent_id)}`; `cancel(oid)`
   works before and after draining.
2. Fills settle with `settle_fill` (`market/settlement.py:47`) on `world_ledger(world)`; NPC dividends are
   paid monthly from the economy's published payout through `settle_npc_dividends` (`:127`).
3. At `Phase.PUBLISH` each agent gets `fills`, `orders` and `portfolio` in the §10.3 shapes (private key =
   `agent_id`), re-emitted every tick.
4. `WorldManager.submit_order` (`api/sessions.py:220-234`) calls `accept` when a market module exists and
   keeps today's inbox path otherwise; `rec.open_orders` decrements when an order dies. REST order routes
   read the module instead of fabricating `OpenOrder` rows (`api/rest.py:908-919`), keeping the idempotency
   cache behaviour.
5. Update the negative-result docstring at `tests/validation/test_intervention_consistency.py:61-72` and add
   the positive case on a live venue (same seed with and without an agent's order now **must** differ).

**Tests:** a marketable limit buy fills on the next tick and appears in `observe(agent)["fills"]`;
`assert_consistent(world_ledger(world))` holds with an `AGENT:` counterparty; resubmitting the same
`Idempotency-Key` returns the same ack and does not double-fill; a cancel removes the resting order and
decrements `open_orders`; `observe(agent)["orders"]` shows only that agent's orders.
**Out of scope:** margin and forced liquidation (already T6.x); firm decisions (T10.14).

### T10.14 — Firm decisions, bankruptcy and reports in the hybrid step
**Depends:** T10.07 (merged) · **Size:** M · **Files:** `src/marketsim/firms/hybrid.py`,
`src/marketsim/world.py` (`:152-177` `FirmAgentModule`), new `tests/validation/test_hybrid_world.py`
**Read first:** `05-phase5-firms.md` §5.4, §5.6, §5.7. **Build:** `FirmAgentModule` routes **only**
`FirmDecision` payloads (ignoring `OrderRequest`s and raw dicts — §10.3; today it swallows all three at
`world.py:164-171`) into the registry through `validate_decision` (`firms/levers.py:71`), returning the
machine-readable adjustments in that agent's private observation; `should_resolve` / `resolve`
(`firms/bankruptcy.py:32`, `:40`) run inside `post_month`; `ReportDesk.publish_month`
(`firms/reports.py:16`) emits `reports` at `Phase.PUBLISH` with the §5.7 lags (live for the operator,
lagged for everyone else).
**Tests:** a submitted price cut lowers that firm's share the next month; a strategic default resolves in one
step, SFC-green, and emits `large_bankruptcy` above the threshold; `observe(agent)["reports"]` is non-empty
for the operator and carries no un-lagged books for anyone else; an `OrderRequest` in the inbox is ignored,
not mis-applied.
**Out of scope:** the order seam (T10.13); new lever semantics.

### T10.15 — R=3 regional monthly step
**Depends:** every other Phase-10 card (merged) · **Size:** L · **Files:**
`src/marketsim/real/steady_state.py` (`:106-115`), new `src/marketsim/real/regional_step.py`,
`src/marketsim/real/economy.py` (six spans), `src/marketsim/ledger/opening.py`,
new `tests/golden/data/r3_baseline.npz`, new `tests/validation/test_r3_step.py`
**Read first:** `03-phase3-regions-demand.md` §3.1–§3.7; `QUESTIONS.md` "T3.15 — R = 3 live monthly stepper".
**Build:** `compute_real_baseline(..., geom=None)` takes a geometry so `r_dim` is no longer hardcoded to 1
(`real/steady_state.py:115`); a `RealEconomy(..., regions=False)` switch; `real/regional_step.py` with pure
functions for regional demand, inter-regional trade, regional labour, regional prices and the regional
settlement loop; six `if self.R == 1: <today's code verbatim> else: <regional>` wrappers in `step_month`,
`_open_books` and the aggregates; `ledger/opening.py` opens per-region books. Record the `r3_baseline.npz`
golden (`git add -f`) and a replay test.
**Tests:** the R=1 parity golden is still bitwise / 1e-12 where it is today; R=3 runs 120 months SFC-green
with no NaN; a regional shock propagates and prices converge inside the §3.1 band;
`python scripts/bench.py --small-world` prints `n_regions_live=3`. After merge the coordinator re-verifies the
R=1 parity golden and T10.07's zero-firms-bitwise test.
**Out of scope:** multi-country / FX (ADR-017); regional financial markets beyond the existing
`RE:<REGION>` index.

### T10.16 — Run recorder and the `run.npz` series contract
**Status:** shipped — `src/marketsim/viz/recorder.py`, `tests/unit/viz/test_recorder.py`.
**Depends:** — · **Size:** M · **Files:** `src/marketsim/viz/__init__.py`, `src/marketsim/viz/recorder.py`,
`tests/unit/viz/test_recorder.py`
**Read first:** §10.3 (Observations), `real/aggregates.py:12-32`, `tests/validation/test_regional_shocks.py:139`.
**Build:** a **caller-side sampler, not a `World` module** — a module would enter `World.to_state()` and change
every `state_hash`. After each tick it reads `last_agg` off the economy and appends one row per closed month.
Series names match `tests/golden/data/aggregate_baseline.npz` (`gdp` = `Aggregates.gdp_prod_real`, plus `cpi`,
`u`, `r`, `w`, `x`, `p`) so a recording and a golden load through one reader and overlay key by key; the other
eleven `Aggregates` fields ride along. `write_run` emits `<stem>.npz` (arrays) + `<stem>.json` (metadata, units
per series, published news), following the `layer1/betas.py:273-285` precedent. `read_series` also reads the
committed goldens, with a `prefix` for the `ns_` / `irf_` families in `r1_phase2.npz`.
**Tests:** a recorded run and a plain run of the same seed have equal `state_hash`; 63 ticks yield exactly 3
rows at ticks 21/42/63; `gdp`/`cpi`/`u`/`r`/`x` equal what `step_month()` returns; write → read round-trips;
both committed goldens load; a world with no economy records nothing.
**Out of scope:** rendering (T10.17); recording market or per-agent state — add those series when T10.04 and
T10.13 publish them.

### T10.17 — Self-contained HTML dashboard
**Status:** shipped — `src/marketsim/viz/dashboard.py`, `scripts/plot_run.py`, `docs/viz.md`,
`tests/unit/viz/test_dashboard.py`, `tests/unit/scripts/test_plot_run.py`.
**Depends:** T10.16 · **Size:** M · **Files:** `src/marketsim/viz/dashboard.py`, `scripts/plot_run.py`,
`tests/unit/viz/test_dashboard.py`, `tests/unit/scripts/test_plot_run.py`, `docs/viz.md`
**Read first:** T10.16, `05-phase5-firms.md` §5.1 (the P5-1 tolerances), `scripts/bench.py:415-559` (the
string-built report precedent).
**Build:** `render(rec, golden=None) -> str` returns one HTML page with inline SVG marks and a small embedded
hover script — **no plotting dependency, no CDN**, so AGENTS.md rule 13 and the no-network rule both hold and
the page opens from disk. Output is a pure function of the recording (no timestamps, sorted iteration).
Sections: KPI row, **Macro** (one measure per axis, never two scales), **Run vs golden** (overlay + relative
deviation strip with the gate tolerance shaded — GDP 1 %, sector output 2 %, CPI 0.5 %), **Sector output** and
**Sector prices** as small multiples, **Events** from the recorded news. Every section carries a table view, so
no value is hover-gated. `scripts/plot_run.py` records and renders in one command and takes repeatable
`--override key=value` (the goldens need `dynamics.banks.mode=passthrough`).
**Tests:** two renders are byte-identical; the page contains no `http`/`https` reference and no external
`<script src>`/`<link>`; every SVG coordinate stays inside its `viewBox`; text never carries a series colour;
gridlines are never dashed; the deviation verdict cell reads `within` / `over` against the gate tolerance;
labels are escaped — a series name containing `</script>` must not break out of the embedded JSON block.
**Out of scope:** a live WebSocket UI (needs T10.11 to mount `include_ws`); market and per-agent panels until
T10.04 / T10.13 publish those series; any new runtime dependency.
