# AGENTS.md — marketsim

Instructions for coding agents (Cursor, Claude Code, Codex…). This file lives at the **marketsim package
root** (`marketsim/AGENTS.md`), not the Victoria 3 repo root. The full plan lives in `claude/plan/`.
**Read this file, then the task card you are about to work on.**

## Mission

Build a headless, deterministic **economy + financial-market simulation engine** that is (1) a professional
testing/training environment for trading and firm-operating agents and (2) the backbone of a game — both through
the same API. One country with several regions, 18 sectors on an input-output backbone, five institutional nodes
(HOUSEHOLD, GOVT, CENBANK, ROW, LABOUR), random real-world-style events that sometimes cascade, agents that trade
assets **and** own/operate firms whose shares other agents can buy. The engine prices assets; strategies live outside.
Government and central bank are policy authorities with levers (autopilot, scripted or agent-controlled); government
and pooled corporate bonds trade in a bond market; every firm borrows at one common rate.

## How to work

1. Open `claude/plan/PROGRESS.md`. Take the **lowest-numbered unchecked task whose dependencies are all checked**
   (or the task the human names). One task per session/PR. Do not start a second task.
2. Read, in this order: this file → `claude/plan/00-MASTER-PLAN.md` §6–§7 (conventions, tick pipeline) → the phase
   file's spec sections named in the card's **Read first** → the files listed in **Files**.
3. Write the tests named in the card first, then the code. Stay inside the card's **Files** list; if you must touch
   another file, say why in the PR/commit body.
4. Run `make check` (ruff + `pytest -m "not slow"`). From Phase 2 on, the SFC assertion and the determinism test
   are part of that suite and must stay green.
5. Tick the box in `PROGRESS.md` (`- [x] T2.07 … — 2026-10-02, note`), commit as `T2.07: <title>`.
6. **If the card is ambiguous, contradicts another doc, or a required test cannot pass without changing the
   economics: STOP.** Append the question to `claude/plan/QUESTIONS.md` with the task ID and your best two options.
   Do not invent economics, do not loosen a tolerance, do not delete or rewrite a test to get green.

## Non-negotiable design rules

1. **Two graphs, not one.** Real layer (production, capacity, inventories, labour, prices of goods) and financial
   layer (discount rates, premia, leverage, flows) stay separate. Asset prices are where they meet. Never add a
   hand-set correlation matrix or a hand-set sector beta.
2. **Institutional nodes are first-class.** Demand originates at HOUSEHOLD, GOVT, ROW and investment — never
   inject a shock directly into a sector's output.
3. **Edges are typed objects** `{src, dst, channel, elasticity, lag kernel, sign, state gate, saturation}`.
4. **Drive with the 7 primitive shocks only** — demand, supply, cost_push, monetary, risk_appetite, fiscal,
   catastrophe. A narrative *event* is a named composition of primitives with a target mask and probabilistic
   follow-ups. Every downstream sector move is a response through A and the typed edges.
5. **Stock-flow consistency from day one.** Every payment is a balanced ledger posting between two entities.
   Every financial instrument sums to zero across entities. If money leaks, it is a bug — never clip a balance
   to hide it. SFC is built **before** the credit and collateral edges.
6. **Derive, don't assert.** Betas, rotation, regime flips must emerge from primitives + A. Config elasticities on
   financial edges are calibration targets unless a card says they are mechanisms.
7. **Bound the price *step*, never the price *level*.** Real commodity shocks run 3–5×. Level stickiness exists only
   through `pass_through` / `pass_lag_q` (regulated and service sectors).
8. **Inventories leak** (spoilage, carrying cost, obsolescence) and are controlled by a proportional stock-gap rule.
9. **Nominal smoothing is drift-compensated**: `s ← s·g + (x − s·g)/τ`, `g = exp(π_e/12)`. Do not deflate by realised
   CPI (verified: it removes a needed damper). The no-shock economy must be exactly stationary in real terms at
   π\* = 0 **and** π\* = 2 %.
10. **Headless, deterministic core.** The game UI is an API client like any other. Same seed + same inputs → same
    state hash. No global RNG, no wall-clock, no iteration over unordered sets/dicts in the core.
11. **NPC mass + agent entities.** Aggregate state is dense `(region, sector)` arrays; agent firms and portfolios are
    small structured tables plugged into the same ledger and markets. "All firms on autopilot ≈ aggregate model"
    is a regression test.
12. **The engine runs no trading strategies.** Background order flow is a statistical model, not an agent.
13. **Licence posture:** own implementation, permissive dependencies only (MIT/BSD/Apache-2.0/PSF). Never copy
    code from unlicensed repos (e.g. AlphaTrade/JAX-LOB) or EPL projects (PAMS). Papers and ideas are fine.
14. **Scope discipline (Mizuta):** model only the mechanism the question needs. No new subsystem without a card.
15. **No unbounded integrators in feedback loops.** Any rule that accumulates a gap (price-level makeup terms, inventory
    stocks, cumulative deficits feeding a reaction function) must leak and be clipped. Verified twice: undamped inventory
    stocks diverge at every gain, and an unbounded makeup term produced a −93 % output gap.
16. **Policy is a lever set with an autopilot.** `GOVT` and `CENBANK` act only through the policy-authority framework
    (`autopilot | scripted | agent`). Never hard-code a policy reaction anywhere else. Every policy action is financed
    and posted on the ledger — nothing by fiat.
17. **One corporate borrowing rate.** With `credit.pricing: uniform` (default) no loan or bond price may depend on firm,
    sector, region or rating: it is `policy rate (or the matching government yield) + one global spread`. Leverage is
    disciplined by quantity limits (caps, collateral, rationing), never by price.

## Coding standards

- Python ≥ 3.11, `src/marketsim/` layout, type hints on all public functions, `ruff` clean, numpy-vectorised over
  `(R, S)`; no Python loops over sectors in per-tick code paths.
- **Every parameter lives in `config/*.yaml`** and is validated by a pydantic model. No magic numbers in code —
  a literal other than 0/1/2 needs a named config key or a comment citing the spec equation.
- Docstrings state **units** (cr/month, index, annual decimal, 100bp, months). Rates are annual decimals unless the
  name ends in `_m` (per month) or `_d` (per day). Lags in config are quarters (`*_q`) or months (`*_m`).
- Sector order is the order in `config/io_table.json` — import `CODES`, never retype the list.
- Randomness only through `core.rng.stream("<name>")`. State only in objects that implement `to_state()/from_state()`.
- Modules expose pure functions where possible; `World` owns orchestration. No import-time computation.
- Tests: `tests/unit` (fast, deterministic), `tests/validation` (economic behaviour; marker `validation`),
  `tests/market`, `tests/adversarial`, `tests/golden`. Long runs carry marker `slow`.

## Definition of done (every task)

- [ ] Tests named in the card exist and pass; `make check` green; no test weakened.
- [ ] New config keys documented in the config registry table (master plan §8) and validated by pydantic.
- [ ] Public functions typed and documented with units; no new global state; determinism test still passes.
- [ ] From Phase 2: SFC assertion on for every test that steps the world.
- [ ] `PROGRESS.md` updated; commit message starts with the task ID.

## Never

- Never cap a price level, clip a debt/cash balance, or add noise to make a test pass.
- Never change `config/sectors.yaml`, `config/edges.yaml` values or a validation tolerance without an ADR entry in
  `claude/plan/DECISIONS.md` approved by the human.
- Never call the network from the core or from tests (the BEA fetch script is the only network code).
- Never expose hidden state through `observe()` outside debug mode.
