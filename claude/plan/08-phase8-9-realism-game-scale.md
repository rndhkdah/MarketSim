# Phases 8–9 — Realism, game, scale-out

Coarse on purpose. After the Phase-7 gate, split each card below into cards of the usual granularity (one PR each) in
a planning task of its own. Nothing here may weaken an earlier gate.

## 8.1 Gates

**Phase 8:** calibration report accepted by the human; a playable campaign runs end-to-end through the public API;
explainability answers "why did my demand fall?" for any firm-month. **Phase 9:** none fixed yet.

## 8.2 Game requirements carried from the brief (§12)

Explainability traces for prices and firm results; entry fairness (founding capital, NPC scale as difficulty);
planning gameplay from lags and events (build times, inventories before shocks); takeover drama from the equity
market; pacing (pause, speed), narrative event text, save / load, scoring (net worth, valuation, market share).

---

## 8.3 Task cards — Phase 8

### T8.01 — Financing depth
**Depends:** T7.14 · **Size:** L · **Files:** `src/marketsim/firms/financing.py`, `src/marketsim/pricing/bonds.py`
**Build:** term loans with maturities and refinancing risk, covenants, committed vs uncommitted lines, extra pool maturities —
all at the common corporate rate (D14; pooled corporate bonds already exist from T6.29). **Tests:** refinancing wall under a closed credit gate produces distress; SFC green.

### T8.02 — Supply contracts
**Depends:** T7.14 · **Size:** M · **Files:** `src/marketsim/firms/contracts.py`
**Build:** fixed-price / fixed-quantity forward contracts between firms and supplier cells, priority in shortage
allocation. **Tests:** a contracted buyer is served first during a supply event; contract cash flows on the ledger.

### T8.03 — Quality and brand
**Depends:** T7.14 · **Size:** M · **Files:** `src/marketsim/firms/goods_market.py`
**Build:** quality term in the share function, built by R&D and marketing spend, decaying. **Tests:** hybrid ≈ aggregate
unchanged when quality is equal.

### T8.04 — Calibration pass — **HUMAN GATE**
**Depends:** T7.14, T0.09 · **Size:** L · **Files:** `scripts/fetch_bea_io.py`, `scripts/calibrate_*.py`, `claude/plan/reports/calibration.md`
**Build:** run the BEA fetch; add the **final-demand concordance** (BEA PCE, investment, government, export columns →
18 sectors) so final-demand vectors stop being seeded; fit business-cycle moments to FRED / BLS targets — sd(I)/sd(GDP)
3–4, sd(C)/sd(GDP) < 1, GDP persistence; size of the cost-push response; event magnitudes cross-check. Propose parameter
changes with before/after tables. **Tests:** all earlier gates still green with the proposed set.

### T8.05 — Historic replay comparison
**Depends:** T8.04 · **Size:** M · **Files:** `tests/validation/test_historic_replay.py`
**Build:** scripted 1973, 2008 and 2020–22 scenarios vs stylised historic paths: directions, ordering of sector
responses, rough magnitude bands. **Tests:** bands documented with sources.

### T8.06 — Explainability traces
**Depends:** T7.04 · **Size:** L · **Files:** `src/marketsim/explain/traces.py`, `src/marketsim/api/rest.py`
**Build:** exact decompositions — a firm's demand change = cell demand (income, relative price, rate, typed-edge and want
shifters, events) × share (own price vs reference, availability, stickiness) × rationing; an asset's return = earnings
expectation + discount rate + sentiment + impact + noise. **Tests:** components sum to the total (1e-9); API endpoint.

### T8.07 — Game layer contract
**Depends:** T7.15 · **Size:** L · **Files:** `docs/game-contract.md`, `src/marketsim/scenarios/campaign.py`
**Build:** pacing controls, save / load slots, scoring, narrative text templates from `NewsItem`, difficulty = founding
capital × NPC scale × event severity, campaign scenarios, takeover notifications. **Tests:** a scripted "player" finishes
a campaign through the public API only.

### T8.08 — Regulator
**Depends:** T7.14 · **Size:** M · **Files:** `src/marketsim/firms/controls.py`, `config/events/regulator_*.yaml`
**Build:** cell share caps, fines, merger control, antitrust events. **Tests:** fines are ledger postings; caps bind.

### T8.09 — Regional real-estate asset
**Depends:** T7.14 · **Size:** M · **Files:** `src/marketsim/pricing/realestate.py`, `config/markets.yaml`
**Build:** tradable regional property index tied to REALESTATE value and collateral. **Tests:** collateral channel
consistent with the traded price.

### T8.10 — Tier-3 counterfactual validation (research)
**Depends:** T7.14 · **Size:** M · **Files:** `claude/plan/reports/tier3-counterfactual.md`, `tests/validation/test_intervention_consistency.py`
**Build:** intervention-consistency experiments (same seed with and without an agent's order; invariance of unrelated
paths; dose-response monotonicity). Write up what can and cannot be claimed — this is an unsolved problem.

### T8.11 — Korea-flavoured calibration set (optional)
**Depends:** T8.04 · **Size:** M · **Files:** `config/kr/*`, `scripts/fetch_bok_io.py`
**Build:** Bank of Korea / KOSIS IO concordance to the 18 sectors, policy and structural parameters as overrides.

### T8.12 — Goods layer (only if ADR T3.16 approved it)
**Depends:** T3.16, T7.14 · **Size:** L · **Files:** `src/marketsim/demand/goods.py`, `config/goods.yaml`
**Build:** goods beneath STAPLES / DISCRET / AGRIFOOD for game legibility; sectors remain the financial unit.

## 8.4 Task cards — Phase 9

### T9.01 — Multi-country and FX — **deferred (ADR-017)**
**Depends:** T8.04 · **Size:** L · **Files:** `src/marketsim/regions/countries.py`, `src/marketsim/market/fx.py`
**Build:** second economy, FX market, bilateral trade replacing part of ROW; FX in the ledger as an instrument. **Out of v1.**

### T9.02 — Storage backend
**Depends:** T7.10 · **Size:** L · **Files:** `src/marketsim/storage/*`
**Build:** event log and snapshots to parquet / a database; world registry. State has been serialisable since T0.11.

### T9.03 — Parallel worlds at scale — **deferred (ADR-017)**
**Depends:** T7.12 · **Size:** M · **Files:** `src/marketsim/sdk/cluster.py`
**Build:** process pool / cluster execution, batch evaluation service. **Out of v1.** T7.12 `VectorEnv` remains.

### T9.04 — M&A depth
**Depends:** T6.16 · **Size:** L · **Files:** `src/marketsim/equity/mna.py`
**Build:** tender offers, non-voting share classes, defences, mergers.

### T9.05 — Rust core for hot paths (only if profiling demands)
**Depends:** T7.13 · **Size:** L · **Files:** `rust/*`, bindings
**Build:** CLOB and the monthly step behind unchanged Python interfaces; bitwise-parity tests against Python.

### T9.06 — gRPC transport (optional)
**Depends:** T7.04 · **Size:** M · **Files:** `src/marketsim/api/grpc/*`
**Build:** same schemas over gRPC for high-frequency clients.
