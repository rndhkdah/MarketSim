# Phase 5 — Agent-operated firms

**Goal.** Agents own and operate firms (D9): price, produce, hire, procure, invest, finance, exit — inside the same
cells as the NPC mass, on the same ledger. Every lever has an **autopilot** so an agent can control any subset; with all
firms on autopilot the hybrid economy must reproduce the aggregate one. Bounded bankruptcy and abuse controls keep the
world safe against adversarial RL policies.

## 5.1 Gate

1. **Hybrid ≈ aggregate:** split all NPC capacity into N autopilot firms per cell; against the Phase-3 golden series
   (10 years, fixed shock script) GDP path within 1 %, sector outputs within 2 %, CPI level within 0.5 %.
2. **Adversarial:** price-at-floor, input cornering, max leverage, unbounded hiring, strategic default → economy bounded
   (gap attributable to the attacker < 5 %), attacker bankrupt within 36 months or strategy unprofitable, no SFC leak,
   no negative physical stock.
3. **Monopoly:** a firm holding 100 % of a cell that prices +30 % loses share to NPC entry; excess profit under the entry
   threshold within 8 years.
4. **Cascade:** a large bankruptcy writes down bank capital, emits an event, stays bounded.
5. **One agent** on price / production / capex in-process for 10 simulated years, plus a 10,000-tick random-policy
   smoke run — no invariant violation.
6. **Books:** every firm's balance-sheet identity holds monthly; income statement ties to ledger tags.

## 5.2 Structure

Each (region, sector) cell = aggregate NPC producer + zero or more agent firms; cell supply = NPC output + Σ firm
output. The NPC mass is the stabilising counterweight; its initial share is a world / difficulty parameter
(`firms.npc_initial_share`). Cells with **no** NPC producer must work: reference price = sales-weighted posted prices
of the firms; NPC entry (T3.14) can re-create NPC capacity. Firms use the cell's technology (column of A, labour and
import coefficients) times a firm productivity factor (R&D raises it). Aggregates (GDP, CPI, unemployment, sector
output, utilisation) sum NPC mass + firms.

**Firm state:** id, operator, status (active | distressed | bankrupt | liquidated); plants `{region, sector, capacity,
productivity, vintage, construction pipeline}`; employees and wage offer, vacancies; finished and input inventories,
backlog; posted price and customer-share state per cell; cash, credit line (limit, drawn), term debt, shares
outstanding (cap table in Phase 6), rating, tax-loss carry-forward; autopilot flag per lever. Full double-entry
accounts on the central ledger (`FIRM:<id>`): every outflow is another entity's inflow.

## 5.3 Goods market with heterogeneous sellers

Per cell, sellers f ∈ {NPC, firms} post prices. `p_ref` = sales-weighted mean posted price.

```
target share   σ*_f ∝ capacity_f · (p_f / p_ref)^(−ε_s) · avail_f^κ        ε_s = 4.0, κ = 1.0
stickiness     σ_f += (σ*_f − σ_f) / τ_loyalty                              τ_loyalty = 6 months
demand         D_f = σ_f · D_cell ;  sellers ration as in R6
spill-over     unmet demand → other sellers pro-rata to spare supply → backlog (split by share, with loss)
```

Equal prices → shares ∝ capacity (needed for gate 1). Stickiness prevents knife-edge switching. The NPC seller prices
with the Phase-2 rule on its own utilisation. The cell price used by buyers and the CPI is the sales-weighted
transaction price. Quality / brand: Phase 8.

## 5.4 Levers (`FirmDecision`, every field optional → autopilot)

| lever | fields | limits / timing |
|---|---|---|
| pricing | posted price per cell | step ≤ 15 %/month; floor 1 % of `p_ref`; month boundary |
| production | target output per plant | ≤ capacity × overtime cap |
| labour | vacancies, wage offer, layoffs | hires ≤ 10 % of regional unemployed/month; firing cost `firing_cost_months` (2) × wage; 1-month notice |
| procurement | input cover targets (0–6 months), order multipliers, shortage bid premium (0–0.5) | supply contracts: Phase 8 |
| capex | expand plant, new plant in another cell (+10 % set-up cost), R&D spend | cost `v_s·p_I` per unit capacity, time-to-build `build_lag_q`, payments Erlang(3); ≤ +50 % capacity/year |
| financing | borrow / repay (floating bank loan or fixed corporate-pool funding — same spread), dividends; issue equity, buybacks (Phase 6) | next tick; leverage cap ND/EBITDA ≤ 4 (hard cap 6) |
| treasury | switch allowing the firm to trade financial instruments | default off |
| exit | liquidate, sell | plants to NPC mass at 30 % discount; creditors first |

**Autopilot = the Phase-2 rules at firm scale:** price R7 (cost-plus × tightness, two-speed filter); plan R4; hiring R8;
input orders R5; capex §2.6 on own K with supply line; payout and the slow leverage norm R9. Needed for playability,
for RL curricula (price → production → hiring → capex → financing) and for gate 1.

## 5.5 Labour, procurement, financing

**Labour matching.** Regional pool of unemployed. Vacancies `V_f` at wage offer `w_f`:
`hires_f = V_f · min(1, m0·(U_r/V_r)^0.5) · (w_f/w̄_r)^ε_w`, scaled so Σ hires ≤ pool; hired workers produce from next
month; quits flow to better payers ∝ `(w̄_r/w_f − 1)⁺`; incumbent wages fall no faster than `1/wage_stickiness_q`.
**Procurement under shortage.** When a supplier cell's fill < 1: `alloc_b ∝ order_b · exp(β·premium_b) · (1 + ρ·rel_b)`,
`rel_b` = 12-month EMA of the buyer's purchase share (supplier relationship), water-filled and capped at the order;
otherwise pro-rata. Holding inventory (or, later, contracts) before a shock is a core strategy.
**Financing.** Retained earnings; bank credit line — limit = min(leverage cap × EBITDA_12m, 50 % of plant replacement
value) × lagged lending capacity `Λ̃` (the credit gate applies to agents too); **rate = the common corporate rate (D14)** — policy + the
single global spread `s_t` on a floating bank loan, or the matching government yield + `s_t` through the corporate bond pool
(`06-…` §6.11) — identical for every firm, sector, region and rating; corporate tax on positive EBT with 5-year loss carry-forward. **Rating:** score from ND/EBITDA, 1/ICR, size,
earnings volatility → AAA…CCC → **credit-limit multipliers and covenants** (indicative × 1.2, 1.1, 1.0, 0.9, 0.7, 0.5,
0.25) and public disclosure — never the price of credit. Because leverage is not priced, the leverage cap and the collateral
limit are hard constraints; default losses are mutualised and raise `s_t` for everyone.
**Hard budget constraint:** no cash below zero beyond the undrawn line. Payments that would breach are scaled back by
seniority — wages, taxes, suppliers, interest, principal, capex, dividends — and unpaid obligations mark distress.

## 5.6 Bankruptcy — deterministic, single-step, bounded

Trigger: equity < 0 for 3 consecutive months, or cash + undrawn credit < obligations due. Resolution at month end:
(1) freeze levers; (2) plants to the NPC mass at replacement value × 0.70, financed by a new NPC loan; inventories at
50 %; (3) waterfall: wages and taxes → secured bank debt → suppliers; (4) creditor losses capped at exposure, posted as
write-offs → bank capital → credit gate (the systemic link); (5) workers to the regional pool; (6) shareholders wiped,
shares delisted; (7) if liabilities > `large_threshold` × monthly GDP emit `large_bankruptcy` so it propagates through
the same machinery as external shocks.

## 5.7 Information rules and abuse controls

Operators see their own books live; everyone else sees lagged reports (monthly summary, 10-day lag; quarterly
statements, 21-day lag), prices and news. `world.mode: game` allows insider knowledge; `professional` enforces
disclosure timing, blackout windows (no own-share trading from quarter end to publication) and immediate disclosure of
dividends, issuance and large capex.
Expected abuse: pricing at the floor, cornering inputs, unbounded hiring, max leverage, self-pumping, strategic default,
collusion. Controls: hard budget constraint, leverage cap, capacity and labour-pool limits, per-tick decision and order
size limits, non-negativity invariants; economic counterweights (demand elasticity, NPC entry, share stickiness,
losses → bankruptcy in bounded time); optional regulator (cell share cap, fines); evaluation against fixed NPC
competitors to catch collusion.

---

## 5.8 Task cards

### T5.01 — `firms.yaml` and firm state
**Depends:** T3.15 · **Size:** M · **Files:** `config/firms.yaml`, `src/marketsim/firms/firm.py`, `tests/unit/firms/test_firm_state.py`
**Read first:** §5.2, §5.4. **Build:** pydantic config (all limits above), `Firm`, `Plant` dataclasses, registry with stable
ids and sorted iteration, state round trip. **Tests:** serialisation; invalid plant (unknown cell) rejected.

### T5.02 — Firm accounts on the ledger
**Depends:** T5.01, T2.03 · **Size:** M · **Files:** `src/marketsim/firms/accounts.py`, `tests/unit/firms/test_accounts.py`
**Build:** register `FIRM:<id>` / `AGENT:<id>`; chart of accounts from flow tags; monthly close → income statement and
balance sheet; real assets at replacement cost. **Tests:** gate 6 on a scripted 24-month firm; founding capital enters as
a tagged `capital_transfer` / `equity_issue`, SFC green.

### T5.03 — Founding and entry
**Depends:** T5.02 · **Size:** S · **Files:** `src/marketsim/firms/founding.py`, `tests/unit/firms/test_founding.py`
**Build:** found a firm with capital; acquire capacity by buying from the NPC mass (price = replacement value) or
greenfield (time-to-build); `npc_initial_share`. **Tests:** cell capacity conserved on purchase; greenfield arrives after
the lag; cash paid = cash received.

### T5.04 — Cell aggregation with firms (incl. zero-NPC cells)
**Depends:** T5.03 · **Size:** M · **Files:** `src/marketsim/real/economy.py`, `src/marketsim/firms/cells.py`, `tests/unit/firms/test_cells.py`
**Build:** cell supply, utilisation, employment and aggregates over NPC + firms; reference price fallback with no NPC.
**Tests:** aggregates equal sums; a cell with zero NPC capacity runs 120 months; NaN-free.

### T5.05 — Heterogeneous-seller goods market
**Depends:** T5.04 · **Size:** M · **Files:** `src/marketsim/firms/goods_market.py`, `tests/unit/firms/test_goods_market.py`
**Read first:** §5.3. **Tests:** equal prices → shares ∝ capacity; a 5 % price cut gains share gradually (half-life ≈ τ);
no knife-edge (share path smooth under ±1 % price noise); spill-over conserves demand; with one seller ≡ Phase-2 R6.

### T5.06 — Decision levers and validation
**Depends:** T5.04 · **Size:** M · **Files:** `src/marketsim/firms/levers.py`, `tests/unit/firms/test_levers.py`
**Read first:** §5.4. **Build:** `FirmDecision`, validation and clipping to limits with a machine-readable list of
adjustments returned to the agent; timing (month boundary vs next tick). **Tests:** each limit; partial decisions leave
other levers on autopilot; decisions for a firm the agent does not control are rejected.

### T5.07 — Autopilot
**Depends:** T5.06 · **Size:** M · **Files:** `src/marketsim/firms/autopilot.py`, `tests/unit/firms/test_autopilot.py`
**Build:** firm-scale versions of R4, R5, R7, R8, §2.6, R9 reusing the Phase-2 pure functions (no copies).
**Tests:** a single autopilot firm owning a whole cell reproduces the NPC path for that cell to 1e-9.

### T5.08 — Labour market with matching
**Depends:** T5.06 · **Size:** M · **Files:** `src/marketsim/firms/labour_market.py`, `tests/unit/firms/test_labour_market.py`
**Read first:** §5.5. **Tests:** Σ hires ≤ pool; higher wage offer fills faster; firing cost charged; autopilot wages
reproduce the regional wage path; unbounded vacancies cannot hire more than the pool share cap.

### T5.09 — Procurement and shortage allocation
**Depends:** T5.05 · **Size:** M · **Files:** `src/marketsim/firms/procurement.py`, `tests/unit/firms/test_procurement.py`
**Tests:** no shortage → pro-rata and premiums irrelevant (no payment); shortage → bid and relationship raise allocation;
allocations ≤ orders and sum to supply; premium paid is posted; cornering attempt limited by order-size cap.

### T5.10 — Financing, rating, tax, hard budget constraint
**Depends:** T5.02, T2.20 · **Size:** M · **Files:** `src/marketsim/firms/financing.py`, `src/marketsim/firms/rating.py`, `tests/unit/firms/test_financing.py`
**Read first:** §5.5. **Tests:** credit limit shrinks when the gate closes and with a worse rating; the borrowing rate is identical across firms whatever
their rating or leverage (D14); seniority scaling when
cash is short; loss carry-forward; no negative cash beyond the line in a fuzz run.

### T5.11 — Plants, capex and R&D
**Depends:** T5.06 · **Size:** M · **Files:** `src/marketsim/firms/plants.py`, `tests/unit/firms/test_plants.py`
**Build:** construction pipeline (capacity completions as queue items), payments profile, new cell set-up premium, R&D →
productivity with diminishing, stochastic returns (stream `firms.rnd`). **Tests:** capacity online after `build_lag_q`
(ENERGY 12q vs BIZSVC 1q); capex demand lands on the capex routing sectors; R&D deterministic per seed.

### T5.12 — Bankruptcy resolution
**Depends:** T5.10 · **Size:** M · **Files:** `src/marketsim/firms/bankruptcy.py`, `tests/unit/firms/test_bankruptcy.py`
**Read first:** §5.6. **Tests:** waterfall amounts by hand; creditor loss ≤ exposure; SFC green through resolution;
workers return to the pool; event emitted above the threshold; resolution is one step (no loops).

### T5.13 — Reports and information rules
**Depends:** T5.02, T4.07 · **Size:** S · **Files:** `src/marketsim/firms/reports.py`, `tests/unit/firms/test_reports.py`
**Read first:** §5.7. **Tests:** non-operators never see un-lagged books; blackout enforced in professional mode only.

### T5.14 — Abuse controls and optional regulator
**Depends:** T5.06, T5.10 · **Size:** S · **Files:** `src/marketsim/firms/controls.py`, `tests/unit/firms/test_controls.py`
**Tests:** each control triggers on a constructed violation; regulator fines are ledger postings to `GOVT`.

### T5.15 — Firm-level events
**Depends:** T4.12, T5.11 · **Size:** S · **Files:** `src/marketsim/events/firm_hooks.py`, `config/events/firm_*.yaml`
**Tests:** plant accident removes capacity of one plant; strike zeroes a firm's labour for the duration; recall hits
inventory and cash; all through the ledger.

### T5.16 — Hybrid ≈ aggregate test
**Depends:** T5.07, T5.08, T5.09 · **Size:** M · **Files:** `tests/validation/test_hybrid_vs_aggregate.py`
**Tests:** gate 1 for N ∈ {1, 5} firms per cell (`slow`); report deviations per sector.

### T5.17 — Adversarial, monopoly and cascade tests
**Depends:** T5.12, T5.14 · **Size:** M · **Files:** `tests/adversarial/test_firm_attacks.py`, `tests/adversarial/test_monopoly.py`, `tests/adversarial/test_bankruptcy_cascade.py`
**Tests:** gates 2–4 with scripted attacker policies.

### T5.18 — Minimal single-agent loop (in-process)
**Depends:** T5.07 · **Size:** S · **Files:** `src/marketsim/world.py`, `scripts/run_world.py`, `tests/integration/test_single_agent.py`
**Build:** `World.submit(agent, FirmDecision)` path for price / production target / capex; scripted and random policies.
**Tests:** gate 5; determinism with an agent in the loop.

### T5.19 — Collusion evaluation harness
**Depends:** T5.18 · **Size:** S · **Files:** `src/marketsim/scenarios/evaluation.py`, `tests/unit/scenarios/test_evaluation.py`
**Build:** evaluate a policy against fixed NPC-quality competitors; report margin vs competitive benchmark.
**Tests:** two firms scripted to collude are flagged; competitive autopilots are not.
