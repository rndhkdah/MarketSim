# Phase 6 — Asset pricing and markets

**Goal.** The half we actually care about: fundamentals from the real layer, discount rates from the financial layer,
price = V·exp(ξ) with agents' own orders moving ξ (D7); an engine market maker for sector equity, bonds and
commodities; a CLOB for agent-firm shares (finite float → the engine cannot be a bottomless counterparty); cap tables,
listing, dividends, control transfer (D10); margin and forced liquidation; then the feedback edges 4 → 1/2.

**Track B** (T6.07–T6.13: impact, market maker, background flow, CLOB, settlement, margin, shorting) depends only on
`core/` and `ledger/` and may start right after T2.03 with synthetic fundamentals.

## 6.1 Gate

1. **Betas emerge:** simulated valuation sensitivities rank-correlate with `config/betas.md` — Spearman ≥ 0.7 for rate,
   growth and oil, ≥ 0.6 for credit; BANKS is the only sector with a positive rate sensitivity; REALESTATE, SOFTWARE,
   UTILITIES are the most rate-negative.
2. **Regime flip end-to-end:** corr(equity index return, bond index return) < −0.15 in demand-shock episodes and
   > +0.15 in supply / cost-push episodes.
3. **Tier 2 (CI):** meta-order peak impact ∝ (Q/ADV)^δ with fitted δ ∈ [0.4, 0.7]; only weakly dependent on the
   execution horizon (log-log slope vs duration, 1–20 days, between 0 and 0.25); concave; for meta-orders of 5–20 days
   40–75 % of peak impact reverts within 20 days of completion; response function positive and decaying; average
   round-trip P&L ≤ 0 (no price-manipulation arbitrage).
4. **Tier 1:** daily index returns kurtosis > 4; ACF(|r|) > 0 at lags 1–20; |ACF(r, 1)| < 0.1.
5. **Takeover test:** control passes at > 50 % at the next month boundary; ledger balances; Σ Δ net worth = 0 apart from
   fees and taxes.
6. **Liquidation-cascade test:** bounded, monotone deleveraging, SFC holds, losses follow the waterfall.
7. §2.12 stability suite green with each feedback edge switched on.
8. Deterministic with all market streams; ≥ 3,000 ticks/s for 25 instruments without agents.

## 6.2 Instruments (`config/markets.yaml`)

| instrument | what it is | venue |
|---|---|---|
| `EQ:NPC:<SECTOR>` × 18 | claim on the national NPC mass of the sector; pays its dividends pro-rata | engine MM |
| `GBOND10` | 10-year benchmark; index duration 7 (`sectors.yaml: bond_index`) | engine MM |
| `CASH` | deposits at the policy-linked deposit rate | — |
| `OIL`, `METALS`, `GRAINS` | cash-settled on the national reference price of ENERGY, MATERIALS, AGRIFOOD + carry | engine MM |
| `EQ:FIRM:<id>` | agent-firm shares (finite float) | **CLOB** + thin engine quote |

Published, non-tradable: sector indices (cap-weighted NPC sector equity **plus listed agent firms**), market index
(`market_cap_weights` at baseline, float-cap afterwards), bond index. Later: regional real estate, FX.
Liquidity ∝ market cap (or float): `ADV_j = turnover × cap_j` (turnover 0.4 %/day) drives spread, budget and impact.

## 6.3 Valuation (L3)

`P_j = V_j · exp(ξ_j)` — momentum + mean reversion, agents can move prices, mapping not trivially invertible.

**Earnings expectations** use only information public at *t*: `E^e_j` = drift-compensated 12-month EMA of *published*
sector after-tax profit (vintages from T4.07), plus an anchored long-run growth view `g_lr,j = 0.1 · published growth`.
**Curve.** Expected policy path decays to neutral: `E[r_{t+h}] = r_n + π* + (r_t − r_n − π*)·λ^h`, monthly λ = 0.978, so
a policy move passes ≈ 0.35 into the 10-year average — reproducing Layer 1's `DISC_PASS` rather than asserting it.
`y10 = mean expected path + tp_t`; `tp` is AR(1) with a risk-appetite loading. Bond return `= −D·Δy + carry`.
**Discount rate.** `ρ_j = y10_real + ERP_t + adj_j`; `ERP_t = ERP_0 + z_risk + sentiment term`.
**Fundamental.** `ln V_j = ln E^e_j + ln PE0_j − cf_duration_j·(ρ_j − ρ_j0) + cf_duration_j·Δg_lr,j`.
**Financials.** BANKS earnings already contain the margin channel (Phase-2 banks); the `financials` block supplies
only what the ledger does not model (`curve_beta`, INSURANCE `float_rate_beta`, `bond_mtm_duration`) exactly as
`derive_betas` does; `nim_rate_beta` is used **only** with `banks.mode: passthrough` (no double counting).
**Agent-firm shares:** the same model on the firm's lagged public reports; dividends, dilution and leverage move value.
**Corporate debt:** rate + rating spread. **Commodities:** cell reference price + carry.
`pricing.mode: structural | factor_lite` — `factor_lite` prices sector equity directly from the derived betas (fast
training mode); `structural` is the default and the one the gates test.

**Mispricing.** `ξ_j = I_j + s_j + n_j`: impact state (§6.4); sentiment `s ← φ_s·s + κ_mom·(r̄_20d − r̄_250d) + κ_news·news
+ common risk-appetite loading` (bubble events raise drift and the crash hazard reads `ξ`); `n` is OU noise with
σ ∝ `idio_vol_j`, scaled so the calm-regime index volatility is 15–18 % a year. **Limits to arbitrage:** mean-reversion
speed `θ_t = θ_0 · clip(A_t/A_0, 0.2, 2)`; arbitrageur capital `A` loses when `|ξ|` widens against it — so dislocations
persist exactly when capital is scarce. Anti-exploit: noise, hidden state, per-episode domain randomisation, limits.

## 6.4 Impact kernel (responsiveness — "your orders move the price")

```
ΔI_j   = Y · σ_j · sign(q) · (|q| / ADV_j)^δ            q = net signed executed volume this tick (agents + background)
I_k   ← ρ_k · I_k + w_k · ΔI_j ,  k = 1..5              half-lives {1, 5, 20, 60, 250} days
ξ_impact = Σ_k I_k                                       weights w_k fitted to G(τ) = (1 + τ/τ0)^(−β)
```

Defaults δ = 0.5, β = 0.5, Y = 0.8, τ0 = 1 day. With β = δ = ½ a meta-order's peak impact is Yσ·√(Q/ADV) times a
factor that only drifts from 1.0 (one day) to ≈ 1.7 (twenty days) and tends to 2 — i.e. the √-law holds **almost
independently of how the order is split** (checked numerically: log-log slope vs duration ≈ 0.18; ≈ 50–70 % of peak
impact reverts within 20 days for 5–20-day executions; a 5-exponential fit tracks the power law within 3 % on 1–250 days). `β + δ ≥ 1` is the no-dynamic-arbitrage boundary. Both
properties are asserted by tests, not assumed. Fills pay the full post-impact mid ± half-spread (nobody trades ahead of their own
impact). O(K) state per instrument. This is the daily-resolution reading of the target spec in the landscape doc
(concave impact × power-law decay); a message-level queue-reactive book is out of scope for v1.

## 6.5 Engine market maker and background flow

Quotes `P·(1 ∓ s/2)` with `s = s0 + k_σ·σ_j + k_inv·|inv_j|/cap_j` and a mid skew `−k_skew·inv_j/cap_j`. Next-tick
fills. Per-tick liquidity budget `participation_cap × ADV_j` (0.25): excess is partially filled; the rest rests (GTC)
or cancels (IOC). Counterparty = `MM`, the NPC-investor sub-account of households → SFC-safe. Order types on both
venues: market, limit, stop.
**Background flow** (own implementation, stream `flow.<instrument>`): signed volume with long-memory sign
autocorrelation (sum of AR(1)s), intensity ∝ ADV, a mean-reversion and a momentum component reacting to `ξ`. On MM
venues it is NPC investors trading with NPC investors: it feeds volume and the impact kernel and **posts nothing** on
the ledger. On the CLOB it must be real orders from `MM` (they match against agents): a queue-reactive-lite generator —
limit / cancel / market intensities conditional on spread and top-of-book imbalance — sized to float so books are
never empty.

## 6.6 CLOB for agent-firm shares

Price-time priority; tick and lot size; market / limit / stop, IOC / GTC / day; cancel-replace; self-trade prevention
(cancel newest); opening call auction for IPOs and after halts; ±20 % daily band → halt → auction (configurable);
the engine posts only a **thin quote** around fundamental (`thin_quote_size` at `V·(1 ± w)`, w 2–5 %). Deterministic:
within a tick orders are processed by `(agent id, sequence)` under lockstep, arrival order in real time.
`sortedcontainers.SortedDict` price levels → FIFO deques. Order expiries are queue items.

## 6.7 Cap tables, listing, control (D10)

Founder holds 100 % (default 1,000,000 shares). **Listing:** primary issuance (new shares, cash to the firm) or
secondary sale (founder's shares, cash to the founder) through a call auction with a reserve; the float is what trades.
Dividends per cap table each period (record date = declaration + 2 ticks); issuance dilutes; buybacks (firm buys on the
CLOB and cancels) concentrate. **Any agent may buy any listed share, funds permitting.** Net worth — the game score and
RL reward base — = cash + market value of holdings + own-firm equity (market value if listed, book if not).
**Control:** > 50 % of voting shares → operating control passes at the next month boundary; the previous operator keeps
shares and dividends; a founder who wants control does not issue past 49 % (engine warns). Holdings ≥ 5 % are disclosed
in professional mode. Tender offers, non-voting classes, defences: Phase 9.
**Ledger:** secondary trades are cash-for-shares swaps between accounts; primary issuance is cash-into-firm; the issuer
carries the negative `EQ` position so the instrument sums to zero.

## 6.8 Margin, shorting, forced liquidation — where systemic risk meets microstructure

Margin loans via the broker (`MM`) funded by `BANKSYS` at policy + broker spread. Equities: initial 50 %, maintenance
25 %; commodities 10 % / 7 %. Shorts: NPC sector equity and commodities only in v1 (agent-firm shares not lendable by
default); borrow fee rises with utilisation. Margin check at end of tick → forced market orders next tick, sized to
restore initial margin, largest loser first, executed **with impact** → cascades are possible by construction. Losses
beyond account equity: broker up to a cap, then `BANKSYS` write-off → bank capital → credit gate. That chain — forced
selling → prices → bank capital → lending → real economy — is the cross-layer coupling nobody models (landscape gap 3).

## 6.9 Feedback edges 4 → 1/2 (one config switch each, each re-running §2.12)

(a) cost of capital → capex: `Δerp` and the Q term of §2.6 go live; (b) wealth → consumption: household equity wealth at
market value enters consumption with MPC 0.03 a year on a 12-month smoothed value; (c) credit spreads → refinancing →
hiring: employment target scaled by `(ICR/1.5)^0.2` when interest cover < 1.5.

## 6.10 Surveillance and fees

Fees, transaction tax (default 0, to `GOVT`), borrow fee. Position limits for leveraged and commodity positions
relative to ADV / float; wash-trade detection (self-cross blocked, circular trades within N ticks flagged); pump
detection (agent's volume share > 50 % during a > 20 % five-day run-up); disclosure timing in professional mode;
optional minimum holding period as a game rule.

---

## 6.11 Task cards

### T6.01 — `markets.yaml` and the instrument registry
**Depends:** T2.03 · **Size:** S · **Files:** `config/markets.yaml`, `src/marketsim/market/instruments.py`, `tests/unit/market/test_instruments.py`
**Read first:** §6.2. **Tests:** registry round trip; dynamic listing of `EQ:FIRM:*`; ADV rule.

### T6.02 — Earnings expectations from public information
**Depends:** T4.07, T2.21 · **Size:** S · **Files:** `src/marketsim/pricing/expectations.py`, `tests/unit/pricing/test_expectations.py`
**Tests:** expectations never use unpublished vintages (poison the true series → output unchanged); exact under steady π\*.

### T6.03 — Yield curve, term premium, bond pricing
**Depends:** T2.14 · **Size:** M · **Files:** `src/marketsim/pricing/curve.py`, `src/marketsim/pricing/bonds.py`, `tests/unit/pricing/test_curve.py`
**Read first:** §6.3 curve. **Tests:** +100bp policy move → y10 +35bp (±5); bond index return ≈ −7 × Δy; carry accrues.

### T6.04 — Discount rates and fundamental value
**Depends:** T6.02, T6.03 · **Size:** M · **Files:** `src/marketsim/pricing/discount.py`, `src/marketsim/pricing/fundamentals.py`, `src/marketsim/pricing/provider.py`, `tests/unit/pricing/test_fundamentals.py`
**Build:** replaces the Phase-2 stub behind the same interface; financials override; `factor_lite` mode.
**Tests:** Q = 1 at baseline; duration ordering of rate sensitivity; stub parity when earnings are constant.

### T6.05 — Betas-emerge validation
**Depends:** T6.04 · **Size:** M · **Files:** `tests/validation/test_pricing_vs_betas.py`, `claude/plan/reports/pricing-validation.md`
**Tests:** gate 1 — run rate, growth, oil and spread impulses through the full model, regress valuation changes.

### T6.06 — Mispricing: sentiment, noise, limits to arbitrage
**Depends:** T6.04 · **Size:** M · **Files:** `src/marketsim/pricing/mispricing.py`, `tests/unit/pricing/test_mispricing.py`
**Read first:** §6.3 mispricing. **Tests:** ξ stationary without flow; reversion slows when arbitrage capital is low;
calm-regime index volatility 15–18 %; streams isolated.

### T6.07 — Impact kernel
**Depends:** T6.01 · **Size:** M · **Files:** `src/marketsim/market/impact.py`, `tests/unit/market/test_impact.py`
**Read first:** §6.4. **Build:** weight fitting (non-negative least squares on a log-spaced grid to 500 days), state
update, fill-price rule. **Tests:** fitted kernel within 10 % of the power law on 1–250 days; single-order concavity;
state decays to 0; O(K) per instrument.

### T6.08 — Engine market maker
**Depends:** T6.07 · **Size:** M · **Files:** `src/marketsim/market/mm.py`, `src/marketsim/market/venue.py`, `tests/unit/market/test_mm.py`
**Read first:** §6.5. **Tests:** spread widens with volatility and inventory; budget → partial fills; next-tick fills;
skew sign; `Venue` protocol shared with the CLOB.

### T6.09 — Background order flow
**Depends:** T6.08 · **Size:** M · **Files:** `src/marketsim/market/background.py`, `tests/market/test_background_flow.py`
**Tests:** sign autocorrelation positive and slowly decaying; volume ≈ ADV; no ledger postings on MM venues; deterministic.

### T6.10 — CLOB
**Depends:** T6.01 · **Size:** L · **Files:** `src/marketsim/market/clob.py`, `tests/unit/market/test_clob.py`
**Read first:** §6.6. **Tests:** price-time priority; partial fills; stop trigger; IOC/GTC/day and expiries; cancel-replace
loses priority on size-up only; self-trade prevention; call auction clears at the volume-maximising price; halt band;
property test — no crossed book after any sequence; ≥ 20,000 order events/s.

### T6.11 — CLOB liquidity: thin engine quote + queue-reactive-lite
**Depends:** T6.10, T6.09 · **Size:** M · **Files:** `src/marketsim/market/clob_liquidity.py`, `tests/market/test_clob_liquidity.py`
**Tests:** book never empty over 10 years; depth ∝ float; engine quote tracks fundamental within `w`.

### T6.12 — Settlement, fees, taxes
**Depends:** T6.08, T6.10 · **Size:** S · **Files:** `src/marketsim/market/settlement.py`, `tests/unit/market/test_settlement.py`
**Tests:** every fill is a balanced cash-for-asset Tx; fees and transaction tax routed; NPC dividends reach holders
pro-rata; commodity variation margin nets to zero.

### T6.13 — Margin, shorting, forced liquidation
**Depends:** T6.12 · **Size:** M · **Files:** `src/marketsim/market/margin.py`, `src/marketsim/market/shorting.py`, `tests/market/test_liquidation_cascade.py`
**Read first:** §6.8. **Tests:** gate 6; margin arithmetic; borrow fee accrues; loss waterfall reaches bank capital.

### T6.14 — Cap tables and corporate actions
**Depends:** T5.02, T6.12 · **Size:** M · **Files:** `src/marketsim/equity/captable.py`, `src/marketsim/equity/corporate_actions.py`, `tests/unit/equity/test_captable.py`
**Read first:** §6.7. **Tests:** Σ holdings = shares outstanding always; dividends per record date; buyback cancels shares;
dilution arithmetic; issuer carries the negative position.

### T6.15 — Listing and IPO auction
**Depends:** T6.14, T6.10 · **Size:** S · **Files:** `src/marketsim/equity/listing.py`, `tests/unit/equity/test_listing.py`
**Tests:** primary cash lands in the firm, secondary in the founder; reserve price respected; 49 % warning.

### T6.16 — Control transfer and the takeover test
**Depends:** T6.15, T5.06 · **Size:** M · **Files:** `src/marketsim/equity/control.py`, `tests/market/test_takeover.py`
**Tests:** gate 5; lever authority switches exactly at the month boundary; 5 % disclosures in professional mode; news emitted.

### T6.17 — Agent-firm valuation and thin quote feed
**Depends:** T6.04, T5.13 · **Size:** S · **Files:** `src/marketsim/pricing/firm_value.py`, `tests/unit/pricing/test_firm_value.py`
**Tests:** value reacts to published reports only; dividend cut and dilution lower value; leverage raises risk adj.

### T6.18 — Commodities
**Depends:** T6.08 · **Size:** S · **Files:** `src/marketsim/pricing/commodities.py`, `tests/unit/pricing/test_commodities.py`
**Tests:** OIL tracks the ENERGY reference price + carry; a 3× cost-push event is transmitted without any cap.

### T6.19 — Feedback edges 4 → 1/2
**Depends:** T6.06, T2.24 · **Size:** M · **Files:** `src/marketsim/real/feedbacks.py`, `tests/validation/test_feedback_stability.py`
**Read first:** §6.9. **Tests:** gate 7, one switch at a time and all together (`slow`); report stable ranges.

### T6.20 — Surveillance and limits
**Depends:** T6.12 · **Size:** M · **Files:** `src/marketsim/market/surveillance.py`, `tests/market/test_surveillance.py`
**Read first:** §6.10. **Tests:** scripted wash trade, circular trade and pump are flagged; honest market making is not.

### T6.21 — Tier-1 / Tier-2 statistics and the regime flip
**Depends:** T6.09, T6.06, T6.07 · **Size:** M · **Files:** `src/marketsim/market/metrics.py`, `tests/market/test_tier1.py`, `tests/market/test_tier2_impact.py`, `tests/validation/test_stock_bond_regime.py`
**Tests:** gates 2–4. Tier 2 runs scripted meta-orders (sizes 0.1–30 % of ADV, durations 1–20 days) in CI.

### T6.22 — Domain randomisation and hidden state
**Depends:** T6.06 · **Size:** S · **Files:** `src/marketsim/scenarios/randomise.py`, `config/world.yaml`, `tests/unit/scenarios/test_randomise.py`
**Tests:** sampled overrides within ranges, logged in the replay header; `observe()` leaks no hidden field (schema whitelist).

### T6.23 — Market performance
**Depends:** T6.21 · **Size:** S · **Files:** `scripts/bench.py`, `tests/integration/test_perf_market.py`
**Tests:** gate 8; profile report in `claude/plan/reports/`.
