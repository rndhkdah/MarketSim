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
| `GB_BILL`, `GB_NOTE`, `GB_BOND` | government bond buckets ≈ 3m / 3y / 10y (§6.11); `GB_BOND` duration 7 = `sectors.yaml: bond_index` | auctions (primary) + engine MM (secondary) |
| `CORP_POOL` | pooled corporate bonds ≈ 5y — the only corporate bond, so every firm borrows on the same terms (D14) | tap auctions + engine MM |
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
**Corporate debt:** the common rate (D14) — floating loans at `r + s_t`, pooled bonds at the matching government
yield `+ s_t` (§6.11); no issuer-specific spread. **Commodities:** cell reference price + carry.
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

## 6.11 Bond market (D12)

**Instruments — fungible decaying-coupon buckets**, one instrument per bucket instead of one per issue. Each month a unit of
face pays the coupon `κ/12` and redeems `δ/12` of the remaining face; the rest rolls on. All units of a bucket are identical
for ever (fixed κ, fixed δ), so issuance is simply more units sold at the market price.

| bucket | issuer | decay δ / yr | average life | duration at 4.2 % | role |
|---|---|---|---|---|---|
| `GB_BILL` | `GOVT` | 4.0 | 3 months | 0.25 y | cash management, anchored on the policy rate |
| `GB_NOTE` | `GOVT` | 1/3 | 3 years | 2.7 y | belly of the curve |
| `GB_BOND` | `GOVT` | 0.10 | 10 years | 7.1 y (= `bond_index.duration`) | benchmark; long yield behind equity discount rates |
| `CORP_POOL` | `CORPPOOL` | 0.20 | 5 years | 4.2 y | all corporate bond funding (D14) |

Arithmetic (checked): at the bucket's flat monthly yield `y_m`, price per unit of remaining face `P = (κ_m + δ_m)/(y_m + δ_m)`;
duration `(1 + y_m)/(y_m + δ_m)` months; monthly holding return `(κ_m + δ_m + (1 − δ_m)·P_t)/P_{t−1} − 1`; +35bp on `GB_BOND`
→ −2.4 %. κ is fixed at the steady-state yield, so `P = 1` at baseline and the Phase-2 steady state stays exact. Positions are
units of remaining face; holder and issuer are valued at the same market price (the instrument still sums to zero); the fiscal
rule and debt/GDP use face value. Coupons and redemptions are postings (`interest_bonds`, `bond_redeem`); price changes are
revaluations — never income.

**Fair value and term premium.** `y_b = Σ_k ω_bk·E[r_{t+k}] + tp_b` (ω = the bucket's cash-flow weights; expected path from
§6.3, including the committee's published projection path and forward guidance × credibility, `02-…` §2.14.5).
`tp_b = tp0_b + (D_b/7)·[κ_debt·(b − b*) − κ_qe·(h_cb − h_cb0)] − λ_fq·z_risk + noise`, `b` = face debt / annual GDP, `h_cb` =
central-bank holdings / GDP. Indicative: κ_debt 0.03 (≈ 3bp on the 10-year per point of debt/GDP), κ_qe 0.05 (purchases of
10 % of GDP ≈ −50bp), λ_fq flight-to-quality loading — all flagged for calibration (T8.04). Price = fair value × exp(ξ_b) with
ξ from the §6.4 impact kernel (agents, auctions, central bank). **No sovereign default in v1:** fiscal stress shows up as term
premium → higher discount rates and cost of capital (crowding out).

**Primary market — the debt-management office** (a `GOVT` lever set, `02-…` §2.13; autopilot below). Financing need = deficit +
redemptions + deposit-buffer top-up, split 20 / 40 / 40 across buckets. Sizes are announced (news) 5 ticks ahead; auctions on
day 10 of each month. **Uniform-price auction:** agents bid (yield, quantity) through the API; NPC investors (households via
`MM`, `BANKSYS`, INSURANCE, `ROW`) bid the elastic schedule `Q_npc(y) = Q0·exp(η_a·(y − y_fair)·100)`; the stop-out yield clears
the size and all winners pay the stop-out price. Published: bid-to-cover and the tail against the pre-auction secondary yield.
The NPC schedule is unbounded in yield, so an auction always clears — the government is never cash-constrained, it pays for
size with yield. Absorbed supply feeds the impact kernel (auction concession) and, persistently, `tp` through `b`.

**Secondary market.** Engine MM venue (§6.5) for all four instruments; agents trade with impact; ADV ∝ outstanding face. NPC
holders rebalance slowly toward target portfolio shares — background flow. `ROW` holdings can be shocked by events (foreign
sell-off).

**Central-bank operations** (levers of `02-…` §2.13). QE / QT and open-market operations trade on the secondary venue **with
impact**. A purchase of x: bonds `MM → CB`; reserves created `(CB, RES, −x), (BANKSYS, RES, +x)`; seller paid
`(BANKSYS, DEP, −x), (MM, DEP, +x)`. Two effects: flow (impact) and stock (`h_cb` in `tp`). Coupon income is remitted to
`GOVT`. The central bank may also buy `CORP_POOL` (credit easing).

**Corporate bond pool — how bonds coexist with the common borrowing rate (D14).** `CORPPOOL` sells `CORP_POOL` units to
investors (weekly tap, same auction code) and on-lends to firms through `CLOAN` with the same decay at `y_match + s_t`
(`y_match` = government yield of equal duration, interpolated NOTE / BOND). At any moment every firm faces the same menu:
floating bank loan at `r + s_t` or fixed pool funding at `y_match + s_t`. NPC cells use `corp_funding_mix`; agent firms
choose. Defaults write down `CLOAN` → pool NAV per unit falls → holders share the loss pro rata. Fair pool spread =
`02-…` §2.7's `s_t` + `λ_s·z_risk`; the **borrowing** spread is that fair spread plus the 21-day mean of the ξ-implied spread
deviation — so agents dumping corporate bonds raise everybody's funding cost, equally.

**Holders at opening** (`bonds.yaml`): INSURANCE float 60 % `GB_BOND`, 25 % `CORP_POOL`, 15 % `GB_NOTE`; `BANKSYS` liquidity
book in `GB_BILL` / `GB_NOTE`; `CB` holds bills and notes equal to reserves; `ROW` 10 % of government debt; households (through
`MM`) the rest. With `bonds.pricing: market`, INSURANCE and BANKS mark-to-market **emerges from these holdings** — the
`bond_mtm_duration` and `float_rate_beta` overrides of §6.3 are switched off (no double counting).

**Additional gate items (continue §6.1):**

9. `bonds.pricing: par` reproduces Phases 2–5 bitwise; switching to `market` keeps the steady state exact.
10. Policy +100bp (4q): `GB_BOND` yield +35bp (±10), price −2.4 % (±0.7); `GB_BILL` price move < 0.3 %; the curve flattens.
11. The regime-flip test (gate 2) uses actual `GB_BOND` total returns.
12. Auctions: bid-to-cover > 1 and |tail| < 5bp in calm conditions; doubling the size raises the stop-out yield; winning bids
    settle on the ledger; `GOVT` deposits never negative.
13. QE of 10 % of GDP over 12 months lowers the `GB_BOND` yield by 30–80bp and raises reserves one-for-one; QT reverses it; SFC holds.
14. A bond-financed fiscal expansion raises `tp` and lowers capex relative to the same expansion under QE; the QE-financed
    variant shows higher inflation.
15. A +100bp parallel yield rise lowers INSURANCE equity by holdings × duration (from the ledger) and BANKS equity by less.
16. A large default lowers pool NAV and raises the borrowing rate of **all** firms by the same amount.

---

## 6.12 Task cards

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

### T6.24 — Bond buckets: instruments, arithmetic, par ↔ market switch (D12)
**Depends:** T6.01, T2.16 · **Size:** M · **Files:** `config/bonds.yaml`, `src/marketsim/pricing/bond_buckets.py`, `src/marketsim/ledger/opening.py`, `tests/unit/pricing/test_bond_buckets.py`
**Read first:** §6.11 instruments. **Build:** bucket definitions, price / duration / return functions, monthly coupon and
redemption postings, revaluation accounts, `bonds.pricing` switch. **Tests:** formulas at three yields; `GB_BOND` duration
7.07 y at 4.2 %; revaluation never appears as income; gate 9.

### T6.25 — Bucket fair yields and term premium
**Depends:** T6.03, T6.24 · **Size:** M · **Files:** `src/marketsim/pricing/curve.py`, `tests/unit/pricing/test_term_premium.py`
**Tests:** gate 10; `tp` rises with debt/GDP and falls with central-bank holdings, scaled by duration; forward guidance moves
the expected path in proportion to `credibility`; flight to quality lowers yields when `z_risk` rises.

### T6.26 — Debt-management office and auctions
**Depends:** T6.25, T2.28 · **Size:** M · **Files:** `src/marketsim/market/auction.py`, `src/marketsim/real/policy/debt_management.py`, `tests/market/test_auctions.py`
**Read first:** §6.11 primary market. **Build:** financing-need calculation, issuance-mix lever with autopilot, calendar and
announcements as queue items, uniform-price clearing with the NPC schedule, statistics, buybacks, settlement.
**Tests:** gate 12; a hand-computed clearing example; losing bids pay nothing; announcement precedes auction by 5 ticks.

### T6.27 — Secondary bond market and NPC holders
**Depends:** T6.24, T6.08, T6.09 · **Size:** M · **Files:** `src/marketsim/market/bond_market.py`, `tests/market/test_bond_secondary.py`
**Tests:** agent trades move yields through the impact kernel and revert; ADV ∝ outstanding face; NPC rebalancing converges
to target shares; a `ROW` sell-off event raises yields; gate 11.

### T6.28 — Central-bank operations: QE, QT, open-market operations
**Depends:** T6.27, T2.29 · **Size:** M · **Files:** `src/marketsim/real/policy/cb_operations.py`, `tests/market/test_cb_operations.py`
**Read first:** §6.11 central-bank operations. **Tests:** the four postings of a purchase by hand; gates 13 and 14; coupon
remittance to `GOVT`; purchases of `CORP_POOL` lower the common borrowing spread.

### T6.29 — Corporate bond pool
**Depends:** T6.24, T2.30, T5.10 · **Size:** M · **Files:** `src/marketsim/market/corp_pool.py`, `src/marketsim/firms/financing.py`, `tests/market/test_corp_pool.py`
**Read first:** §6.11 corporate bond pool, AGENTS rule 16. **Tests:** gate 16; two firms with different ratings get identical
terms on the same tick; NAV accounting through a default; tap issuance settles; selling pressure on `CORP_POOL` raises every
firm's borrowing rate equally.

### T6.30 — Financial-sector bond holdings and mark-to-market
**Depends:** T6.27 · **Size:** S · **Files:** `src/marketsim/real/banks.py`, `src/marketsim/pricing/fundamentals.py`, `tests/validation/test_financials_mtm.py`
**Tests:** gate 15; the §6.3 overrides are off in market mode (assert no double counting); T6.05's beta checks still pass —
BANKS remains the only rate-positive sector.

### T6.31 — Bond-market validation report
**Depends:** T6.26, T6.28, T6.29, T6.30 · **Size:** S · **Files:** `tests/validation/test_bond_market.py`, `claude/plan/reports/bond-market-validation.md`
**Tests:** gates 9–16 collected; Tier-2 impact test (§6.1 gate 3) repeated on `GB_BOND`; report the measured κ_debt and κ_qe effects.

### T6.32 — Policy surprises and announcement effects
**Depends:** T6.25, T2.32 · **Size:** S · **Files:** `src/marketsim/pricing/policy_surprise.py`, `tests/market/test_announcement_effects.py`
**Read first:** `02-…` §2.14.5, §6.3. **Build:** market-implied expected policy path from the published projection and
guidance; surprise = decision − expectation on announcement day, fed to `ξ` for bonds and equities.
**Tests:** a fully anticipated move barely moves prices; a surprise hike raises yields and lowers equities on the day,
with the front bucket moving most and `GB_BOND` least in yield terms; the surprise series is unforecastable from public
data (regression R² < 0.2), so Fed-watching is learnable but not free.
