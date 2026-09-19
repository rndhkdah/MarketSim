# Phase 2 — Ledger + dynamic real economy

**Goal.** Step the real economy forward in time on a stock-flow-consistent ledger: production with capacity,
inventories and backlog; input rationing; cost-plus prices with bounded steps; labour and wages; the capex loop that
makes the static table cycle; households, government, central bank, ROW; then banks and the credit/collateral edges.

**Provenance.** §2.3–§2.6 and the defaults in §2.11 were verified numerically against the real Layer-1 config
(`reference/dynamic_core_prototype.py`; results in master plan §13). §2.2 (ledger), §2.7 (banks/credit), §2.8 (typed
edges) and the catastrophe / risk-appetite shocks are **design-only**: each is switched on behind a config flag and
must leave the §2.12 suite green.

**Build order matters:** ledger → steady state → real blocks → settlement on the ledger (SFC green) → *only then*
banks, credit and collateral edges.

## 2.1 Gate

1. No-shock run, 360 months: `max |x/x0 − 1| < 1e-9` at π\* ∈ {0, 0.02}; 12-month inflation = π\* (±1e-9); SFC residuals < 1e-8 × monthly GDP.
2. Production-side GDP = expenditure-side GDP every month (±1e-9 relative).
3. Sign-restriction IRFs for all seven primitive shocks (+ ROW) pass on the windows in §2.12.
4. Monetary +100bp: hump-shaped, trough month 9–24, depth 0.15–0.60 %, rebound ratio < 0.6, troughs AUTOS < CONSTRUCT < CAPGOODS.
5. 100-year stochastic runs × 3 seeds bounded (§2.12); volatility ranking test passes.
6. With banks + credit + collateral edges on: gates 1–5 still pass, and the credit-crunch test passes.
7. **Policy authorities (D13):** with `GOVT` and `CENBANK` on autopilot the run is bitwise identical to gates 1–6; the lever
   tests and the extreme-policy boundedness test of §2.13 pass.
8. **One borrowing rate (D14):** every firm's loan rate is identical at every tick in `credit.pricing: uniform`.
9. **Monetary framework (D15):** parity with the simple rule under the parity settings (bitwise); steady state exact with
   the full framework on; §2.14.7 validation passes; 3–6 rate moves a year averaging 20–40bp over 100 years.

## 2.2 Ledger (L0)

**Positions.** `pos[entity, instrument]`, signed: + asset, − liability. Financial instruments: `DEP` (liability of
`BANKSYS`), `LOAN`, government bonds `GB_BILL`, `GB_NOTE`, `GB_BOND` (liabilities of `GOVT`), pooled corporate bonds `CORP_POOL`
(liability of the vehicle `CORPPOOL`, which on-lends to firms through `CLOAN`), `RES` (liability of `CB`), `EQ:<issuer>`. Real assets (`CAPITAL`, `INVENTORY`,
`HOUSING`) are one-sided and valued at replacement cost. Net worth is the balancing item, never stored.

**Entities (Phase 2).** `HH:<r>`, `NPC:<r>:<SECTOR>`, `BANKSYS` (financial balance sheet of sector BANKS), `GOVT`,
`CB`, `ROW`, `MM` (created now, used from Phase 6). Later phases only *register* `FIRM:<id>` and `AGENT:<id>`.

**Posting.** `Tx(tick, tag, entries, memo)`, `Entry(entity, instrument, amount)`. Rule: within a Tx, amounts sum to
zero per financial instrument. A payment of `x` from A to B is `(A, DEP, −x), (B, DEP, +x)`. Because `BANKSYS` holds
the negative `DEP` position, paying the bank shrinks its liability and raises its net worth automatically — no
special case. New loan: `(firm, DEP, +x), (BANKSYS, DEP, −x), (BANKSYS, LOAN, +x), (firm, LOAN, −x)` — endogenous
money. Write-off: `(BANKSYS, LOAN, −x), (firm, LOAN, +x)`, tag `loan_writeoff`. NPC-mass flows are posted netted per
`(payer, payee, tag)` per month. Storage is columnar numpy; full `Tx` objects only when `ledger.debug_journal: true`.

**Flow tags** (rows of the transaction-flow matrix): `consumption, vat, govt_purchases, investment, residential,
exports, imports, intermediate, wages, transfers, income_tax, corp_tax, interest_loans, interest_deposits,
interest_bonds, interest_reserves, cb_remittance, dividends, loan_new, loan_repay, loan_writeoff, bond_issue,
bond_redeem, equity_issue, equity_trade, fees, transaction_tax, insurance_claims, capital_transfer`.

**Assertions** (`sfc.assert_consistent(ledger, period)`): (a) each Tx balanced; (b) Σ_entities pos[·, k] = 0 for every
financial instrument; (c) each TFM row sums to zero across entities; (d) for each entity, current + capital account
balance = Δ net financial assets. Failures raise `SFCError` listing the offending tag/entity/amount.

**Opening balance sheet** (`ledger.yaml`): firm loans `L_j = nd_ebitda_j × 12 × EBITDA0_j`; government debt
`B = 0.60 × 12 × GDP0`; `BANKSYS`: equity = 0.125 × loans, assets = `asset_leverage` × equity, the non-loan assets
split `RES` = 8 % of deposits, rest government bonds; `CB` holds government bonds = `RES`; households hold all deposits,
the remaining bonds and all equity (at book value in Phase 2). Consumption-relevant wealth `W = DEP_hh + bonds_hh`.
Stage 1 (T2.16–T2.17) runs with `banks.mode: passthrough` (zero margins, no capital dynamics — reproduces the
prototype, where households hold firm debt directly); stage 2 (T2.19) switches to `banks.mode: full`.

**Bonds from day one, priced at par until Phase 6 (D12).** Government debt sits in three fungible maturity buckets —
`GB_BILL` ≈ 3 months, `GB_NOTE` ≈ 3 years, `GB_BOND` ≈ 10 years, opening mix 20 / 40 / 40 % — and half of firm debt is
funded through the corporate bond pool (`corp_funding_mix: 0.5`), the rest by bank loans. With `bonds.pricing: par` every
bucket trades at 1 and its coupon floats with the policy rate (+ the global corporate spread for `CORP_POOL`) — exactly the
verified prototype. `06-…` §6.11 switches to `bonds.pricing: market` (fixed coupons, yields, auctions, secondary trading,
central-bank operations) without changing the ledger layout.

## 2.3 Steady-state initialiser (closed form, verified)

Notation: `d0` baseline final demand (cr/month, from `fd_weights × scale`), `λ` monthly leak, `cover` target months
of finished-goods stock, `u*` = `util_target`, `G = exp(π*/12)`.

```
cover_j   = cover_scale · 3 · max(inv_lag_q_j, 1)        (stock mode, else 0)
Λ         = diag(1 + λ_j · cover_j)
x0        = (I − Λ A)⁻¹ Λ d0            # output incl. replacement of leaked stock
s0        = x0 / (1 + λ · cover)        # sales
m_j       ∝ import_prior_j, scaled so Σ m_j x0_j = Σ exports0     # non-competing imports carved out of value added
va_j      = 1 − mu_j − m_j ;  ell_j = wage_share_j · va_j ;  gos_j = (1 − wage_share_j) · va_j
markup_j  = 1 / (Σ_i a_ij + ell_j + m_j)                 # so that p* = 1 at w = p_imp = 1
K0        = x0 / u* ;  starts0 = (δ/12) · K0 ;  seed capacity and spending chains with starts0, v·starts0
ρ_K       = δ · Σ(gos · x0) / I_bus0 ;  v_j = 12 · gos_j · u*_j / ρ_K      # cost of one unit of monthly capacity
I_bus0    = (1 − res_share) · I0 ;  res0 = res_share · I0 ;  route_bus ∝ fd_INVESTMENT·I0 − res0·e_CONSTRUCT
EBITDA0_j = s0_j − (mu_j + ell_j + m_j) · x0_j           # revenue on sales, cost on output (spoilage is a cost)
debt_j    = nd_ebitda_j · 12 · EBITDA0_j ;  spread_j = s0   # ONE rate for all firms (D14); risk_based mode: s0·max(nd_j,0.2)/2.5
int_j     = (r0 + spread_j) · debt_j / 12 / G            # interest on LAST month's stock, in this month's prices
grow      = (G − 1) / G                                  # real net issuance that keeps stock/price constant
tax_j     = τ_c · max(EBITDA0_j − int_j − dep_j, 0) ,  dep_j = (δ/12) v_j K0_j
payout_j  = (prof_j − capex0_j + grow · debt_j) / prof_j ,  prof_j = EBITDA0_j − int_j − tax_j , capex0_j = dep_j
B         = debt_to_gdp · 12 · GDP0 ;  GDP0 = Σ (s0 − (mu + m) x0)         # net of spoilage
W         = consumption-relevant household wealth from the opening balance sheet (prototype: B + Σ debt)
YD0       = C0·(1 + vat) + res0 + grow · W
α2        = (C0·(1 + vat) − α1 · YD0) / W        (must be > 0)
τ_y       = 1 − YD0 / pretax0 ,  pretax0 = wages0 + dividends0 + interest received + transfers0
n0_j      = ell_j · x0_j (at w = 1) ;  LF = Σ n0 / (1 − U*)
r0 = r_neutral + π* ; π_e = π* ; CPI and core histories seeded on the π* path (13 points, G^(k−12))
backlog0  = (τ_ob − 1) · x0 for order mode ;  input stocks0 = n_in · a_ij · x0_j for storable inputs ;  inv0 = cover · s0
```

Observed at the seed calibration: payout 0.50–0.61 (V3's firms retain 25–50 % — same ballpark), α2 ≈ 0.0064–0.0075
per month, τ_y ≈ 0.136. `grow`, `1/G` and the seeded histories are what make π\* = 2 % exact; do not "simplify" them.

## 2.4 Monthly step (order is part of the spec)

**R1 Expectations.** `s_e += (sales − s_e)/τ_e`.

**R2 Final demand.** `Pc = Σ θ_i p_i` (θ = baseline household basket). `y = (YD_e / Pc) / YD0`.
`rate_gap = Erlang2(6m)[(r − π_e − r_n)·100 + Δspread_hh·100]` (in 100bp).
`C_nom = (α1·YD_e + α2·W)·exp(z_dem)`.
`z_i = θ_i · y^(η_i − 1) · (p_i/Pc)^(1 + ε_i) · exp(dem_rate_semi_i/100 · rate_gap) · edge_shifter_i` ;
`c_i = C_nom · (Σz/Σθ)^ζ · (z_i/Σz) / (p_i (1 + vat))`, ζ = `rate_budget_passthrough` = 0.5.
`g_i = G_real_i · exp(z_fisc)` ; `ex_i = X0_i · exp(z_row) · (p_i/p_imp)^(−ε_x)`.
`dem_rate_semi` acts on the household component only (master plan §14).

**R3 Investment.** `u_s` = smoothed `x/K`. Start rate per §2.6 → `starts = K · rate/12`.
`spend_real = SpendChain.push(v · starts)` (Erlang(3), mean `min(3·capex.lag_q, build months)`);
`K ← (1 − δ/12)·K + PipeChain.push(starts)` (Erlang(3), mean `3·build_lag_q`); `K_eff = K · exp(z_sup)`.
Residential: `res = Erlang2(3m)[res0 · y^el · exp(res_rate_semi/100 · rate_gap) · exp(z_dem)]`, added to CONSTRUCT
final demand and paid by households. Investment goods: `route_bus · Σ spend_real` (+ `res` on CONSTRUCT).

**R4 Plans and caps.** stock mode: `plan = s_e + λ·inv + (cover·s_e − inv)/τ_inv + backlog/τ_b`; others `plan = s_e`;
clip to `[0, K_eff]`. Labour cap: `n_var = n − fc·ell·K·u*`; `lab_cap = overtime · n_var / ((1 − fc)·ell·exp(−z_sup))`.
Input cap: storable critical inputs `min_i S_in[i,j]/a_ij`; flow-critical inputs `min_i fill_prev_i`.
`x_goods = min(plan, K_eff, lab_cap, in_cap) · flow_crit`. An input is *critical* when its supplier ∈
`critical_input.suppliers` and its share of the buyer's intermediate spend ≥ 5 %.

**R5 Orders — all on buyers' plans, so nothing is solved simultaneously.** `O_ij = a_ij·plan_j + [storable]
(n_in·a_ij·plan_j − S_in[i,j])/τ_in`, floored at 0. `new_orders_i = Σ_j O_ij + final_i`;
`D_i = new_orders_i + backlog_i` (stock mode).

**R6 Supply, rationing, deliveries.**
- stock: `avail = x + (1−λ)·inv`; `fill = min(1, avail/D)`; `sales = fill·D`; `inv' = avail − sales`; `backlog' = (D − sales)(1 − backlog_loss)`.
- order: `ob = backlog + new_orders`; `x = min(ob/τ_ob, cap·flow_crit)`; `sales = x`; `backlog' = ob − x`; deliveries pro-rata on the book (`x/ob`).
- flow: `x = min(D, cap·flow_crit)`; unmet demand is lost.
- Pro-rata rationing; `S_in' = S_in + deliveries − a_ij·x_j` (≥ 0 by construction); `fill_prev = fill`.
This is the two-regime handoff borrowed from V3, with the bound in the right place: price carries imbalances through
bounded *steps*; beyond that, rationing, backlog and input caps take over.

**R7 Prices.** `nuc_j = Σ_i a_ij p_i + w·ell_j·exp(−z_sup) + m_j·p_imp`;
`tight_j = exp(κ_u (x_j/K_j − u*_j)) · (cover*_j / max(cover_j, cover_floor))^γ` (second factor stock mode only);
`lp* = ln(markup_j · nuc_j · tight_j) + π_e/12 + z_cost_j`.
Two-speed pass-through with drift `d = π_e/12`: `pf += d + a_f (lp* − pf − d)`, `a_f = 1/(1 + fast_mean_m)`;
`ps += d + a_s (lp* − ps − d)`, `a_s = 1/(1 + 3·pass_lag_q)`; `ln p_new = pt·pf + (1 − pt)·ps`;
`|Δ ln p| ≤ step_max_month` (0.15). **No level cap.** Then `p_imp ← p_imp · G · exp(Δz_imp)`.

**R8 Labour and wages.** `n* = ell·exp(−z_sup)·(fc·K·u* + (1 − fc)·plan)`; adjust with `τ_hire` = 3 / `τ_fire` = 6
months; cap Σn ≤ 0.995·LF; `U = 1 − Σn/LF`; `dw = π_e + phillips_slope·(U* − U)`, divided by `wage_stickiness_q` when
negative; `w ← w·exp(dw/12)`.

**R9 Incomes and settlement (every line is a ledger posting).** Revenue `p·sales`; input purchases on *deliveries*;
wages; imports `p_imp·m·x`; interest on beginning-of-month stocks; `EBITDA = revenue − cost of inputs used − wages −
imports`; tax depreciation `(δ/12)·v·K·p_I`; corporate tax; `prof_s`, `eb_s` drift-compensated smoothers (6 / 12 months);
`div_j = max(0, payout_j·max(prof_s,0) − κ_D/12·(debt_j·g − nd_j·12·max(eb_s,0)))`; `Δdebt_j = p_I·spend_real_j −
(profit_j − div_j)` (**never clipped**; a negative balance is a deposit); households: `pretax = wages + dividends +
interest + transfers`, `YD = (1 − τ_y_eff)·pretax`, `W += YD − C_spent − p_CONSTRUCT·res`, `YD_e` drift-compensated
(12 months). Government per §2.5.

**R10 Expectations and policy.** `sales_ma`, `g_now = 12·ln(sales/sales_ma)/τ_g`, `g_e += (g_now − g_e)/τ_g`; CPI, core
CPI (ex ENERGY, AGRIFOOD), π12, π3; `π_e += (infl_anchor·π* + (1 − infl_anchor)·π12 − π_e)/τ_infl`; real GDP
`= Σ(x − λ·inv_prev − (mu + m)·x)`; output gap vs GDP0; policy per §2.5.

## 2.5 Policy blocks

**Fiscal.** `G_real` fixed in real terms (× `exp(z_fisc)`); transfers = `benefit_replacement · w · (LF − Σn)`;
income tax rate `τ_y_target = τ_y0 + κ_B·(B·g/GDP_nom_annual − debt_to_gdp)`, clipped to [0, 0.6], smoothed 12
months; corporate tax `τ_c`; VAT (default 0). Deficit financed by `bond_issue` absorbed by households at the policy
rate (term structure arrives in Phase 6). Without the debt rule `r > g` snowballs — verified. The config value
`automatic_stabiliser: −0.35` is a **validation target**: simulated Δ(deficit/GDP) per point of output gap must land
in [0.2, 0.5]. Discretionary fiscal *events* act through an Erlang(2) lag of `3·discretionary_lag_q` months.

**Central bank (skeleton; the full framework is §2.14).** Quarterly: `target = r_n + π* + φ_π(π_pol − π*) + φ_y·gap`, `π_pol = ½π3 + ½π12` (optionally a
`core_weight` blend); `r_rule ← ρ·r_rule + (1 − ρ)·target`; `r = max(elb, r_rule + z_mon)`. Parameters from
`edges.yaml: policy.taylor`. `CB` remits profits to `GOVT`. Policy meetings are scheduled queue items.

Both blocks are the **autopilot** of a policy authority; §2.13 turns their settings into levers that a script or an agent
can operate, and §2.14 replaces the central-bank skeleton above with a committee, a calendar, a dual mandate and the
published data it actually sees.

## 2.6 Capex unit contract (fixes B4 — verified)

```
rate_j = δ + unit_scale · [ φ · 100·g_exp_j  +  ψ · 100·((u_j − u*_j) − sl_j)  −  χ · cc_gap_j  +  q_term_j ]
rate_j = clip(rate_j, 0, start_rate_cap_mult · δ)                    # annual starts as a fraction of K
g_exp_j  = (1 − anchor_growth) · g_e_j
sl_j     = supply_line_weight · u*_j · (pipe_j − pipe0_j·K_j/K0_j) / K_j     # capacity already under construction
cc_gap   = (r − π_e − r_n)·100 + Δs·100 + Δerp·100      # in 100bp; identical for every firm (D14): s = the single global
                                                        # corporate spread. risk_based mode: (nd_j/2.5)·Δspread_j instead of Δs
q_term_j = q_scale · q_tobin · 100 · clip(ln Q_j smoothed 12m, −q_clip, +q_clip)   # Q ≡ 1 until Phase 6
```

`unit_scale = 0.01` means **φ, ψ, χ are percentage points of the capital stock per year**: +100bp cost of capital
lowers the annual start rate by 0.45 pp (from δ = 6 pp). Under the common borrowing rate (D14) the cost-of-capital
gap is the same for every firm — re-verified in the prototype (`credit_pricing="uniform"`: exact steady state, all sign
tests, monetary trough −0.29 % at month 16, rebound 0.41, AUTOS m10 < CONSTRUCT m17 < CAPGOODS m19, 100-year runs bounded).
`risk_based` mode scales the spread term by `nd_j/2.5` (`edges.yaml: spread_scaling`). Verified facts to preserve: `supply_line_weight = 0` is unstable at every accelerator
scale; `anchor_growth`, the 12-month income smoothing and the residential block are all needed for damping.

## 2.7 Banks, credit and collateral — after SFC is green (design-only)

`banks.mode: full`: loan rate `r + spread_j`; deposit rate `max(r − dep_margin, 0)` (`dep_margin` 0.005 — margin
compresses near the ELB, so bank earnings rise with the short rate: the NIM story emerges, it is not asserted);
bonds and reserves earn `r`. Expected loss `ll_j = ll0·(nd_j/2.5)·exp(κ_ll·(ICR0_j/ICR_j − 1))`, `ll0` = 0.004/yr,
capped at 5× base, posted as `loan_writeoff`; the initialiser includes baseline write-offs in `payout_j` and bank
profit so the steady state stays exact. Bank payout `= clip((c − 0.085)/(0.125 − 0.085), 0, 1)` of profit;
`c = equity / RWA`, RWA = loans.

- **Gate:** `g = logistic_gate(c)` (= 1 at baseline, T0.14). Lending-capacity index `Λ = g · Λ_coll`.
- **Collateral:** `Λ_coll = asymmetric_gate(ln(V_RE / V_RE,trend))` with elasticity 0.55, `up` 0.6, `down` 1.8, then
  Erlang(2) 2q. Level-based, so a round trip leaves no ratchet. `V_RE` from §2.10.
- **One global corporate spread (price channel, D14):** `s_t = s0 + s_gate·(1 − g) + s_loss·(ll̄ − ll̄0)`, `s0` 0.015,
  `s_gate` 0.04, `s_loss` 2.0, `ll̄` = economy-wide loss rate. **Every firm — NPC cell or agent firm, any sector, any region,
  any rating — borrows at `r + s_t` (floating bank loan) or at the matching government yield `+ s_t` (fixed, through the
  corporate bond pool, `06-…` §6.11).** Leverage is not priced; it is disciplined by quantity: leverage caps, collateral
  limits and the rationing exponents below. Default losses are mutualised (banks, pool) and return to everyone as a higher
  `s_t`. `credit.pricing: risk_based` restores `spread_j = base_j + (nd_j/2.5)·[…]`. The household spread moves with `s_t`.
  Spreads enter `cc_gap`, the residential rate gap and R2.
- **Rationing (quantity channel):** typed credit edges are exponents on lagged lending capacity `Λ̃` (Erlang per edge):
  capex starts of all sectors × `Λ̃^0.35`; CONSTRUCT capex and residential × `Λ̃^0.90`; household credit-financed
  demand (AUTOS, DISCRET) × `Λ̃^0.60`. At baseline `Λ̃ = 1` → no effect; at the gate midpoint ≈ ×0.78 / ×0.54 / ×0.66.
  This is the non-linearity that turns a drawdown into a credit crunch.
- **Credit-crunch test:** write off 40 % of bank equity at month 12 → gate < 0.6 within a quarter; GDP trough ≥ 1.3×
  deeper than with `credit.gate.enabled: false`; capital ratio back above 0.11 within 8 years; |gap| < 0.5 % by
  year 12; SFC holds throughout.

## 2.8 Typed edges — substitution and complement (design-only)

`TypedEdge{src, dst, channel, elasticity, lag (k, mean_m), sign, gate, saturation}`. For `substitution` /
`complement` entries: `shifter_dst = exp(Σ_e clip(elasticity_e · Smooth_e[ln(p_src/Pc)], ±saturation))`, saturation
0.3, applied to **all final-demand components of dst** (household, investment goods, exports) — never to intermediate
orders (technology). Physical shortages are already handled by critical inputs in R4, so SEMIS→AUTOS is not counted
twice. Optional T2.26 adds CES on intermediates (σ = 0 recovers Leontief).

## 2.9 Primitive shocks

AR(1) monthly states, `ρ = exp(−1/(3·persistence_q))`, persistence from `edges.yaml: shocks`.

| shock | state | acts on | + shock gives |
|---|---|---|---|
| demand | `z_dem` | household nominal consumption and residential × exp(z); optional `animal_spirits_weight`·z added to `g_exp` (default 0) | output +, inflation + |
| supply | `z_sup[r,s]` | `K_eff = K·exp(z)`; labour requirement and unit labour cost × exp(−z) | output +, inflation − |
| cost_push | `z_cost[r,s]` | added to `lp*` (default targets ENERGY 1.0, AGRIFOOD 0.3) and to `p_imp` for the same goods | output −, inflation + |
| monetary | `z_mon` | residual on the policy rate, added after the committee's decision (§2.14) | output −, inflation − |
| risk_appetite | `z_risk` | `Δerp = z` into `cc_gap`, valuations (§2.10), `Δspread += 0.5·z` | output −, inflation ≈ 0 |
| fiscal | `z_fisc` | `G_real × exp(z)` | output +, inflation + |
| catastrophe | one-off | destroy share κ of `K` and inventories in a (region, sector) mask; INSURANCE pays `cat_exposure`-bounded claims, routed as demand to `claims_to` [CONSTRUCT, AUTOS, HEALTH]; INSURANCE equity falls | output −, inflation +, later reconstruction boom |
| (row) | `z_row`, `z_imp` | world-demand index on exports; import prices | — |

`ShockBus.inject(kind, magnitude, persistence_q=None, targets=None, weight=1.0)` is the only entry point (Phase 4 uses it).

## 2.10 AssetPriceProvider stub

`ln V_j = ln E^e_j + ln PE0_j − cf_duration_j · Δρ`, `E^e` = 12-month drift-compensated after-tax profit,
`Δρ = 0.35·(r − π_e − r_n) + z_risk`, `PE0` set so `Q_j = 1` at baseline. Serves `Q_j` (capex), `V_RE` (collateral).
Household equity wealth effect **off** in Phase 2. Same interface as the Phase-6 provider (`pricing/provider.py`).

## 2.11 `config/dynamics.yaml` (verified defaults)

```yaml
expectations: {tau_sales_m: 3, tau_growth_m: 12, anchor_growth: 0.75, infl_anchor: 0.6, tau_infl_m: 6}
production:
  mode: {ENERGY: stock, MATERIALS: stock, AGRIFOOD: stock, SEMIS: stock, AUTOS: stock, STAPLES: stock,
         DISCRET: stock, HEALTH: stock, CAPGOODS: order, CONSTRUCT: order, UTILITIES: flow, TRANSPORT: flow,
         SOFTWARE: flow, TELECOM: flow, BIZSVC: flow, BANKS: flow, INSURANCE: flow, REALESTATE: flow}
  cover_scale: 0.25          # finished-goods cover (months) = cover_scale * 3 * max(inv_lag_q, 1)
  tau_inv_mult: 1.0          # stock gap closed over max(2, mult * cover) months
  order_book_scale: 0.5      # order-book lead time (months) = scale * 3 * max(inv_lag_q, 1)
  input_cover_m: 1.0
  tau_input_m: 2.0
  backlog_loss: 0.10
  tau_backlog_m: 3.0
  overtime_cap: 1.10
  critical_input: {min_share: 0.05, suppliers: [ENERGY, UTILITIES, MATERIALS, AGRIFOOD, SEMIS, CAPGOODS, AUTOS, TRANSPORT]}
  leak_per_month: {AGRIFOOD: 0.015, STAPLES: 0.008, DISCRET: 0.008, SEMIS: 0.006, AUTOS: 0.006, HEALTH: 0.006,
                   MATERIALS: 0.004, ENERGY: 0.004}
prices: {kappa_util: 1.2, gamma_cover: 0.35, cover_floor: 0.15, step_max_month: 0.15, fast_mean_m: 1.0}
capex: {delta_annual: 0.06, start_rate_cap_mult: 3.0, tau_util_m: 1.0, q_clip: 0.5}   # phi, psi, chi, q, unit_scale,
                                                                                      # supply_line_weight live in edges.yaml
residential: {share_of_investment: 0.20, rate_semi: -4.0, income_elasticity: 1.0, lag: {k: 2, mean_m: 3}}
households: {alpha1: 0.70, tau_income_m: 12, rate_budget_passthrough: 0.5, rate_lag: {k: 2, mean_m: 6}}
labour: {tau_hire_m: 3, tau_fire_m: 6, u_star: 0.05, lf_cap: 0.995}
fiscal: {debt_to_gdp: 0.60, kappa_debt: 0.04, tau_tax_m: 12, tax_rate_bounds: [0.0, 0.6],
         benefit_replacement: 0.40, corp_tax: 0.20, vat: 0.0}
firms: {kappa_leverage: 0.03, tau_profit_m: 6, tau_ebitda_m: 12, base_spread: 0.015, spread_leverage_floor: 0.2}
row: {export_price_elasticity: 0.8,
      import_prior: {ENERGY: 0.15, MATERIALS: 0.10, SEMIS: 0.12, AUTOS: 0.08, CAPGOODS: 0.08, AGRIFOOD: 0.05,
                     STAPLES: 0.04, DISCRET: 0.03, default: 0.01}}
policy: {elb: 0.0, core_weight: 0.5}      # the full monetary framework lives in config/policy.yaml (§2.14.6)
banks: {mode: passthrough, dep_margin: 0.005, ll0: 0.004, kappa_ll: 1.0, ll_cap_mult: 5.0,
        capital_target: 0.125, reserves_to_deposits: 0.08}
credit: {pricing: uniform, s0: 0.015, corp_funding_mix: 0.5, gate_enabled: false, s_gate: 0.04, s_loss: 2.0,
         collateral_trend_years: 5}      # pricing: uniform (D14) | risk_based
shocks: {animal_spirits_weight: 0.0, cost_push_targets: {ENERGY: 1.0, AGRIFOOD: 0.3}}
```

Sensitivity facts (prototype): κ_u 0.6 → 1.2 cuts the monetary rebound ratio 0.85 → 0.33; `kappa_leverage` 0.10
under-damped, 0.30 explosive; `tau_income_m` 8 with α1 0.8 unstable; removing the residential block or the price
tightness term destabilises; leak ×0/×1/×3 all bounded.

## 2.12 Validation suite (tests/validation/test_dynamics_*.py)

IRF protocol: shock at month 1, AR(1) decay, 240 months, compare with the (exactly stationary) baseline.
`gap_t` = real GDP deviation; `lvl_t = ln CPI_t − π*·t/12`. **Price tests use the level**, because 12-month inflation
overshoots later by construction.

| shock (size, persistence) | output test | price test | prototype |
|---|---|---|---|
| demand (+2 %, 6q) | Σ gap[1..24] > 0 | mean lvl[18..36] > 0 | peak +1.3 % m4 |
| monetary (+100bp, 4q) | Σ gap[1..36] < 0 | mean lvl[18..36] < 0 | trough −0.30 % m16, rebound 0.44 |
| cost_push (ENERGY +30 %, 8q) | Σ gap[6..48] < 0 | mean lvl[6..24] > 0 | CPI +2.0 % @12m, GDP −2.7 % m22 |
| supply (−3 %, 12q) | Σ gap[1..48] < 0 | mean lvl[6..36] > 0 | trough ≈ −1 % m28 |
| fiscal (G +5 %, 8q) | Σ gap[1..24] > 0 | mean lvl[18..36] > 0 | peak +0.85 % m19 |
| row (exports −10 %, 6q) | Σ gap[1..24] < 0 | — | trough −0.6 % m6 |
| risk_appetite (+100bp ERP, 3q) | Σ gap[1..24] < 0 | not tested | design-only |
| catastrophe (5 % of K in CONSTRUCT+REALESTATE+AUTOS) | gap[1..6] < 0; CONSTRUCT output above baseline within 24m | mean lvl[3..18] > 0 | design-only |

Timing (monetary): GDP trough month ∈ [9, 24]; depth ∈ [0.15 %, 0.60 %]; `max(gap after trough, ≤120m)/|trough| < 0.6`;
sector trough months AUTOS < CONSTRUCT < CAPGOODS (prototype 11 < 17 < 20) — housing turns before capital goods.
Stochastic (slow): AR(1) shocks `z_dem` (ρ 0.90, σ 0.004), `z_sup` (0.95, 0.0015), `z_cost[ENERGY]` (0.93, 0.02);
1,200 months × 3 seeds at π\* = 2 %: all finite; |gap| < 10 %; U ∈ (1 %, 12 %); envelope(last 20y)/envelope(first
20y) < 2; SEMIS has the highest sd of yoy output, CONSTRUCT and CAPGOODS are in the top 5, and INSURANCE, TELECOM,
BANKS are in the bottom 6. Soft targets reported, not gated: sd(I)/sd(GDP) 3–4 (prototype ≈ 7), sd(C)/sd(GDP) < 1
(≈ 1.45), cost-push GDP effect (large).

## 2.13 Policy authorities — the government and the central bank as actors (D13)

`GOVT` (treasury + debt-management office) and `CENBANK` are **policy authorities**. Each owns a set of levers and every
lever has an autopilot — the rules of §2.5. Control per authority (`config/policy.yaml: control`): `autopilot` (default) ·
`scripted` (scenario files) · `agent` (an API client with the `policymaker` role: the player as finance minister or
governor, a policy-RL agent, or a fixed regime under evaluation). Same pattern as firms: a decision sets any subset of
levers, the rest stay on autopilot. Arbitration per lever: agent > scripted > event-set > autopilot. Phase-4 policy events
act *through* these levers.

**Government levers** — effective at the next month boundary after `legislative_lag_m`; discretionary spending also passes
the Erlang(2) implementation lag of `3·discretionary_lag_q` months:

| lever | what it does | hook |
|---|---|---|
| purchases | level of real G, its **sector composition** (the GOVT final-demand vector) and **regional allocation** — infrastructure → CONSTRUCT / CAPGOODS, defence → CAPGOODS, health → HEALTH | R2 `g_i`; tag `govt_purchases` |
| taxes | income-tax rate, corporate-tax rate, VAT, sector excise or subsidy wedge (energy subsidy, carbon tax), import tariff | price wedges in R2 / R7, taxes in R9; tags `income_tax`, `corp_tax`, `vat`, `excise`, `tariff` |
| transfers | benefit replacement rate; one-off transfers to households (by region, by tier after Phase 3); capex subsidy per sector (industrial policy — lowers the effective `v_s`) | tags `transfers`, `subsidy` |
| rescue | capital injection into `BANKSYS` or a firm (equity purchase), guarantees | tag `equity_issue`; bank capital → gate |
| fiscal rule | debt target `b*`, reaction speed κ_B, on / off — the autopilot's own parameters are levers | §2.5 |
| debt management | issuance mix across maturity buckets, auction sizes and calendar, buybacks | `06-…` §6.11 (par mode: mix only) |

**Central-bank levers:** the rate decision (override) or the rule's parameters (π\*, φ_π, φ_y, smoothing); forward guidance
(an announced path, weighted by `credibility` in the expected path of `06-…` §6.3); macroprudential settings — bank capital
requirement (moves the gate midpoint and the payout target), LTV cap (scales `Λ_coll`); lender-of-last-resort loans to
`BANKSYS`; asset purchases and sales (QE / QT) once bonds are market-priced (§6.11).

**Budget constraint.** `GOVT` pays from its deposit account; any shortfall is covered by bond issuance, so every policy is
financed on the ledger — never by fiat. **Guard rails** (`policy.yaml: limits`): lever ranges (tax rates ∈ [0, 0.6], tariff
≤ 0.5, ΔG ≤ 2 % of GDP per quarter, rate moves ≤ 200bp per meeting), announcement → implementation lags, and a `policymaker`
cannot hold a trading account in professional mode. Every decision is published as a `NewsItem` when announced.
Scoring hooks for policy agents: quadratic loss in inflation and the output gap + a debt penalty; a game "approval" index.

**Validation** (T2.31): autopilot parity (bitwise); bond-financed G +1 % of GDP for 8 quarters → cumulative output multiplier
∈ [0.5, 2.0], CPI level ↑; income-tax cut → output ↑; VAT +2 pp → one-off CPI-level step ≈ 2 pp × pass-through and no
permanent inflation; tariff +10 % → import prices and CPI ↑; bank recapitalisation reopens the gate in the credit-crunch
scenario; capital requirement +2 pp lowers lending capacity; adversarial policy (G +10 % of GDP, tax rates at their bounds,
rate pinned at the ELB for 10 years) → finite, SFC holds, inflation and debt move in the expected direction.

## 2.14 The monetary policy framework — how the central bank moves the rate (D15)

§2.5's Taylor rule is the skeleton. This section is the body: a committee that meets on a calendar, sees **published,
lagged, revised** data, weighs inflation against employment, moves in 25bp steps, and sometimes surprises the market.
It is the autopilot of the `CENBANK` authority (§2.13) — a scripted regime or a `policymaker` agent overrides any part.

### 2.14.1 Committee and calendar

Eight meetings a year (every ~6 weeks; `meetings_per_year`, 4 = the old quarterly rule). The decision is announced on the
**25bp grid** (`rate_step`) and only if the smoothed target has moved at least `deadband` from the current rate — the
internal target stays continuous, so gradualism is not lost to rounding. Verified over 100 years: the grid alone turns
8 meetings into ≈ 4.1 moves a year averaging 28bp, with no cost in inflation or unemployment volatility; a 50bp grid gives
2.2 moves of 50bp. Committee dispersion (`dispersion_bp`, default 0) adds a deterministic-per-seed draw around the rule,
which is what makes the decision forecastable but not exactly predictable — the market's "Fed watching" problem.

### 2.14.2 The reaction function

```
r*_t     = r_neutral + κ_r · g_trend                       # neutral rate tracks trend growth (60m smoother)
π_pol    = (1 − w_core)·(½π₃ + ½π₁₂) + w_core·(½core₃ + ½core₁₂)      # published vintages, lagged
target   = r*_t + π*                                        # nominal anchor
         + φ_π · (π_pol − π*)                                # price stability
         + φ_u · (U* − U)                                    # maximum employment (dual mandate)
         + φ_y · gap                                         # output gap (0 by default: U carries it)
         + φ_m · clip(plevel_gap, ±c)                        # makeup / average inflation targeting (OFF)
         − φ_f · FCI_tighten                                 # financial conditions (OFF until Phase 6)
         − risk_off                                          # risk management, §2.14.4
r_rule  ← ρ^(4/mpy) · r_rule + (1 − ρ^(4/mpy)) · target      # gradualism, per meeting
r_ann    = grid(r_rule) if |grid(r_rule) − r_ann| ≥ deadband else r_ann
r        = max(elb, r_ann + guidance_override + z_mon)
```

All inputs are **published vintages** (§4.3): CPI at a 1-month lag, unemployment 1 month, GDP one quarter with revisions.
The committee therefore acts on the data it can see, which is the honest version and also the source of policy error.

**Defaults and what the prototype showed** (100-year runs, π\* = 2 %, all bounded, steady state exact to 1e-14):

| term | default | evidence |
|---|---|---|
| `phi_u` (dual mandate) | **1.0** | with Okun ≈ 2 this equals the config's `phi_output_gap: 0.5`, so the anchor is unchanged. φ_u 0.5–1.0 behaves like the output-gap rule; **2.0 is too hot** (sd(π) 1.10 → 1.24, U min 0.8 %) |
| `phi_y` | 0.0 | the unemployment gap already carries the real side; set `phi_y_mult: 1` to use both |
| `w_core` (`core_weight`) | **0.5** | looking through relative-price shocks: pure-core reaction cut the cost-push GDP trough from −2.86 % to −2.56 % and CPI@12m from 2.01 % to 1.91 % |
| `meetings_per_year`, `rate_step`, `deadband` | **8, 25bp, 10bp** | ≈ 4.1 moves/yr at 28bp; volatility unchanged vs a continuous quarterly rule |
| `data_lag` | **CPI 1m, U 1m, GDP 1q** | cost of acting on stale data is real but small and monotone: sd(π) 1.08 (no lag) → 1.10 (1m) → 1.15 (2m) → 1.21 (3m) |
| `kappa_rstar` | **1.0** | mildly stabilising (sd(π) 1.10 → 1.08); makes the anchor move with trend growth instead of being frozen |
| `makeup` | **0.0** | see below |
| `risk_management`, `phi_fci`, `dispersion_bp` | 0 (shipped, off) | verified bounded when on |

### 2.14.3 Makeup strategies — a verified trap

Average-inflation targeting adds the *cumulative* price-level gap to the rule. An unbounded cumulative term is a pure
integrator inside a feedback loop, and it blew the economy up: `makeup: 0.2` with no leak and no clip gave a −93 % output
gap and 73 % unemployment over 100 years. The same failure mode as the undamped inventory stock in
`vic3-demand-model.md` §4 — **integrators in feedback loops must leak**. With `makeup_decay: 0.98` per month (≈ 4-year
memory) and `makeup_clip: 0.02` it is bounded and behaves like a mildly more persistent inflation target. Default off;
if switched on, the leak and the clip are mandatory and the stability suite must be re-run.

### 2.14.4 Risk management, financial conditions, and the ELB toolkit

- **Asymmetric easing (Sahm-type trigger).** When the 3-month average unemployment rate exceeds its previous-12-month
  minimum by `sahm_threshold` (0.5 pp), subtract `sahm_cut` (50bp, decaying over 12 months): central banks cut fast and
  hike slowly. Verified bounded; it lowered the 100-year minimum policy rate from 1.8 % to 1.5 % without raising volatility.
- **Financial conditions** (`phi_fci`, needs Phase 6 for the equity term): `FCI_tighten` = a weighted z-score of the
  corporate spread `s_t`, the bank capital gate `1 − g`, the equity drawdown and the bond term premium. Tightening
  financial conditions substitute for policy tightening; this is also the channel through which a market crash pulls the
  rate down before any macro data has moved.
- **At the ELB**, in order: forward guidance (an announced path with `credibility` ∈ [0,1] weighting it into the expected
  path of `06-…` §6.3), then QE (T6.28) sized by a rule (`qe_per_gap_point` of GDP per point of output gap), then
  lender-of-last-resort and macroprudential easing. `elb: 0.0`; negative rates are a config option, not a default.
- **Lean or clean:** credit growth, house prices and leverage act on the **macroprudential** levers (capital requirement,
  LTV), not on the policy rate. `phi_credit` exists for experiments and is 0 by default — this is a genuinely contested
  question and the plan refuses to take a side by default.

### 2.14.5 Communication as an observable

Every decision publishes a `NewsItem` with the new rate, the statement tone (an ordinal from the rule's gap terms), and a
projection path ("dot plot") = the rule's own forecast under current conditions, with `projection_noise`. Minutes follow
at `minutes_lag_days` (21). This is what an agent forecasts; the gap between the projection and the outcome is the policy
surprise that `06-…` §6.3 and the announcement-effect test (T6.32) key on. In professional mode there is a blackout
window: no policy `NewsItem` other than the decision itself in the 10 days before a meeting.

### 2.14.6 Config (`config/policy.yaml: monetary`)

```yaml
monetary:
  control: autopilot                  # autopilot | scripted | agent  (§2.13)
  calendar: {meetings_per_year: 8, rate_step: 0.0025, deadband: 0.0010, blackout_days: 10, minutes_lag_days: 21}
  rule:
    phi_inflation: 1.50               # from edges.yaml: policy.taylor
    phi_u: 1.0
    phi_y_mult: 0.0
    core_weight: 0.5
    smoothing: 0.80                   # quarterly-equivalent; rescaled to ^(4/meetings_per_year)
    kappa_rstar: 1.0
    rstar_tau_m: 60
  data: {cpi_lag_m: 1, unemployment_lag_m: 1, gdp_lag_q: 1, use_published_vintages: true}
  strategy: {makeup: 0.0, makeup_decay: 0.98, makeup_clip: 0.02}     # leak + clip are mandatory if makeup > 0
  risk_management: {enabled: false, sahm_threshold: 0.005, sahm_cut: 0.005, decay_m: 12}
  financial_conditions: {phi_fci: 0.0, weights: {corp_spread: 0.4, capital_gate: 0.3, equity_drawdown: 0.2, term_premium: 0.1}}
  credit: {phi_credit: 0.0}           # lean-against-the-wind experiment; prefer macroprudential levers
  elb: {rate: 0.0, allow_negative: false, guidance_credibility: 0.7, qe_per_gap_point: 0.02}
  committee: {dispersion_bp: 0.0, projection_noise_bp: 25}
```

### 2.14.7 Validation (T2.35)

Autopilot parity with the old rule when `phi_u: 0, meetings_per_year: 4, rate_step: 0, data_lag: 0` (bitwise) · steady
state exact at π\* ∈ {0, 2 %} with the full framework on · §2.12 sign and timing tests unchanged within tolerance
(monetary trough −0.28 % at month 15, rebound ≤ 0.6) · a demand shock raises the rate, a cost-push shock raises it less
than a pure-headline rule would · dual mandate: a labour-market-only shock (labour supply −2 %) moves the rate with the
sign of the unemployment gap · ≈ 3–6 moves a year averaging 20–40bp over 100 years · makeup with `decay: 1.0` and no clip
is rejected at config load · risk management triggers in the recession scenario and not in a soft landing · ELB: at
`elb` the rule's shadow rate is recorded, guidance and QE engage, and the economy still recovers.

---

## 2.15 Task cards

### T2.01 — `dynamics.yaml` and its schema
**Depends:** T0.14 · **Size:** S · **Files:** `config/dynamics.yaml`, `src/marketsim/core/config.py`, `tests/unit/core/test_dynamics_config.py`
**Read first:** §2.11. **Build:** full file + pydantic model; every sector has a production mode; leak only on stock
mode; ranges checked (shares in [0,1], τ > 0). **Tests:** loads; bad mode / negative τ / leak on a flow sector rejected.

### T2.02 — Ledger core
**Depends:** T0.11 · **Size:** M · **Files:** `src/marketsim/ledger/{entities,instruments,journal}.py`, `config/ledger.yaml`, `tests/unit/ledger/test_journal.py`
**Read first:** §2.2. **Build:** entity/instrument registries (dynamic registration, stable integer ids), `pos` matrix,
`post(tx)` with per-instrument balance check, columnar journal with tags, period flow accumulator `(payer, payee, tag)`,
`debug_journal` switch, `to_state/from_state`. **Tests:** unbalanced Tx rejected; payment to `BANKSYS` reduces its DEP
liability; new loan creates deposit; registration after start keeps ids stable; state round trip.
**Out of scope:** matrices (T2.03).

### T2.03 — Balance-sheet and transaction-flow matrices, SFC assertion
**Depends:** T2.02 · **Size:** M · **Files:** `src/marketsim/ledger/{matrices,sfc}.py`, `tests/unit/ledger/test_sfc.py`
**Read first:** §2.2 assertions. **Build:** `balance_sheet_matrix()`, `transaction_flow_matrix(period)`,
`assert_consistent()` checks (a)–(d), readable `SFCError`. **Tests:** hand-built 4-entity economy passes; a deliberately
one-sided posting, a missing counter-entry and a stock changed without a flow are each caught and named.

### T2.04 — Steady state, real side
**Depends:** T2.01, T0.04, T0.12 · **Size:** M · **Files:** `src/marketsim/real/steady_state.py`, `tests/unit/real/test_steady_state_real.py`
**Read first:** §2.3 (first 12 lines). **Build:** `RealBaseline` dataclass (x0, s0, m, va, ell, gos, markup, K0, v, ρ_K,
route_bus, res0, inv0, backlog0, S_in0, n0, LF) for shape `(R,S)` with R = 1. **Tests:** with all leaks 0, `x0 = L·d0`
and gross output = 209.3 per 100 cr (±0.1); trade balanced to 1e-9; `Σ (δ/12)·v·K0 = I_bus0`; `markup·(Σa + ell + m) = 1`;
wage share of value added ≈ 0.518 before the import carve-out (document the after value).

### T2.05 — Steady state, financial side and opening postings
**Depends:** T2.04, T2.03 · **Size:** M · **Files:** `src/marketsim/real/steady_state.py`, `src/marketsim/ledger/opening.py`, `tests/unit/real/test_steady_state_fin.py`
**Read first:** §2.3 (rest), §2.2 opening balance sheet. **Build:** debt, spreads, `G`, `grow`, interest timing, tax,
payout, `B`, `W`, `YD0`, α2, τ_y for `banks.mode: passthrough`; post the opening balance sheet; assert SFC.
**Tests:** payout ∈ (0.3, 0.8) for every sector; α2 > 0; sector balances sum to zero (household saving = deficit +
firm net borrowing + trade balance) at π\* ∈ {0, 0.02}; vat ∈ {0, 0.1} both solvable.

### T2.06 — Production plans and capacity caps (R1, R4)
**Depends:** T2.04 · **Size:** M · **Files:** `src/marketsim/real/production.py`, `tests/unit/real/test_production.py`
**Read first:** §2.4 R1, R4. **Build:** pure functions `expected_sales`, `plan_output`, `labour_cap`, `input_cap`,
`critical_mask(A, mu, cfg)`. **Tests:** at baseline plan = x0 exactly; stock gap raises plan by gap/τ_inv; each cap binds
in a constructed case; critical mask matches the rule (e.g. ENERGY→UTILITIES critical, BIZSVC→anything not).

### T2.07 — Orders, rationing, deliveries, input stocks (R5–R6)
**Depends:** T2.06 · **Size:** M · **Files:** `src/marketsim/real/orders.py`, `tests/unit/real/test_orders.py`
**Read first:** §2.4 R5–R6. **Build:** order matrix on plans, the three supply modes, pro-rata fill, deliveries,
`S_in` update, backlog with loss. **Tests:** conservation per mode (`inv' = (1−λ)inv + x − sales`; `ob' = ob − x`);
`S_in ≥ 0` under a 50 % supplier outage; at baseline everything is a fixed point; hypothesis property test: Σ deliveries
+ final sales = sales.

### T2.08 — Price formation (R7)
**Depends:** T2.04 · **Size:** M · **Files:** `src/marketsim/real/prices.py`, `tests/unit/real/test_prices.py`, `tests/validation/test_cost_push_parity.py`
**Read first:** §2.4 R7, AGENTS rule 7. **Build:** unit cost, tightness, drift-compensated two-speed filter, step bound,
`p_imp` drift. **Tests:** fixed point at π\* ∈ {0, 0.02}; a +400 % target moves price by exactly 0.15 log-points per
month and keeps going (no level cap); with quantities and wages frozen and tightness off, the long-run response to an
ENERGY cost shock ranks sectors as Layer 1's `G` does — top-3 non-energy = {UTILITIES, MATERIALS, TRANSPORT}, bottom
two ⊂ {SOFTWARE, BANKS, HEALTH, INSURANCE}; low `pass_through` sectors lag (UTILITIES half-life > TRANSPORT's).

### T2.09 — Labour and wages (R8)
**Depends:** T2.06 · **Size:** S · **Files:** `src/marketsim/real/labour.py`, `tests/unit/real/test_labour.py`
**Build:** target employment with fixed-cost share, asymmetric adjustment, LF cap, Phillips wage rule with downward
stickiness. **Tests:** fixed point; hiring faster than firing; wage falls 4× slower than it rises for symmetric gaps;
operating leverage: a 10 % output drop cuts employment by (1 − fc)·10 % in the long run.

### T2.10 — Capex, capacity pipeline, supply line (R3, §2.6)
**Depends:** T2.04, T0.12, T0.14 · **Size:** M · **Files:** `src/marketsim/real/capex.py`, `tests/unit/real/test_capex.py`
**Read first:** §2.6. **Build:** start-rate function, clip, spending and completion chains, depreciation, `Q` term wired
to the provider interface. **Tests:** baseline starts = δK/12 and K constant; +100bp `cc_gap` → start rate −0.45 pp/yr
(identical for every sector under the common rate; in `risk_based` mode +100bp spread at nd = 5 → −0.90 pp); supply line: doubling the pipeline lowers the rate; completions
lag starts with mean `3·build_lag_q` months; ENERGY (12q) slower than CONSTRUCT (2q).

### T2.11 — Residential investment block
**Depends:** T2.10 · **Size:** S · **Files:** `src/marketsim/real/residential.py`, `tests/unit/real/test_residential.py`
**Build:** §2.4 R3 residential line; household pays; routes to CONSTRUCT. **Tests:** baseline = 20 % of I; +100bp real
rate gap → −4 % starts after the lag; income elasticity 1.

### T2.12 — Household sector (aggregate, scalar-η demand system)
**Depends:** T2.05 · **Size:** M · **Files:** `src/marketsim/real/households.py`, `src/marketsim/demand/system.py`, `tests/unit/real/test_households.py`
**Read first:** §2.4 R2, AGENTS rule 9. **Build:** `DemandSystem` protocol (`allocate(budget, prices, income_index,
rate_gap, shifters) -> quantities`) with `ScalarEtaDemand`; consumption function; drift-compensated `YD_e`; wealth
from the ledger. **Tests:** baseline basket = `final_demand.HOUSEHOLD` exactly; shares sum to 1; numerical income
elasticity of each sector's share has the sign of (η − 1); AUTOS falls most for +100bp rate gap.

### T2.13 — Government and fiscal rule
**Depends:** T2.05 · **Size:** S · **Files:** `src/marketsim/real/government.py`, `tests/unit/real/test_government.py`
**Read first:** §2.5. **Build:** purchases, transfers, taxes (income, corporate, VAT), debt rule, bond issuance posting.
**Tests:** deficit = `grow·B` at baseline; debt ratio returns to 0.60 after a +10 pp displacement (half-life < 25 yr);
without the rule (`kappa_debt: 0`) and r > g the ratio diverges (diagnostic, marked `slow`).

### T2.14 — Central bank and inflation expectations
**Depends:** T2.04 · **Size:** S · **Files:** `src/marketsim/real/cenbank.py`, `tests/unit/real/test_cenbank.py`
**Build:** CPI/core histories, π3/π12, anchored expectations, quarterly Taylor rule with smoothing and ELB, `z_mon`,
meeting scheduled on the event queue. **Tests:** rule value by hand for two cases; rate changes only at quarter ends;
ELB binds; histories seeded so π12 = π\* from month 1.

### T2.15 — ROW
**Depends:** T2.04 · **Size:** S · **Files:** `src/marketsim/real/row.py`, `tests/unit/real/test_row.py`
**Build:** exports (world index × relative price elasticity), non-competing imports at `p_imp`, `p_imp` drift and shock,
trade balance posted; `ROW` accumulates deposits / government bonds when trade is unbalanced (no FX in v1). **Tests:** balanced at
baseline for π\* ∈ {0, 0.02}; +10 % domestic prices → exports −7.3 % (ε_x 0.8); ROW position = −Σ trade balances.

### T2.16 — Income settlement on the ledger (R9)
**Depends:** T2.03, T2.07–T2.15 · **Size:** L · **Files:** `src/marketsim/real/settlement.py`, `tests/unit/real/test_settlement.py`
**Read first:** §2.4 R9, §2.2. **Build:** all monthly flows as netted postings with tags; firm profit, tax, dividends
with the leverage norm, debt dynamics via `loan_new`/`loan_repay` (no clipping — negative debt becomes a deposit);
household income and wealth; `banks.mode: passthrough`. **Tests:** SFC assertion after every month of a 120-month
shocked run; household saving = government deficit + firm net borrowing + trade balance each month (±1e-9);
reproduce the prototype bug as a regression: forcing `max(debt, 0)` makes the SFC test fail.

### T2.17 — Monthly orchestrator, aggregates, World integration
**Depends:** T2.16, T0.13 · **Size:** M · **Files:** `src/marketsim/real/economy.py`, `src/marketsim/real/aggregates.py`, `tests/integration/test_real_economy.py`
**Build:** `RealEconomy` module running R1–R10 at month end; aggregates: GDP (production and expenditure side, nominal
and real, net of spoilage), CPI, core CPI, unemployment, utilisation, sector output, government balance, debt/GDP, trade
balance, saving rate, leverage; `published(series, lag)` view; state round trip. **Tests:** gate items 1–2; world hash
deterministic; save/load mid-run identical; one monthly step < 5 ms at R = 1 (ledger on, debug journal off).

### T2.18 — Shock bus and the seven primitives
**Depends:** T2.17 · **Size:** M · **Files:** `src/marketsim/real/shocks.py`, `tests/unit/real/test_shocks.py`
**Read first:** §2.9. **Build:** AR(1) shock states with target masks, `ShockBus.inject`, catastrophe (capital and
inventory destruction, insurance claims routed to `claims_to`, INSURANCE equity hit, all on the ledger).
**Tests:** persistence half-life matches `persistence_q`; masks hit only targets; catastrophe conserves money
(claims = insurer outflow = claimant inflow); mid-month weighting `(21 − d)/21`.

### T2.19 — Banking system (`banks.mode: full`)
**Depends:** T2.17 · **Size:** M · **Files:** `src/marketsim/real/banks.py`, `src/marketsim/real/steady_state.py`, `tests/unit/real/test_banks.py`
**Read first:** §2.7, §2.2 opening balance sheet. **Build:** margins, deposit rate floor, expected losses and write-offs,
capital ratio, payout rule; initialiser extended so the steady state is still exact. **Tests:** gate 1 with banks on;
bank profit rises with the policy rate (compare r = 1 % vs 4 % steady states); capital ratio 0.125 at baseline.

### T2.20 — Credit and collateral edges
**Depends:** T2.19, T2.21 · **Size:** M · **Files:** `src/marketsim/real/credit.py`, `tests/validation/test_credit_crunch.py`
**Read first:** §2.7. **Build:** gate, collateral index, spreads, rationing exponents with per-edge Erlang lags; flag
`credit.gate_enabled`. **Tests:** no effect at baseline (bitwise-equal run with flag on and no shock); credit-crunch
test (§2.7); asymmetric collateral: −20 % `V_RE` cuts capacity ≈ 3× more than +20 % raises it; round trip leaves Λ_coll = 1.

### T2.21 — AssetPriceProvider stub
**Depends:** T2.17 · **Size:** S · **Files:** `src/marketsim/pricing/provider.py`, `tests/unit/pricing/test_stub_provider.py`
**Read first:** §2.10. **Build:** interface + Gordon-style stub. **Tests:** Q = 1 at baseline; +100bp real rate lowers
`V` by ≈ 0.35 × `cf_duration` % (REALESTATE ≈ −6.3 %, AUTOS ≈ −2.5 %).

### T2.22 — Typed substitution / complement edges
**Depends:** T2.12, T0.12 · **Size:** M · **Files:** `src/marketsim/real/edges.py`, `tests/unit/real/test_typed_edges.py`
**Read first:** §2.8. **Build:** `TypedEdge`, per-edge smoother, saturation, shifters fed into R2/R3/ROW. **Tests:** no
effect at baseline; ENERGY relative price +20 % → long-run AUTOS final demand ≈ −5.3 % (complement −0.30), UTILITIES ≈ +6.6 % (substitution 0.35), each following its Erlang lag profile; saturation caps.

### T2.23 — Validation suite: stationarity, IRFs, timing
**Depends:** T2.18 · **Size:** M · **Files:** `tests/validation/test_dynamics_stationarity.py`, `test_dynamics_irf.py`, `test_dynamics_timing.py`, `src/marketsim/scenarios/irf.py`
**Read first:** §2.12. **Build:** IRF harness (`run_irf(kind, size, persistence_q, months)`), level-window tests, timing
tests; re-run automatically with banks/credit/edges on (parametrised). **Tests:** the §2.12 table and timing block.

### T2.24 — Stability runs, sweeps, moments report
**Depends:** T2.23 · **Size:** M · **Files:** `scripts/sweep.py`, `tests/validation/test_dynamics_stability.py`, `claude/plan/reports/phase2-validation.md`
**Build:** stochastic harness, envelope metric, volatility ranking, sweep CLI over {`unit_scale` 0.0075/0.01,
`supply_line_weight` 0.85/1.0, `kappa_util` 0.6/1.2, `anchor_growth` 0.5/0.75, leak ×0/×1/×3}; report with business-cycle
moments and the known gaps (I/GDP, C/GDP, cost-push size) and tuning proposals. **Tests:** §2.12 stochastic block
(`slow`); every sweep cell bounded. **Out of scope:** changing defaults — propose in the report, human decides.

### T2.25 — Performance baseline
**Depends:** T2.17 · **Size:** S · **Files:** `scripts/bench.py`, `tests/integration/test_perf_smoke.py`
**Build:** benchmark daily ticks with the real economy only. **Tests:** ≥ 5,000 ticks/s at R = 1 with the debug journal
off (the real step runs 1 tick in 21); report written to `claude/plan/reports/`.

### T2.26 — CES substitution on intermediates (optional, off by default) — **HUMAN GATE**
**Depends:** T2.23 · **Size:** M · **Files:** `src/marketsim/real/ces.py`, `tests/unit/real/test_ces.py`
**Build:** `a_ij,t = a_ij·(p_i/p̄_j)^(−σ_ij)` renormalised so real intermediate intensity per unit output is
unchanged; σ from the `substitution` pairs that are input relationships (ENERGY↔UTILITIES fuel switching,
MATERIALS→CONSTRUCT); Erlang lag per pair. **Tests:** σ = 0 reproduces Leontief bitwise; suite §2.12 green with it on.

### T2.27 — Policy-authority framework (D13)
**Depends:** T2.13, T2.14, T2.18 · **Size:** M · **Files:** `config/policy.yaml`, `src/marketsim/real/policy/authority.py`, `tests/unit/real/test_policy_authority.py`
**Read first:** §2.13. **Build:** `PolicyDecision` (every field optional), control modes, lever registry with ranges and
per-period change limits, legislative and implementation lags as queue items, announcement `NewsItem`, per-lever arbitration,
state round trip. **Tests:** on autopilot the world hash is identical to the pre-framework run; out-of-range decisions are
clipped and the adjustments reported; lags honoured; partial decisions leave other levers on autopilot.

### T2.28 — Fiscal instruments
**Depends:** T2.27, T2.16 · **Size:** M · **Files:** `src/marketsim/real/policy/fiscal.py`, `src/marketsim/real/government.py`, `tests/unit/real/test_fiscal_instruments.py`
**Read first:** §2.13 table. **Build:** purchase composition and regional allocation; tax rates incl. excise / subsidy wedge
and tariff (consumer-price wedges, revenue postings); one-off and targeted transfers; capex subsidy; capital injections.
**Tests:** every instrument posts balanced Tx; a VAT / excise wedge raises the consumer price, not the producer price; tariff
revenue = rate × import value; `GOVT` deposits never negative (shortfall → issuance); SFC green.

### T2.29 — Monetary and macroprudential instruments
**Depends:** T2.27, T2.20 · **Size:** M · **Files:** `src/marketsim/real/policy/monetary.py`, `tests/unit/real/test_monetary_instruments.py`
**Build:** rate override and rule-parameter levers, forward-guidance path (stored for §6.3), capital requirement, LTV cap,
lender-of-last-resort loan; QE / QT interface declared here, implemented in T6.28. **Tests:** overrides apply only at
meetings; a higher capital requirement lowers the gate; LOLR posting balanced (`RES` / `LOAN`).

### T2.30 — One corporate borrowing rate (D14)
**Depends:** T2.20 · **Size:** S · **Files:** `src/marketsim/real/credit.py`, `src/marketsim/real/steady_state.py`, `config/dynamics.yaml`, `claude/plan/DECISIONS.md`, `tests/unit/real/test_uniform_rate.py`
**Read first:** §2.7 first bullet, §2.6, AGENTS rule 16. **Build:** `credit.pricing: uniform | risk_based`; a single global
spread state; no sector / region / firm index anywhere in loan or pool pricing in uniform mode. **Tests:** every borrower's
rate equal (1e-12) in every month of a shocked run; steady state exact; §2.12 green in both modes; a default in one sector
raises the rate of all sectors by the same amount.

### T2.31 — Policy validation
**Depends:** T2.28, T2.29, T2.23 · **Size:** M · **Files:** `tests/validation/test_policy_levers.py`, `tests/adversarial/test_policy_extremes.py`, `claude/plan/reports/policy-validation.md`
**Tests:** the §2.13 validation list; report the measured fiscal multiplier and pass-through numbers.

### T2.32 — Monetary policy framework: committee, calendar, reaction function (D15)
**Depends:** T2.27, T2.29 · **Size:** M · **Files:** `config/policy.yaml`, `src/marketsim/real/policy/monetary_rule.py`, `src/marketsim/real/cenbank.py`, `tests/unit/real/test_monetary_rule.py`
**Read first:** §2.14.1–§2.14.2. **Build:** meeting calendar as queue items, continuous internal target with per-meeting
smoothing `ρ^(4/mpy)`, dual-mandate reaction function with core/headline blend and time-varying r\*, 25bp grid with
deadband, committee dispersion, decision `NewsItem` with statement tone and projection path, blackout window.
**Tests:** hand-computed target for three states; parity with the old quarterly rule under the parity settings (bitwise);
steady state exact at π\* ∈ {0, 2 %}; grid and deadband produce 3–6 moves a year of 20–40bp over 100 years; dispersion is
deterministic per seed.

### T2.33 — The committee's information set (published vintages)
**Depends:** T2.32, T4.07 · **Size:** S · **Files:** `src/marketsim/real/policy/monetary_rule.py`, `tests/unit/real/test_policy_information.py`
**Read first:** §2.14.2, `04-…` §4.3. **Build:** the rule reads only published vintages (CPI 1m, unemployment 1m, GDP 1q
with revisions); `use_published_vintages: false` restores the oracle for ablations.
**Tests:** poisoning the true series without republishing does not change any decision; the oracle and the vintage rule
differ after a revision; documented volatility cost of the lag (sd(π) rises monotonically with the lag).

### T2.34 — Strategy options: makeup, risk management, financial conditions
**Depends:** T2.32, T2.20 · **Size:** M · **Files:** `src/marketsim/real/policy/monetary_rule.py`, `src/marketsim/core/config.py`, `tests/unit/real/test_policy_strategies.py`, `tests/validation/test_policy_strategy_stability.py`
**Read first:** §2.14.3–§2.14.4. **Build:** leaky, clipped price-level gap; Sahm-type asymmetric easing with decay;
FCI term (equity weight inactive until Phase 6); `phi_credit`; ELB toolkit ordering (guidance → QE → LOLR).
**Tests:** `makeup > 0` with `makeup_decay >= 1.0` or `makeup_clip <= 0` is **rejected at config load** with the
prototype's numbers in the message; each strategy leaves the 100-year runs bounded; Sahm triggers in a scripted recession
and not in a soft landing; a wider corporate spread lowers the rate when `phi_fci > 0`.

### T2.35 — Monetary validation and the policy-rule report
**Depends:** T2.33, T2.34, T2.31 · **Size:** M · **Files:** `tests/validation/test_monetary_framework.py`, `claude/plan/reports/monetary-validation.md`
**Tests:** the §2.14.7 list; report measured move frequency and size, the sacrifice ratio of a disinflation, the ELB
episode share, and a comparison table of the rule variants (dual mandate vs output gap, headline vs core, lag 0–3 months).
