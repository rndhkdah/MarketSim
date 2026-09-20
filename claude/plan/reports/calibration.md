# T8.04 — Calibration pass (HUMAN GATE)

Date: 2026-09-20. Method: **ADR-015** (authoritative). Historic figures:
`reports/event-calibration.md`. Seed moments: `reports/phase2-validation.md`
(T2.24) and `reports/monetary-validation.md` (T2.35). BEA workbook handling:
`reports/bea-howto.md` (local `--xlsx` / `--fd-xlsx` only).

**Not applied; human must tick T8.04.** No shipped yaml was rewritten. Seed
`config/io_table.json` still has hand-set `final_demand` and `fd_weights`
0.62 / 0.19 / 0.13 / 0.06. `world.io_source` stays `seed`. Coverage floor
stays 0.85. No network call. No test tolerance was loosened.

Scripts (print / optional JSON; refuse `.yaml`):

- `scripts/fetch_bea_io.py` — Use-table aggregate (T0.09) + **final-demand concordance**
- `scripts/calibrate_moments.py` — proposed dynamics / household / capex gains
- `scripts/calibrate_events.py` — published figures → ShockBus `z` in the IRF-safe band

---

## 1. Final-demand concordance

### Method (ADR-015 §1)

BEA NIPA-style columns — personal consumption expenditures (PCE), private
fixed investment, government consumption + investment, exports of goods and
services — are mapped onto the 18 `CODES` (`marketsim.layer1.build_io.CODES`;
never retyped). Each source industry's destination weights sum to 1
(classification map, not a fitted elasticity). Conjunction names (e.g.
Finance + insurance + real estate) use equal weights.

```
fd_matrix[k, s]  = Σ_i  dollars_{k,i} · w_{i→s}     k ∈ {C, I, G, X}
weights[k]       = (Σ_s fd_matrix[k, s]) / grand     Σ_k weights[k] = 1
coverage         = mapped dollars / all source dollars
```

`weights` are the **observed** C/I/G/X shares of that workbook year. The
seed vectors stay in place until a human writes an accepted table and
switches `world.io_source`. Coverage below 0.85 exits non-zero.

CLI (no network; seed `io_table.json` is refused as an output path):

```bash
python scripts/fetch_bea_io.py --dry-run
python scripts/fetch_bea_io.py --xlsx /path/to/use.xlsx --fd-xlsx /path/to/fd.xlsx --fd-out /tmp/fd_proposed.npz
```

`--fd-xlsx` is optional when the same `--xlsx` workbook already carries PCE /
I / G / X as named columns or as `DataFrame.attrs` (`pce`, `investment`,
`government`, `exports`). A local `--concordance` sidecar is `{concordance,
splits}`. Defaults: `DEFAULT_NIPA_CONCORDANCE` in `fetch_bea_io.py`
(15-industry fallbacks + 71-industry names + NIPA PCE / I / G / X lines).

This is **not** a BEA 2023 download. The fetch script still refuses to call
the network (T0.09). There is no claimed “BEA 2023” `fd_weights` in this
report.

### Seed vs proposed (method)

| key | seed (`io_table.json`) | proposed (concordance method) |
|---|---|---|
| `fd_weights.HOUSEHOLD` | 0.62 | observed C / (C+I+G+X) from the local workbook |
| `fd_weights.INVESTMENT` | 0.19 | observed I / (C+I+G+X) |
| `fd_weights.GOVT` | 0.13 | observed G / (C+I+G+X) |
| `fd_weights.EXPORTS` | 0.06 | observed X / (C+I+G+X) |
| `final_demand.*` | hand-set 18-vectors (each sums to 1) | row `fd_matrix[k] / Σ_s fd_matrix[k]` |
| `source` | `seed` | still `seed` until the human switches |
| coverage floor | 1.0 on the seed | 0.85 on a mapped workbook |

### Example arithmetic (synthetic; not a BEA year)

Worked example from `example_fd_arithmetic()` — three PCE lines, two I lines,
two G lines, one export line:

| component | source dollars | mapped to |
|---|---|---|
| PCE | Food 60 + Autos 20 + Housing 20 = **100** | STAPLES, AUTOS, REALESTATE |
| I | Machinery 15 + Construction 5 = **20** | CAPGOODS, CONSTRUCT |
| G | Health 10 + Defence 5 = **15** | HEALTH, CAPGOODS |
| X | Autos 5 = **5** | AUTOS |
| grand | **140** | coverage = 1 |

Proposed weights = 100/140, 20/140, 15/140, 5/140 ≈ **0.714 / 0.143 / 0.107 / 0.036**
versus seed **0.62 / 0.19 / 0.13 / 0.06**.

Identity check (not a new year): if the four source maps are already on
`CODES` and equal `100 × seed_weight × seed_composition`, the method returns
the seed weights exactly. That only says the arithmetic is conservative; it
does not claim the seed *is* a BEA year.

---

## 2. Cost-push: T2.24 IRF vs proposed oil `z`

T2.24 (`reports/phase2-validation.md`): ENERGY cost-push **z = 0.30** for 8q
gives **CPI ≈ +72 % @12m** and GDP trough **−13 %** at month 42. Prototype
(master plan §13) at the same *stated* +30 % was CPI +2.0 % @12m, GDP −2.7 %
month 22. The shock hits `lp*` and `p_imp` with no extra attenuation; price
steps then compound (`step_max` 0.15).

QUESTIONS T4.11 / ADR-007: putting the raw 1973 crude move on `z_cost`
(`ln 4 ≈ 1.39`, YAML median 1.25) overflows SFC by month 17–18.

ADR-015 §2: historic *real* moves are **targets**, not ShockBus values. Map
1973 crude ×4 onto `z_cost` in the **IRF-safe band (~0.30–0.40), not `ln 4`**.

| quantity | current (seed / T2.24) | proposed (not applied) |
|---|---|---|
| cost-push IRF `z` | 0.30 | — (IRF protocol unchanged) |
| CPI @12m at that `z` | **~+72 %** (T2.24) | still ~+72 % at z=0.30; engine overstates vs prototype +2.0 % |
| GDP trough at that `z` | **−13 %** m42 (T2.24) | still the T2.24 path until a human retunes pass-through |
| 1973 crude | Arabian Light $2.90 → $11.65 (≈×4.02) | target *price* path (EIA / BP) |
| naive `z_cost` | `ln(4) ≈ 1.386` / YAML median **1.25** | **do not use** |
| proposed oil `z_cost` | lognormal 1.1–1.4 | **0.30–0.40** (IRF-safe band) |
| 2022 energy `z_cost` ENERGY | uniform 0.50–0.90 | **0.30–0.40** (clip into the same band) |
| 2022 energy `z_cost` AGRIFOOD | 0.15–0.25 | **keep** (already below the band floor) |
| linear “match 9.1 % CPI” `z` | 0.30 × 0.091 / 0.72 ≈ **0.038** | recorded only; **not** proposed (would contradict the oil band) |

T2.24 reopen (ADR-015 §4): the proposed oil band does **not** fix the
cost-push *I* ranking or the +72 % CPI step. Do not invent a new
`edges.yaml` / `sectors.yaml` elasticity (no `pass_through` cut, no
price-level cap). A later accepted ADR may add a `z_cost` scale.

T2.35: dual-mandate cost-push ENERGY +10 % max r 14.5 % (headline 14.75 %).
Unchanged by this pass.

---

## 3. Business-cycle moments (FRED / BLS targets vs the seed)

Card / ADR-015 targets: `sd(I)/sd(GDP)` **3–4**, `sd(C)/sd(GDP)` **< 1**,
GDP persistence. The documented persistence proxy on this tree is the T2.24
100-year **envelope** (last/first 20y `|gap|` ratio **< 2**), plus exact
no-shock stationarity at π\* ∈ {0, 2 %}. No FRED AR(1) coefficient is
invented here.

### What the seed is known to do (T2.24 / T2.35)

T2.24, 1,200 months × 3 seeds, π\* = 2 % (`reports/phase2-validation.md`):

| seed | max \|gap\| | U | envelope | sd(I)/sd(GDP) | sd(C)/sd(GDP) |
|---|---|---|---|---|---|
| 1 | 2.16 % | 3.3–6.5 % | 1.43 | 0.88 | 0.78 |
| 2 | 1.83 % | 3.2–6.1 % | 1.00 | 0.96 | 0.99 |
| 3 | 1.82 % | 3.8–6.0 % | 0.95 | 1.03 | 0.91 |
| spec | < 10 % | (1 %, 12 %) | < 2 | **3–4** | **< 1** |

Soft C/GDP **meets** the target. Soft I/GDP is **≈ 1**, far below 3–4
(prototype ≈ 7 — the other side of the band). Envelope already inside spec.
Volatility ranking (SEMIS #1, CONSTRUCT in the top 5) **fails**; xfail
stands (QUESTIONS T2.24 A).

T2.35 (`reports/monetary-validation.md`): passthrough monetary trough
**month 14, −0.41 %**, rebound 0.24; 4.65 moves/year, 30 bp; sacrifice ≈ 5.1.
Do not move Taylor / D15 knobs in this pass.

### Proposed gains (T2.24 authorized only)

A linear close of the I/GDP gap, `φ ← 1.2 × (3.5 / 0.96) ≈ 4.4`, is
**rejected** — that invents an elasticity (ADR-015 §4). T2.24 already named
the step: `phi_accelerator` 1.2 → **1.6** *or* `anchor_growth` 0.75 → **0.6**,
then re-run the 100-year envelope. Do not stack both without that sweep.

| key | current (shipped) | proposed | target / note |
|---|---|---|---|
| `moment.sd_I_over_sd_GDP` | 0.88–1.03 (mean 0.96) | raise toward 3–4 via φ *or* anchor | 3–4 |
| `moment.sd_C_over_sd_GDP` | 0.78–0.99 (mean 0.89) | **keep** | < 1 (already met) |
| `moment.gdp_persistence_envelope` | 0.95–1.43 | **keep**; re-check if anchor moves | < 2 |
| `edges.capex.coefficients.phi_accelerator` | 1.2 | **1.6** (primary I step) | T2.24; not 4.4 |
| `dynamics.expectations.anchor_growth` | 0.75 | **0.6** (alternative, not stacked) | T2.24 |
| `dynamics.households.alpha1` | 0.70 | **0.70** | C already < 1 |
| `dynamics.households.tau_income_m` | 12 | **12** | prototype damper |
| `dynamics.expectations.tau_growth_m` | 12 | **12** | envelope already OK |
| `dynamics.residential.rate_semi` | −4.0 | **−4.0** | CONSTRUCT ranking xfail; no new elasticity |
| `edges.capex.unit_scale` | 0.01 | **0.01** | ADR-002; 0.0125 was a sweep idea only |
| `rejected.linear_phi_to_close_I_gap` | 1.2 | 4.38 | rejected |

Household gains stay put: seed `sd(C)/sd(GDP)` is already inside the FRED-style
`< 1` band (prototype was 1.45). Capex `ψ`, `χ`, `q_tobin` stay (ADR-P2 unit
contract). Ranking / bullwhip / AGRIFOOD leak ideas in T2.24 remain
**report-only** and still need an ADR.

---

## 4. Event magnitudes (published → `z`)

Full table: `python scripts/calibrate_events.py --dry-run`. Historic *real*
moves stay targets. YAML seed **distributions** (sign / order of magnitude)
stay until the human writes the accepted map.

| id | published figure | seed (kept in yaml) | proposed `z` |
|---|---|---|---|
| oil_embargo_1973 | crude ≈ ×4 | cost_push median 1.25 | **z_cost 0.30–0.40, not ln 4** |
| energy_inflation_2022 | CPI-U 9.1 % y/y | ENERGY 0.5–0.9, AGRIFOOD +0.2 | ENERGY **0.30–0.40**; AGRIFOOD keep |
| asian_crisis_1997 | Asia import collapse | world_demand −8 %, risk +200–400 bp | keep (not z_cost) |
| dotcom_2000 | NASDAQ −78 % | risk +; tech capex −30 % | keep (not −ln 0.22) |
| gfc_2008 | CSUSHPINSA −27 %; SP500 −57 % | collateral −25…30 %, bank_equity −30…50 % | keep (quantity map) |
| tohoku_2011 | Japan autos ≈ −50 % | catastrophe 1–3 % K; link −50 % / 60 d | keep |
| thailand_floods_2011 | HDD −29 % | supply SEMIS −25…30 % | keep |
| covid_2020 | GDP −9 % q/q; U-3 14.7 % | labour −10…15 %; wants; world_demand −10 % | keep labour/wants (T4.11 HEALTH) |
| chip_shortage_2020 | lead times ≈ 26 weeks | supply SEMIS median 8 % | keep |
| suez_2021 | ~12 % of world trade | 6-day pulse | keep (exposure ≠ z) |
| ai_boom_2023 | data-centre / semi capex surge | capex_preference +20…40 % | keep |
| fiscal / monetary generics | IMF / FOMC step bands | +2…5 % G; +100–300 bp | keep |

T4.10 10× storm may re-include oil / energy / covid **after** the oil `z`
map is accepted. Until then those three ids stay omitted from the storm
(QUESTIONS T4.11).

---

## 5. Explicit non-application

- Proposed numbers live in this report and in the script stdout / optional
  JSON. They are **not** in `edges.yaml`, `sectors.yaml`, `dynamics.yaml`,
  `config/events/*.yaml`, or `io_table.json`.
- Applying `phi_accelerator` 1.6 or the oil `z` band is a **human commit**
  (and an accepted ADR for any `edges.yaml` / `sectors.yaml` value).
- Earlier gates stay green because the economics were not changed.
- **Not applied; human must tick T8.04.**
