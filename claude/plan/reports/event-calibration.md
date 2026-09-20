# T4.13 — Event magnitude calibration

Date: 2026-09-20. Sources are published statistical-agency / central-bank figures
(no network call). **Out of scope:** retuning the economy so simulated CPI/GDP
*levels* match history (T8.04). Seeds stay distributions; `verify` is now false.

## Mapping rule

Historic *real* moves (price ×N, GDP %, unemployment pp) are **not** copied 1:1
onto ShockBus `z`. The Phase-2 cost-push IRF of `z = 0.30` already produces
~+72 % CPI @12m (T2.24). Putting `ln(4) ≈ 1.39` on ENERGY `z_cost` (the 1973
crude move) overflows SFC by month 17 (QUESTIONS T4.11). T8.04 must choose the
`z` that reproduces the *published* CPI/GDP path through A and the typed edges.

Until then the YAML keeps the §4.4 seed distributions (they encode the *sign and
order of magnitude* of the composition) and T4.11 asserts **directions** on the
SFC-finite prefix.

## Historic templates

| id | published figure | source | seed kept |
|---|---|---|---|
| oil_embargo_1973 | Arabian Light ≈ $2.90 → $11.65 (≈×4), Oct 1973–Jan 1974 | EIA / BP Statistical Review | cost_push ENERGY lognormal median 1.25, min 1.1 max 1.4 |
| asian_crisis_1997 | emerging-Asia import collapse 1998; no FX in v1 | IMF WEO 1998 | world_demand −8 %, risk +200–400 bp |
| dotcom_2000 | NASDAQ −78 % Mar 2000–Oct 2002 | FRED NASDAQCOM | risk +, capex_preference SOFTWARE/TELECOM/SEMIS −30 % |
| gfc_2008 | CSUSHPINSA peak–trough ≈ −27 %; S&P 500 −57 % | FRED CSUSHPINSA, SP500 | collateral −25…30 %, bank_equity −30…50 %, risk +400 bp |
| tohoku_2011 | Japan auto output ≈ −50 % Mar–Apr 2011 | METI / JAMA | catastrophe 1–3 % of K; link_capacity −50 % 60 d |
| thailand_floods_2011 | HDD shipments ≈ −29 % 2011Q4 / 2011Q3 | iSuppli / TrendFocus | supply SEMIS −25…30 %, 2q |
| covid_2020 | real GDP −31.2 % SAAR Q2 (~−9 % q/q); U-3 14.7 % Apr | BEA NIPA, BLS CPS | labour −10…15 %; want shifts; world_demand −10 % |
| chip_shortage_2020 | chip lead times peaked ≈ 26 weeks | Susquehanna / industry surveys | supply SEMIS lognormal median 8 % |
| suez_2021 | ~12 % of world trade transits Suez | UNCTAD RMT | 6-day pulse on import_price / link_cost / world_demand |
| energy_inflation_2022 | CPI-U 9.1 % y/y Jun 2022; funds +525 bp in 16 months | BLS CPI, FOMC | cost_push ENERGY +0.5…0.9, AGRIFOOD +0.2 |
| ai_boom_2023 | US data-centre and semiconductor equipment capex surge | BEA / 10-K capex | ramp capex_preference +20…40 % SEMIS/SOFTWARE/UTILITIES |

## Generic events

| id | seed | rationale |
|---|---|---|
| fiscal_stimulus | fiscal +2…5 %, 3q | IMF discretionary impulse band for advanced-economy packages |
| covid / gfc fiscal | +4…8 % / +3…6 % | BEA / CBO COVID relief and ARRA-scale packages as % of GDP |
| monetary_tightening | +100–300 bp, 3q | FOMC hiking-cycle steps (not the 2022 525 bp cumulative) |
| oil / energy monetary | +200–400 bp | 1974 / 2022 tightening *increments* around the energy spike |
| world_demand_drop | −5…−10 % | IMF world-trade growth gaps in 1998 / 2009 / 2020 |
| credit_crunch | risk + collateral extra | GFC-style Λ path; quantity rationing, not a firm-specific spread (D14) |
| confidence_drop | demand −2…−5 % | consumer-sentiment recessions (UMCSENT) as a demand primitive |
| vat_change | +1…3 pp | typical European VAT step; lever path, not a CPI target |
| trade_tariff | import_price MATERIALS +5…15 % | 2018–19 steel/aluminium order of magnitude |
| policy_surprise | monetary N(0, 50 bp) clipped ±100 bp | unscheduled 25–50 bp surprises |
| strike / plant_accident / recall | supply or demand 1–10 % | firm-level; bodies in Phase 5 |
| large_bankruptcy | bank_equity write-off 2–8 % | one-off; SFC posted |
| regional_disaster | catastrophe small-κ | Tohoku-lite |
| energy_subsidy | fiscal +1…3 % | 2022 energy-price shields as % of GDP |

## Checks

- Catalog `verify` flags are all false (including `historic_reference.calibrated_params`).
- T4.11 §4.5 direction tests remain green (12-month prefix for oil/energy).
- Engine *level* match is T8.04.
