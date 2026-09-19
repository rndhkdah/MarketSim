# Phase 2 validation report (T2.24)

Measured 2026-09-19 on the reconstructed Layer-1 seed (`R = 1`, 18 sectors). Defaults
were **not** changed. Soft targets and ranking misses are QUESTIONS T2.24; the human
decides any retune.

## Stationarity and SFC

| run | `max \|x/x0 − 1\|` | π12 | GDP identity |
|---|---|---|---|
| passthrough, π\* ∈ {0, 2 %} | `< 1e-9` (360m) | = π\* | `< 1e-9` rel |
| `banks.mode: full`, both π\* | `< 1e-9` | = π\* | holds |
| full + `credit.gate_enabled` | `< 1e-9` | = π\* | holds |

48-cell sweep `{unit_scale 0.0075/0.01 × supply_line 0.85/1.0 × κ_u 0.6/1.2 ×
anchor_growth 0.5/0.75 × leak ×0/×1/×3}`: every cell finite and `|gap| < 1e-6`
over 120 no-shock months.

## Sign-restriction IRFs (π\* = 0, 240m)

| shock | output window | price window | passthrough / banks | credit-on |
|---|---|---|---|---|
| demand +2 %, 6q | Σ gap[1..24] > 0 | mean lvl[18..36] > 0 | pass | fail (sum −2.8 %) — QUESTIONS T2.23 |
| monetary +100bp, 4q | Σ gap[1..36] < 0 | mean lvl[18..36] < 0 | pass | pass (signs) |
| cost_push ENERGY +30 %, 8q | Σ gap[6..48] < 0 | mean lvl[6..24] > 0 | pass | pass |
| supply −3 %, 12q | Σ gap[1..48] < 0 | mean lvl[6..36] > 0 | pass | pass |
| fiscal G +5 %, 8q | Σ gap[1..24] > 0 | mean lvl[18..36] > 0 | pass | pass |
| row X −10 %, 6q | Σ gap[1..24] < 0 | — | pass | pass |
| risk_appetite +100bp, 3q | Σ gap[1..24] < 0 | not tested | pass | pass |
| catastrophe 5 % K | gap[1..6] < 0 | mean lvl[3..18] > 0 | month-1–5 GDP + — QUESTIONS T2.23 | same |

## Monetary timing

| mode | trough month | depth | rebound | AUTOS < CONSTRUCT < CAPGOODS |
|---|---|---|---|---|
| passthrough | 13 | 0.36 % | 0.21 | 9 < 14 < 15 |
| banks full | 12 | 0.41 % | 0.13 | 9 < 13 < 15 |
| credit-on | 47 | 0.86 % | n/a | 41 < 46 < 49 — QUESTIONS T2.23 |

Spec window: month ∈ [9, 24], depth 0.15–0.60 %, rebound < 0.6. Prototype was
≈ −0.30 % at month 16, rebound 0.44, 11 < 17 < 20.

## Credit crunch (§2.7)

40 % bank-equity write-off at month 12, gate on: gate 0.38 within the quarter;
trough −10.0 % vs −0.58 % with the gate off (17×); capital > 0.11 by year 8;
`|gap|` 0.29 % at year 12. SFC held.

## 100-year stochastic (z_dem ρ 0.90 σ 0.004; z_sup 0.95 / 0.0015; ENERGY cost 0.93 / 0.02)

π\* = 2 %, seeds 1–3, 1,200 months.

| seed | max \|gap\| | U range | envelope last/first 20y | sd(I)/sd(GDP) | sd(C)/sd(GDP) |
|---|---|---|---|---|---|
| 1 | 2.16 % | 3.3–6.5 % | 1.43 | 0.88 | 0.78 |
| 2 | 1.83 % | 3.2–6.1 % | 1.00 | 0.96 | 0.99 |
| 3 | 1.82 % | 3.8–6.0 % | 0.95 | 1.03 | 0.91 |
| spec | < 10 % | (1 %, 12 %) | < 2 | soft 3–4 | soft < 1 |

All finite. Soft C/GDP lands in spec (`< 1`). Soft I/GDP is far below 3–4
(prototype ≈ 7).

### Volatility ranking (gated; currently fails)

| seed | #1 | top 5 | CONSTRUCT | bottom 6 |
|---|---|---|---|---|
| 1 | AGRIFOOD | AGRIFOOD, AUTOS, CAPGOODS, ENERGY, SEMIS | 8 | … INSURANCE, BIZSVC, BANKS (TELECOM 12th) |
| 2 | CAPGOODS | CAPGOODS, AGRIFOOD, AUTOS, ENERGY, SEMIS | 7 | … INSURANCE, BANKS (TELECOM 12th) |
| 3 | AGRIFOOD | AGRIFOOD, CAPGOODS, AUTOS, ENERGY, MATERIALS | 7 | TELECOM … INSURANCE, BANKS |

Spec: SEMIS highest; CONSTRUCT and CAPGOODS in the top 5; INSURANCE, TELECOM,
BANKS in the bottom 6. CAPGOODS and the two financials mostly behave; SEMIS
bullwhip and CONSTRUCT are too quiet; AGRIFOOD is too loud (leak 1.5 %/m).

## Known gaps vs the prototype

1. **Investment volatility** — `sd(I)/sd(GDP) ≈ 1` vs 3–4 (proto ≈ 7). The
   accelerator / residential block is damping more than the verified prototype.
2. **Consumption** — `sd(C)/sd(GDP) ≈ 0.8–1.0` meets the soft `< 1` (proto 1.45).
3. **Cost-push size** — ENERGY +30 % (8q) gives CPI `+72 %` @12m and GDP `−13 %`
   at month 42. Prototype: CPI +2.0 % @12m, GDP −2.7 % month 22. The shock hits
   `lp*` and `p_imp` with no extra attenuation; tightness then compounds.
4. **Credit-on IRFs** — stub `V_RE` is earnings-frozen, so every rate rise
   tightens collateral (QUESTIONS T2.23).
5. **Catastrophe month-1 sign** — same-month claims FD (QUESTIONS T2.23).

## Tuning proposals (not applied)

- **Bullwhip / SEMIS:** raise stock-mode `cover_scale` or SEMIS leak slightly,
  or a stronger inventory-accelerator on upstream stock sectors — only with an
  ADR. Goal: SEMIS sd(yoy) > AUTOS > STAPLES.
- **CONSTRUCT in the top 5:** larger residential `rate_semi` or `phi_accelerator`
  would lift I and CONSTRUCT; conflicts with the already-low monetary rebound
  if taken too far. Try `unit_scale` 0.0125 in a one-off sweep first.
- **AGRIFOOD too loud:** its leak (0.015) is the highest; a ×0.5 leak on
  AGRIFOOD only would likely drop it out of #1 without touching others.
- **I/GDP:** `phi_accelerator` 1.2 → 1.6 or `anchor_growth` 0.75 → 0.6. Re-run
  the 100-year envelope before accepting.
- **Cost-push:** a `cost_push` scale on `z_cost` (or a lower ENERGY
  `pass_through`) would shrink the 72 % CPI step. Do not add a price-level cap.
- **Credit-on:** update stub `E^e` from `prof_s` (§2.10) so a demand boom raises
  `V_RE`; expect the monetary trough to deepen unless `collateral.down` is cut
  (ADR).

Human gate: pick none / some of the above. Defaults stay as shipped.
