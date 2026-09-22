# Run recording and visualisation (T10.16 / T10.17)

Two pieces: a **recorder** that samples a stepping `World` into monthly series, and a
**dashboard** that renders a recording as one self-contained HTML page. No plotting
dependency, no network at view time — the page opens from disk.

Series names match `tests/golden/data/aggregate_baseline.npz`, so a recording and a
committed golden load through the same reader and overlay key by key.

## Recorder

`marketsim.viz.recorder` is a caller-side reader, never a `World` module: a module
would enter `World.to_state()` and change every `state_hash`. Recording a run and
running it plain give the same hash (asserted in `tests/unit/viz/test_recorder.py`).

| Call | Does |
|---|---|
| `RunRecorder(world)` / `.sample()` | append one row per closed month |
| `record_run(world, n_ticks)` | step `n_ticks` days, sampling each tick → `Recording` |
| `write_run(path, rec)` | `<stem>.npz` (arrays) + `<stem>.json` (metadata, units, news) |
| `read_run(path)` | read back a recorder artifact |
| `read_series(path, prefix="")` | read any run-shaped `.npz`, including the goldens |

Recorded series, all per closed month:

| Group | Series | Units |
|---|---|---|
| Output | `gdp` (= `Aggregates.gdp_prod_real`), `gdp_exp_real`, `gdp_prod_nom`, `gdp_exp_nom` | cr/month |
| Prices | `cpi`, `core_cpi` | index, 1.0 at baseline |
| Labour | `u`, `w` | fraction of the labour force; cr/person/month |
| Policy | `r`, `pi12` | annual decimals |
| Balances | `govt_balance`, `trade_balance` | cr/month, + surplus |
| Ratios | `utilisation`, `debt_gdp`, `saving_rate`, `leverage` | fractions |
| Per sector | `x`, `p` | `(months, 18)` |

`gdp`, `cpi`, `u`, `r`, `x` are the same numbers `RealEconomy.step_month()` returns, so a
recording made with the golden's settings reproduces `aggregate_baseline.npz` bitwise.

The recorder reads the economy directly. It is a reporting tool, **not** an observation
path for agents — nothing here is published through `observe()`, so the T6.22 hidden-state
whitelist is untouched.

## Dashboard

`marketsim.viz.dashboard.render(rec, golden=None)` returns the page as a string;
`write_dashboard(path, rec, …)` writes it. Output is a pure function of the recording
(no timestamps, sorted iteration), so two renders are byte-identical.

Sections: a KPI row, **Macro** (one measure per axis — never two scales on one plot),
**Run vs golden** when a baseline is supplied (overlay plus a relative-deviation strip
with the gate tolerance shaded), **Sector output** and **Sector prices** as small
multiples, and **Events** from the recorded news. Every section carries a table view, so
no value is reachable only by hovering.

Deviation tolerances come from gate P5-1 (`05-phase5-firms.md` §5.1): GDP 1 %, sector
output 2 %, CPI 0.5 %.

## CLI

```bash
# record ten years and compare against the Phase-3 baseline
python scripts/plot_run.py --ticks 2520 --seed 0 \
  --override dynamics.banks.mode=passthrough \
  --record out/decade --golden tests/golden/data/aggregate_baseline.npz \
  --golden-label "phase-3 baseline" --out out/decade.html

# re-render an existing recording, no simulation
python scripts/plot_run.py --run out/decade.npz --out out/decade.html

# overlay the r1 parity golden's no-shock family
python scripts/plot_run.py --ticks 756 --golden tests/golden/data/r1_phase2.npz \
  --golden-prefix ns_ --out out/r1.html
```

`--ticks` is simulated days (21 = 1 month, 252 = 1 year). `--override` takes repeatable
dotted config paths; the committed goldens are recorded with
`dynamics.banks.mode=passthrough`. `--no-sfc` skips the monthly SFC assertion for speed —
leave it on unless you are benchmarking.

Ten years with SFC on records and renders in about 5 s. `tests/unit/viz/` and
`tests/unit/scripts/test_plot_run.py` are the CI smoke for this page.
