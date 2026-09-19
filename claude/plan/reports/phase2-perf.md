# Phase 2 performance baseline (T2.25)

R = 1, `ledger.debug_journal: false`, SFC assertion off for the bench
(`scripts/bench.py`). One real month-step per 21 daily ticks.

| run | ticks | warmup | wall | ticks/s |
|---|---|---|---|---|
| CLI `--ticks 5040` | 5,040 | 42 | 0.474 s | 10,642 |
| `test_perf_smoke` | 2,520 | 42 | 0.238 s | 10,607 |

Spec floor: **5,000 ticks/s**. Headroom ≈ 2.1× on this host. The month-step
itself stays under the T2.17 5 ms gate; empty intra-month ticks are cheap.
