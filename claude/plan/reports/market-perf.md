# T6.23 — Market performance (gate 8)

Measured 2026-09-20. 25 instruments (18 `EQ:NPC:*` + `GB_*` + `CORP_POOL` +
`OIL`/`METALS`/`GRAINS`), no agents. Each tick: background flow (`stream flow`),
vector impact kernel, mispricing `ξ = I + s + n`.

| run | ticks | warmup | wall | ticks/s |
|---|---|---|---|---|
| `scripts/bench.py --market --ticks 2520` | 2,520 | 42 | 0.167 s | **15,115** |
| `test_perf_market` same seed replay | 2,520 | 42 | — | hash-stable |

Spec floor: **3,000 ticks/s**. Headroom ≈ 5× on this host. Same seed → same
`sha256(q ‖ ξ)` digest; a different seed differs.
