# marketsim

Headless, deterministic **economy + financial-market** engine. Phase 0 is the installable package,
Layer-1 IO backbone, core utilities (clock, RNG, hashing, Erlang), and a `World` walking skeleton.

This tree is a **separate project** from the Victoria 3 harness at the repo root. Run Vic3 tests with
`uv run pytest` from the repo root; run this package from `marketsim/`.

## Setup

```bash
cd marketsim
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

## Layout

```
AGENTS.md          agent rules (this package, not the Vic3 root)
claude/plan/       145-card plan + reference prototype (not production code)
config/            YAML/JSON — every parameter lives here
src/marketsim/     library
scripts/           CLIs (build_io, derive_betas, fetch_bea_io, vic3_compare)
tests/
```

## Layer-1 goldens (reconstructed seed)

A is 18×18, nnz 219, ρ(A) = 0.536, output multipliers 1.58–2.77. Per 100 cr of final demand:
gross output 209.3, wages 51.82, GOS 48.18. Regime demo seed 7: corr(equity, bonds) −0.74 demand /
+0.90 supply; 13/18 sectors flip sign.

There is no legacy `claude/marketsim` tree; goldens are the published invariants, not a bit-identical
`betas.md`.

## Next

`claude/plan/PROGRESS.md` — lowest unchecked card whose dependencies are done (T2.01 after P0).
