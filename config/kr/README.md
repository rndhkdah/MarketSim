# Korea-flavoured overlay (T8.11, optional)

This directory is an **override bundle**, not the default calibration. The shipped
US seed (`config/io_table.json`, `config/*.yaml`) stays in place. ADR-015: do not
apply this as a silent retune of Layer-1 elasticities.

Destinations are the existing 18 codes from `marketsim.layer1.build_io.CODES`.
There is no second sector list.

## 1. Download a local workbook (you do this; the engine never will)

The script does **not** call the network. If you omit `--xlsx` it prints a URL and
exits 2.

1. Bank of Korea ECOS: <https://ecos.bok.or.kr/> → 통계검색 → **2.2 산업연관표** →
   year → 파일다운로드. Prefer the **industry-by-industry** (산업×산업) table at
   통합대분류 (33) or 통합중분류 (83), producers' prices.
2. Or KOSIS: <https://kosis.kr/> → 국민계정 → 산업연관표 (작성기관: 한국은행).
3. English announcement of the 2020 benchmark:
   <https://www.bok.or.kr/eng/bbs/E0000634/view.do?menuNo=400069&nttId=10083839>

Save a square block whose index/columns are industry names (Korean or English) or
codes, with an optional `총산출` / `Gross Output` row.

## 2. Aggregate onto the 18 sectors

```bash
python scripts/fetch_bok_io.py --xlsx /path/to/bok.xlsx \
  --out config/kr/io_table_bok.npz \
  --concordance config/kr/io_concordance.yaml
```

- Default `--classification bok33` (BOK 33-sector short codes). Use `bok83` or
  `ksic` when the workbook is a KOSIS industry extract.
- Coverage below 0.85 exits 1. `--validate` re-runs Layer-1 checks on the
  **seed** table (same as the BEA script); it does not swap `io_table.json`.
- Output is `Z` (18×18 intermediates), `go` (gross output), `coverage`. Review
  `A = Z / go` before anything is used as a world table.

## 3. How a human loads the overlay

Do **not** replace `config/` with `config/kr/`. `load_config` still needs the US
seed files (`sectors.yaml`, `edges.yaml`, `dynamics.yaml`, …).

1. Keep `config/` as the seed. Leave `world.io_source: seed`. There is no `bok`
   enum on `WorldSettings` (see `NOTES.md`); do not edit `config.py` to add one.
2. After reviewing the `.npz`, a human may write a *local* `io_table.json` copy
   and point the process at that directory. Do not commit over the seed.
3. Apply only knobs that already exist, via dotted overrides:

```python
from marketsim.world import World
from marketsim.real.economy import make_real_world

# Flavour knobs that already exist on DynamicsConfig / PolicyFile:
overrides = {
    "dynamics.fiscal.vat": 0.10,          # 부가가치세법; default seed is 0
    "dynamics.fiscal.debt_to_gdp": 0.50,  # D2 ~50 %; seed 0.60
    "dynamics.labour.u_star": 0.035,      # BOK/OECD-style NAIRU; seed 0.05
}
world = World.create("config", overrides=overrides)

# π* is a constructor argument, not a YAML key (config.py extra=forbid):
eco = make_real_world("config", pi_star=0.02, check_sfc=True)
```

4. Optional calendar snippet: merge `config/kr/policy.yaml` into `config/policy.yaml`
   only if you want the BOK meeting comments in-tree. Values match the D15
   defaults (8 meetings, 25 bp) on purpose.

Proposed values that would need a new config key, and the ones above with
citations, live in `NOTES.md`. Layer-1 elasticities stay on the US seed.
