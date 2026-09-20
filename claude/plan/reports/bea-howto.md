# BEA Use-table fetch (T0.09)

The engine never calls the network. Download the workbook yourself from
[apps.bea.gov](https://apps.bea.gov/) (Industry → Input-Output → Use table,
after-redefinitions, latest year) and run:

```bash
cd marketsim
python scripts/fetch_bea_io.py --xlsx /path/to/use.xlsx --out config/io_table_bea.json --validate
```

`--validate` re-runs the Layer-1 structural suite through `load_io` against the
**seed** table (the BEA file is not substituted until `world.io_source: bea` and
the file exists). A coverage share below 0.85 exits non-zero.

## Expected output

`aggregate_use_table` returns the 18×18 intermediate matrix `Z`, a gross-output
vector, and the share of source gross output that mapped onto a seed sector.
`A = Z / go` (column-wise) is written only after a human reviews the
concordance — final-demand vectors stay seeded until T8.04.

## Known follow-up

The seed `final_demand` vectors and `fd_weights` (0.62 / 0.19 / 0.13 / 0.06) are
**not** replaced by this script. PCE / investment / government / export
concordance lives in `scripts/fetch_bea_io.py` and `reports/calibration.md`
(T8.04 / ADR-015). Proposed weights are not written into the seed table.

## Without network

```bash
python scripts/fetch_bea_io.py
# exits 2 with a pointer at this file; no request is made
```
