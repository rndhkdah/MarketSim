#!/usr/bin/env python3
"""Aggregate a BEA Use table onto the 18-sector seed. Never called from the core or CI network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketsim.layer1.build_io import CODES  # noqa: E402
from marketsim.layer1.checks import assert_structure  # noqa: E402
from marketsim.layer1.io import load_io  # noqa: E402

COVERAGE_FLOOR = 0.85


def _destinations(code: str, concordance: dict[str, str], splits: dict[str, dict[str, float]]) -> dict[str, float]:
    if code in splits:
        return {k: float(v) for k, v in splits[code].items()}
    if code in concordance:
        return {concordance[code]: 1.0}
    return {}


def aggregate_use_table(
    df,
    concordance: dict[str, str],
    splits: dict[str, dict[str, float]] | None = None,
    codes: tuple[str, ...] = CODES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map a square industry-by-industry Use table onto `codes`.

    `df` is a pandas DataFrame: rows = selling industries, columns = buying
    industries, values = use in current dollars. A ``Gross Output`` column is
    optional; if missing, column sums of the mapped Z plus an implied VA residual
    are *not* used — callers should pass gross output as the DataFrame's
    ``.attrs['gross_output']`` mapping or a ``Gross Output`` row.

    Returns ``(Z, gross_output, coverage)`` where coverage is the share of
    source gross output that landed on a mapped industry.
    """
    splits = splits or {}
    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    z = np.zeros((n, n))
    go = np.zeros(n)

    go_row = None
    if "Gross Output" in getattr(df, "index", []):
        go_row = df.loc["Gross Output"]
    go_attr = getattr(df, "attrs", {}).get("gross_output")

    mapped_go = 0.0
    total_go = 0.0
    columns = list(df.columns)
    rows = [r for r in df.index if r != "Gross Output"]

    for col in columns:
        dests = _destinations(str(col), concordance, splits)
        col_go = 0.0
        if go_row is not None:
            col_go = float(go_row[col])
        elif go_attr is not None:
            col_go = float(go_attr.get(col, 0.0))
        total_go += col_go
        if dests:
            mapped_go += col_go
        for dst, w_dst in dests.items():
            go[idx[dst]] += w_dst * col_go
            for row in rows:
                srcs = _destinations(str(row), concordance, splits)
                val = float(df.loc[row, col])
                for src, w_src in srcs.items():
                    z[idx[src], idx[dst]] += val * w_src * w_dst

    coverage = mapped_go / total_go if total_go > 0 else 0.0
    return z, go, coverage


def _load_xlsx(path: Path):
    import pandas as pd

    return pd.read_excel(path, index_col=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate a BEA Use table (local file only)")
    p.add_argument("--xlsx", help="Local Use-table workbook")
    p.add_argument("--out", default=None, help="Destination io_table_bea.json")
    p.add_argument("--validate", action="store_true", help="Run Layer-1 checks through load_io")
    p.add_argument("--coverage-floor", type=float, default=COVERAGE_FLOOR)
    p.add_argument("--config-dir", default=str(ROOT / "config"))
    args = p.parse_args(argv)

    if not args.xlsx:
        print(
            "No workbook given. Download the BEA Use table from https://apps.bea.gov/ "
            "and re-run with --xlsx PATH. See claude/plan/reports/bea-howto.md.",
            file=sys.stderr,
        )
        return 2

    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        print(f"workbook not found: {xlsx} (no network fetch is attempted)", file=sys.stderr)
        return 2

    try:
        df = _load_xlsx(xlsx)
    except Exception as exc:  # openpyxl / pandas
        print(f"failed to read {xlsx}: {exc}", file=sys.stderr)
        return 2

    concordance = df.attrs.get("concordance") or {}
    splits = df.attrs.get("splits") or {}
    if not concordance and "concordance" in df.columns:
        raise SystemExit("pass concordance via DataFrame.attrs or a sidecar")

    z, go, coverage = aggregate_use_table(df, concordance, splits)
    if coverage < args.coverage_floor:
        print(f"coverage {coverage:.3f} below floor {args.coverage_floor}", file=sys.stderr)
        return 1

    if args.validate:
        from marketsim.core.config import load_config

        cfg = load_config(args.config_dir)
        io = load_io(Path(args.config_dir) / "io_table.json")
        assert_structure(io, cfg)
        print(f"validated seed table via load_io (coverage of this workbook: {coverage:.3f})")

    if args.out:
        np.savez(args.out, Z=z, go=go, coverage=np.array([coverage]))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
