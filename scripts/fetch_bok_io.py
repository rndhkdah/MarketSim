#!/usr/bin/env python3
"""Aggregate a Bank of Korea / KOSIS industry-by-industry table onto the 18-sector seed.

Never imported by the core. Never called from CI with a network fetch. Download the
workbook yourself, then pass ``--xlsx``.

Sources (human download; the script only prints these URLs):

- Bank of Korea Input-Output Tables (산업연관표), ECOS:
  https://ecos.bok.or.kr/  → 통계검색 → 2.2 산업연관표 → year → 파일다운로드
  (industry-by-industry / 산업×산업, producers' prices). English landing page:
  https://www.bok.or.kr/eng/bbs/E0000634/view.do?menuNo=400069&nttId=10083839
- KOSIS republishes the same BOK tables:
  https://kosis.kr/  → 국민계정 → 산업연관표 (작성기관: 한국은행)

Classification in ``config/kr/io_concordance.yaml``: BOK 2015/2020 통합대분류 (33)
and key 통합중분류 (83) names, plus KSIC Rev. 10 sections and manufacturing
divisions. Destinations are ``marketsim.layer1.build_io.CODES`` only — no second
18-sector list.

See ``config/kr/README.md``. ADR-015: this overlay is optional flavour; do not
replace the US seed by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketsim.layer1.build_io import CODES  # noqa: E402

COVERAGE_FLOOR = 0.85
DEFAULT_CONCORDANCE = ROOT / "config" / "kr" / "io_concordance.yaml"
HOWTO_URL_BOK = "https://ecos.bok.or.kr/"
HOWTO_URL_KOSIS = "https://kosis.kr/"
HOWTO_URL_BOK_EN = "https://www.bok.or.kr/eng/bbs/E0000634/view.do?menuNo=400069&nttId=10083839"

# Row aliases for the gross-output vector (current KRW / year in the source table).
_GO_ROW_ALIASES = frozenset(
    {
        "gross output",
        "gross_output",
        "total output",
        "총산출",
        "총산출액",
        "산출액",
        "국내총산출",
        "총 산출",
    }
)

# Margin / total stubs that appear beside industries in published BOK workbooks.
_SKIP_LABELS = frozenset(
    {
        "중간수요계",
        "중간수요 계",
        "중간수요합계",
        "중간수요",
        "최종수요계",
        "최종수요 계",
        "최종수요합계",
        "최종수요",
        "총수요",
        "총산출계",
        "부가가치계",
        "부가가치",
        "피용자보수",
        "영업잉여",
        "intermediate demand",
        "intermediate total",
        "final demand",
        "value added",
        "total demand",
    }
)

_CODE_PREFIX = re.compile(r"^([A-Za-z]{1,2}|\d{1,3})[.)\s]+(.+)$")


def _is_short_code(label: str) -> bool:
    t = label.strip()
    return (len(t) <= 3 and t.isdigit()) or (1 <= len(t) <= 2 and t.isalpha())


def _label_candidates(label: object) -> list[str]:
    """Workbook header variants: ``'01. 농림수산품'`` → code + name."""
    s = str(label).strip()
    out: list[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)

    _add(s)
    _add(s.casefold())
    match = _CODE_PREFIX.match(s)
    if match:
        code, name = match.group(1), match.group(2).strip()
        _add(code)
        _add(code.casefold())
        if code.isdigit():
            _add(str(int(code)))
            _add(code.zfill(2))
        _add(name)
        _add(name.casefold())
    elif s.isdigit():
        _add(str(int(s)))
        _add(s.zfill(2))
    return out


def _is_go_row(label: object) -> bool:
    return str(label).strip().casefold() in _GO_ROW_ALIASES


def _is_skip(label: object) -> bool:
    raw = str(label).strip()
    if raw.casefold() in _SKIP_LABELS or raw in _SKIP_LABELS:
        return True
    return raw.casefold() in {s.casefold() for s in _SKIP_LABELS}


def _destinations(
    code: str,
    concordance: dict[str, str],
    splits: dict[str, dict[str, float]],
) -> dict[str, float]:
    for key in _label_candidates(code):
        if key in splits:
            return {k: float(v) for k, v in splits[key].items()}
        if key in concordance:
            return {concordance[key]: 1.0}
        folded = key.casefold()
        if folded in splits:
            return {k: float(v) for k, v in splits[folded].items()}
        if folded in concordance:
            return {concordance[folded]: 1.0}
    return {}


def load_bok_concordance(
    path: str | Path,
    *,
    classification: str = "bok33",
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Load ``config/kr/io_concordance.yaml`` into BEA-style concordance + splits.

    Short codes (``01``, ``C``) are registered unprefixed only for ``classification``
    so BOK-33 ``10`` (fabricated metal) does not collide with KSIC ``10`` (food).
    Korean/English names are always registered. Units: weights are shares (sum 1).
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError(f"{path} must contain a top-level 'sources' list")

    concordance: dict[str, str] = {}
    splits: dict[str, dict[str, float]] = {}
    code_set = set(CODES)

    for entry in raw["sources"]:
        if not isinstance(entry, dict):
            raise ValueError("each concordance source must be a mapping")
        weights_raw = entry.get("weights") or {}
        weights = {str(k): float(v) for k, v in weights_raw.items()}
        if not weights:
            raise ValueError(f"source {entry.get('id')!r} has empty weights")
        unknown = set(weights) - code_set
        if unknown:
            raise ValueError(f"source {entry.get('id')!r} maps to unknown CODES {sorted(unknown)}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"source {entry.get('id')!r} weights sum to {total}, not 1")

        own = str(entry.get("classification") or "")
        keys: list[str] = []
        ident = str(entry.get("id") or "").strip()
        if ident:
            keys.append(ident)
        for item in entry.get("keys") or []:
            key = str(item).strip()
            if not key:
                continue
            if own:
                keys.append(f"{own}:{key}")
            if _is_short_code(key):
                if own == classification or not own:
                    keys.append(key)
                    if key.isdigit():
                        keys.append(str(int(key)))
                        keys.append(key.zfill(2))
            else:
                keys.append(key)
                keys.append(key.casefold())

        unique_keys: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                unique_keys.append(key)
        if len(weights) == 1:
            dest = next(iter(weights))
            for key in unique_keys:
                concordance[key] = dest
        else:
            for key in unique_keys:
                splits[key] = dict(weights)

    return concordance, splits


def aggregate_bok_use_table(
    df: Any,
    concordance: dict[str, str],
    splits: dict[str, dict[str, float]] | None = None,
    codes: tuple[str, ...] = CODES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map a square industry-by-industry BOK/KOSIS Use table onto ``codes``.

    ``df`` is a pandas DataFrame: rows = selling industries, columns = buying
    industries, values = use in current KRW (or any current-price unit). A
    gross-output row (``Gross Output`` / ``총산출``) is optional; callers may
    instead pass ``df.attrs['gross_output']`` as a column→value mapping.

    Returns ``(Z, gross_output, coverage)``. Coverage is the share of source
    gross output that landed on a mapped industry. Margin/total stubs are
    ignored and do not enter the coverage denominator.
    """
    splits = splits or {}
    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    z = np.zeros((n, n))
    go = np.zeros(n)

    go_row = None
    for label in getattr(df, "index", []):
        if _is_go_row(label):
            go_row = df.loc[label]
            break
    go_attr = getattr(df, "attrs", {}).get("gross_output")

    mapped_go = 0.0
    total_go = 0.0
    columns = [c for c in df.columns if not _is_skip(c) and not _is_go_row(c)]
    rows = [r for r in df.index if not _is_go_row(r) and not _is_skip(r)]

    for col in columns:
        dests = _destinations(str(col), concordance, splits)
        col_go = 0.0
        if go_row is not None:
            val = go_row[col]
            col_go = float(val) if np.isfinite(val) else 0.0
        elif go_attr is not None:
            col_go = float(go_attr.get(col, 0.0))
        total_go += col_go
        if dests:
            mapped_go += col_go
        for dst, w_dst in dests.items():
            go[idx[dst]] += w_dst * col_go
            for row in rows:
                srcs = _destinations(str(row), concordance, splits)
                raw = df.loc[row, col]
                val = float(raw) if np.isfinite(raw) else 0.0
                for src, w_src in srcs.items():
                    z[idx[src], idx[dst]] += val * w_src * w_dst

    coverage = mapped_go / total_go if total_go > 0 else 0.0
    return z, go, coverage


def _load_xlsx(path: Path, sheet: str | int | None = None):
    import pandas as pd

    kwargs: dict[str, Any] = {"index_col": 0}
    if sheet is not None:
        kwargs["sheet_name"] = sheet
    return pd.read_excel(path, **kwargs)


def _print_howto() -> None:
    print(
        "No workbook given. Download the Bank of Korea Input-Output table from\n"
        f"  {HOWTO_URL_BOK}\n"
        "  (통계검색 → 2.2 산업연관표 → year → 파일다운로드; industry-by-industry)\n"
        f"or KOSIS {HOWTO_URL_KOSIS} (국민계정 → 산업연관표, 작성기관 한국은행).\n"
        f"English note: {HOWTO_URL_BOK_EN}\n"
        "Re-run with --xlsx PATH. See config/kr/README.md. No network request is made.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a Bank of Korea / KOSIS Use table (local file only)"
    )
    parser.add_argument("--xlsx", help="Local industry-by-industry workbook")
    parser.add_argument("--sheet", default=None, help="Workbook sheet name or index")
    parser.add_argument("--out", default=None, help="Destination .npz (Z, go, coverage)")
    parser.add_argument("--concordance", default=str(DEFAULT_CONCORDANCE), help="YAML concordance")
    parser.add_argument(
        "--classification",
        default="bok33",
        help="Short-code namespace: bok33 (default), bok83, or ksic",
    )
    parser.add_argument("--validate", action="store_true", help="Run Layer-1 checks on the seed table")
    parser.add_argument("--coverage-floor", type=float, default=COVERAGE_FLOOR)
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    args = parser.parse_args(argv)

    if not args.xlsx:
        _print_howto()
        return 2

    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        print(f"workbook not found: {xlsx} (no network fetch is attempted)", file=sys.stderr)
        return 2

    conc_path = Path(args.concordance)
    if not conc_path.exists():
        print(f"concordance not found: {conc_path}", file=sys.stderr)
        return 2

    try:
        concordance, splits = load_bok_concordance(conc_path, classification=args.classification)
    except ValueError as exc:
        print(f"invalid concordance: {exc}", file=sys.stderr)
        return 2

    sheet: str | int | None = args.sheet
    if sheet is not None and str(sheet).isdigit():
        sheet = int(sheet)

    try:
        df = _load_xlsx(xlsx, sheet=sheet)
    except Exception as exc:  # openpyxl / pandas
        print(f"failed to read {xlsx}: {exc}", file=sys.stderr)
        return 2

    extra_c = df.attrs.get("concordance") or {}
    extra_s = df.attrs.get("splits") or {}
    concordance = {**concordance, **extra_c}
    splits = {**splits, **extra_s}

    z, go, coverage = aggregate_bok_use_table(df, concordance, splits)
    if coverage < args.coverage_floor:
        print(f"coverage {coverage:.3f} below floor {args.coverage_floor}", file=sys.stderr)
        return 1

    if args.validate:
        from marketsim.core.config import load_config
        from marketsim.layer1.checks import assert_structure
        from marketsim.layer1.io import load_io

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
