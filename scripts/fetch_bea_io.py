#!/usr/bin/env python3
"""Aggregate a local BEA Use table and NIPA final-demand columns onto 18 sectors.

Never called from the core. Never fetches the network. Seed ``config/io_table.json``
is not overwritten (ADR-015 / T8.04).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketsim.layer1.build_io import CODES, FD_KEYS  # noqa: E402
from marketsim.layer1.checks import assert_structure  # noqa: E402
from marketsim.layer1.io import load_io  # noqa: E402

COVERAGE_FLOOR = 0.85
SEED_IO_TABLE = ROOT / "config" / "io_table.json"
SEED_FD_WEIGHTS: dict[str, float] = {
    "HOUSEHOLD": 0.62,
    "INVESTMENT": 0.19,
    "GOVT": 0.13,
    "EXPORTS": 0.06,
}

# NIPA-style column titles on a Use table (after redefinitions) or a sidecar sheet.
# Prefer the aggregate name; otherwise sum the listed components (no double count).
NIPA_FD_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "HOUSEHOLD": (
        "Personal consumption expenditures",
        "PCE",
        "Household consumption",
        "T001",
    ),
    "INVESTMENT": (
        "Private fixed investment",
        "Gross private fixed investment",
        "Nonresidential and residential fixed investment",
        "T006",
    ),
    "GOVT": (
        "Government consumption expenditures and gross investment",
        "Government",
        "T013+T016",
    ),
    "EXPORTS": (
        "Exports of goods and services",
        "Exports",
        "T008",
    ),
}
NIPA_FD_COMPONENT_COLUMNS: dict[str, tuple[str, ...]] = {
    "GOVT": (
        "Federal",
        "Federal government",
        "Federal government consumption expenditures and gross investment",
        "State and local",
        "State and local government",
        "State and local government consumption expenditures and gross investment",
        "T013",
        "T016",
    ),
    "INVESTMENT": (
        "Nonresidential fixed investment",
        "Residential fixed investment",
        "T006A",
        "T006B",
    ),
}

# Classification map: BEA summary / 71-industry / NIPA line names → 18 CODES.
# Conjunction names use equal weights (not a fitted elasticity). Destination
# weights on every row sum to 1. Prefer 71-industry names over 15-industry residuals.
_DEFAULT_NIPA_ROWS: tuple[tuple[str, str | dict[str, float]], ...] = (
    # --- 15-industry fallbacks ---
    ("Agriculture, forestry, fishing, and hunting", "AGRIFOOD"),
    ("Mining", {"ENERGY": 0.5, "MATERIALS": 0.5}),
    ("Utilities", "UTILITIES"),
    ("Construction", "CONSTRUCT"),
    (
        "Manufacturing",
        {
            "ENERGY": 0.125,
            "MATERIALS": 0.125,
            "AGRIFOOD": 0.125,
            "SEMIS": 0.125,
            "AUTOS": 0.125,
            "STAPLES": 0.125,
            "DISCRET": 0.125,
            "CAPGOODS": 0.125,
        },
    ),
    ("Wholesale trade", "BIZSVC"),
    ("Retail trade", "DISCRET"),
    ("Transportation and warehousing", "TRANSPORT"),
    ("Information", {"SOFTWARE": 0.5, "TELECOM": 0.5}),
    (
        "Finance, insurance, real estate, rental, and leasing",
        {"BANKS": 1.0 / 3.0, "INSURANCE": 1.0 / 3.0, "REALESTATE": 1.0 / 3.0},
    ),
    ("Professional and business services", "BIZSVC"),
    ("Educational services, health care, and social assistance", "HEALTH"),
    ("Arts, entertainment, recreation, accommodation, and food services", "DISCRET"),
    ("Other services, except government", "DISCRET"),
    ("Government", "BIZSVC"),
    # --- 71-industry (BEA Use, after redefinitions) ---
    ("Farms", "AGRIFOOD"),
    ("Forestry, fishing, and related activities", "AGRIFOOD"),
    ("Oil and gas extraction", "ENERGY"),
    ("Mining, except oil and gas", "MATERIALS"),
    ("Support activities for mining", "ENERGY"),
    ("Wood products", "MATERIALS"),
    ("Nonmetallic mineral products", "MATERIALS"),
    ("Primary metals", "MATERIALS"),
    ("Fabricated metal products", "MATERIALS"),
    ("Machinery", "CAPGOODS"),
    ("Computer and electronic products", "SEMIS"),
    ("Electrical equipment, appliances, and components", "CAPGOODS"),
    ("Motor vehicles, bodies and trailers, and parts", "AUTOS"),
    ("Other transportation equipment", "AUTOS"),
    ("Furniture and related products", "DISCRET"),
    ("Miscellaneous manufacturing", "DISCRET"),
    ("Food and beverage and tobacco products", "AGRIFOOD"),
    ("Textile mills and textile product mills", "DISCRET"),
    ("Apparel and leather and allied products", "DISCRET"),
    ("Paper products", "MATERIALS"),
    ("Printing and related support activities", "BIZSVC"),
    ("Petroleum and coal products", "ENERGY"),
    ("Chemical products", "MATERIALS"),
    ("Plastics and rubber products", "MATERIALS"),
    ("Motor vehicle and parts dealers", "AUTOS"),
    ("Food and beverage stores", "STAPLES"),
    ("General merchandise stores", "DISCRET"),
    ("Other retail", "DISCRET"),
    ("Air transportation", "TRANSPORT"),
    ("Rail transportation", "TRANSPORT"),
    ("Water transportation", "TRANSPORT"),
    ("Truck transportation", "TRANSPORT"),
    ("Transit and ground passenger transportation", "TRANSPORT"),
    ("Pipeline transportation", "TRANSPORT"),
    ("Other transportation and support activities", "TRANSPORT"),
    ("Warehousing and storage", "TRANSPORT"),
    ("Publishing industries, except internet (includes software)", "SOFTWARE"),
    ("Publishing industries", "SOFTWARE"),
    ("Motion picture and sound recording industries", "DISCRET"),
    ("Broadcasting and telecommunications", "TELECOM"),
    ("Data processing, internet publishing, and other information services", "SOFTWARE"),
    ("Federal Reserve banks, credit intermediation, and related activities", "BANKS"),
    ("Securities, commodity contracts, and investments", "BANKS"),
    ("Insurance carriers and related activities", "INSURANCE"),
    ("Funds, trusts, and other financial vehicles", "BANKS"),
    ("Housing", "REALESTATE"),
    ("Real estate", "REALESTATE"),
    ("Other real estate", "REALESTATE"),
    ("Rental and leasing services and lessors of intangible assets", "REALESTATE"),
    ("Legal services", "BIZSVC"),
    ("Computer systems design and related services", "SOFTWARE"),
    ("Miscellaneous professional, scientific, and technical services", "BIZSVC"),
    ("Management of companies and enterprises", "BIZSVC"),
    ("Administrative and support services", "BIZSVC"),
    ("Waste management and remediation services", "UTILITIES"),
    ("Educational services", "BIZSVC"),
    ("Ambulatory health care services", "HEALTH"),
    ("Hospitals", "HEALTH"),
    ("Nursing and residential care facilities", "HEALTH"),
    ("Social assistance", "HEALTH"),
    ("Performing arts, spectator sports, museums, and related activities", "DISCRET"),
    ("Amusements, gambling, and recreation industries", "DISCRET"),
    ("Accommodation", "DISCRET"),
    ("Food services and drinking places", "DISCRET"),
    ("Other services", "DISCRET"),
    ("Federal general government", "BIZSVC"),
    ("Federal government enterprises", "BIZSVC"),
    ("State and local general government", "BIZSVC"),
    ("State and local government enterprises", "BIZSVC"),
    # --- NIPA PCE (Table 2.4.5-style) ---
    ("Food and beverages purchased for off-premises consumption", "STAPLES"),
    ("Clothing and footwear", "DISCRET"),
    ("Gasoline and other energy goods", "ENERGY"),
    ("Other nondurable goods", "STAPLES"),
    ("Motor vehicles and parts", "AUTOS"),
    ("Furnishings and durable household equipment", "DISCRET"),
    ("Recreational goods and vehicles", "DISCRET"),
    ("Other durable goods", "DISCRET"),
    ("Household utilities", "UTILITIES"),
    ("Health care", "HEALTH"),
    ("Transportation services", "TRANSPORT"),
    ("Recreation services", "DISCRET"),
    ("Food services and accommodations", "DISCRET"),
    ("Financial services and insurance", {"BANKS": 0.5, "INSURANCE": 0.5}),
    ("Other services (PCE)", "BIZSVC"),
    ("Nonprofit institutions serving households", "BIZSVC"),
    # --- NIPA private fixed investment ---
    ("Nonresidential structures", "CONSTRUCT"),
    ("Equipment", "CAPGOODS"),
    ("Information processing equipment", "SEMIS"),
    ("Industrial equipment", "CAPGOODS"),
    ("Transportation equipment", "AUTOS"),
    ("Other equipment", "CAPGOODS"),
    ("Intellectual property products", "SOFTWARE"),
    ("Software", "SOFTWARE"),
    ("Research and development", {"SOFTWARE": 0.5, "BIZSVC": 0.5}),
    ("Residential", "CONSTRUCT"),
    ("Residential structures", "CONSTRUCT"),
    # --- NIPA government ---
    ("National defense", "CAPGOODS"),
    ("Nondefense", "BIZSVC"),
    ("Government gross investment", {"CONSTRUCT": 0.5, "CAPGOODS": 0.5}),
    # --- NIPA / Census export end-use ---
    ("Foods, feeds, and beverages", "AGRIFOOD"),
    ("Industrial supplies and materials", "MATERIALS"),
    ("Capital goods, except automotive", "CAPGOODS"),
    ("Automotive vehicles, engines, and parts", "AUTOS"),
    ("Automotive vehicles, parts, and engines", "AUTOS"),
    ("Consumer goods, except food and automotive", "DISCRET"),
    ("Petroleum and products", "ENERGY"),
    ("Exports of goods", "MATERIALS"),
    ("Exports of services", "BIZSVC"),
)


def _row_to_weights(dest: str | Mapping[str, float]) -> dict[str, float]:
    if isinstance(dest, Mapping):
        return {str(k): float(v) for k, v in dest.items()}
    return {str(dest): 1.0}


def default_nipa_concordance() -> dict[str, dict[str, float]]:
    """BEA NIPA / Use-table industry names → destination weights on ``CODES``."""
    out: dict[str, dict[str, float]] = {}
    for name, dest in _DEFAULT_NIPA_ROWS:
        out[name] = _row_to_weights(dest)
    return out


DEFAULT_NIPA_CONCORDANCE: dict[str, dict[str, float]] = default_nipa_concordance()


def identity_concordance(codes: tuple[str, ...] = CODES) -> dict[str, dict[str, float]]:
    """Each ``CODES`` member maps to itself with weight 1."""
    return {c: {c: 1.0} for c in codes}


def _lookup_key(code: str, mapping: Mapping[str, Any]) -> str | None:
    if code in mapping:
        return code
    folded = {str(k).casefold(): str(k) for k in mapping}
    return folded.get(code.casefold())


def destination_weights(
    code: str,
    concordance: Mapping[str, str | Mapping[str, float]],
    splits: Mapping[str, Mapping[str, float]] | None = None,
    codes: tuple[str, ...] = CODES,
) -> dict[str, float]:
    """Map one source label onto destination sector weights (sum to 1, or empty).

    Units: destination weights are shares (dimensionless) of that source's dollars.
    """
    splits = splits or {}
    split_key = _lookup_key(code, splits)
    if split_key is not None:
        raw = {str(k): float(v) for k, v in splits[split_key].items()}
        return _normalise_row(code, raw, codes, normalize=True)
    conc_key = _lookup_key(code, concordance)
    if conc_key is None:
        return {}
    dest = concordance[conc_key]
    raw = _row_to_weights(dest)
    return _normalise_row(code, raw, codes, normalize=True)


def _destinations(
    code: str,
    concordance: dict[str, str] | Mapping[str, str | Mapping[str, float]],
    splits: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Backward-compatible helper used by ``aggregate_use_table``."""
    return destination_weights(code, concordance, splits)


def _normalise_row(
    source: str,
    raw: Mapping[str, float],
    codes: tuple[str, ...],
    *,
    normalize: bool,
) -> dict[str, float]:
    allowed = set(codes)
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise ValueError(f"concordance {source!r} maps to unknown codes {unknown}")
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError(f"concordance {source!r} has non-positive weight sum")
    if normalize:
        return {k: float(v) / total for k, v in raw.items()}
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"concordance {source!r} destination weights sum to {total}, not 1")
    return {k: float(v) for k, v in raw.items()}


def compose_concordance(
    concordance: Mapping[str, str | Mapping[str, float]],
    splits: Mapping[str, Mapping[str, float]] | None = None,
    *,
    codes: tuple[str, ...] = CODES,
    normalize: bool = False,
) -> dict[str, dict[str, float]]:
    """Merge 1-to-1 maps and splits. Each source row's destination weights sum to 1."""
    splits = splits or {}
    keys = list(dict.fromkeys([*concordance.keys(), *splits.keys()]))
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        raw = destination_weights(str(key), concordance, splits, codes=codes)
        if not raw:
            continue
        out[str(key)] = _normalise_row(str(key), raw, codes, normalize=normalize or True)
    return out


def assert_concordance_normalized(
    concordance: Mapping[str, Mapping[str, float]],
    *,
    codes: tuple[str, ...] = CODES,
    atol: float = 1e-9,
) -> None:
    """Raise if any source row's destination weights miss 1 or leave ``codes``."""
    allowed = set(codes)
    for src, weights in concordance.items():
        if not weights:
            raise ValueError(f"concordance {src!r} is empty")
        unknown = [k for k in weights if k not in allowed]
        if unknown:
            raise ValueError(f"concordance {src!r} maps to unknown codes {unknown}")
        total = float(sum(weights.values()))
        if abs(total - 1.0) > atol:
            raise ValueError(f"concordance {src!r} destination weights sum to {total}, not 1")


def _as_source_map(
    data: Mapping[str, float] | np.ndarray,
    codes: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(data, Mapping):
        return {str(k): float(v) for k, v in data.items()}
    arr = np.asarray(data, dtype=float)
    if arr.shape != (len(codes),):
        raise ValueError(f"ndarray source must have shape ({len(codes)},); got {arr.shape}")
    return {c: float(arr[i]) for i, c in enumerate(codes)}


def fd_component_weights(totals: Mapping[str, float] | np.ndarray) -> dict[str, float]:
    """Observed C/I/G/X shares. Units: 1. Sum to 1 when the grand total is positive."""
    if isinstance(totals, Mapping):
        raw = {k: float(totals[k]) for k in FD_KEYS}
    else:
        arr = np.asarray(totals, dtype=float)
        if arr.shape != (len(FD_KEYS),):
            raise ValueError(f"totals must have length {len(FD_KEYS)}")
        raw = {k: float(arr[i]) for i, k in enumerate(FD_KEYS)}
    grand = float(sum(raw.values()))
    if grand <= 0.0:
        raise ValueError("final-demand grand total is non-positive")
    return {k: v / grand for k, v in raw.items()}


def aggregate_final_demand(
    pce: Mapping[str, float] | np.ndarray,
    investment: Mapping[str, float] | np.ndarray,
    government: Mapping[str, float] | np.ndarray,
    exports: Mapping[str, float] | np.ndarray,
    concordance: Mapping[str, str | Mapping[str, float]],
    splits: Mapping[str, Mapping[str, float]] | None = None,
    *,
    codes: tuple[str, ...] = CODES,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Map four NIPA-style source vectors onto ``codes``.

    Parameters
    ----------
    pce, investment, government, exports
        Current-dollar maps ``{source_industry: dollars}``, or length-``S`` arrays
        already on ``codes``.
    concordance, splits
        Source → destination sector (or weight map). Same contract as the Use table.

    Returns
    -------
    fd_matrix
        Shape ``(4, S)`` current dollars, row order ``FD_KEYS``
        (HOUSEHOLD, INVESTMENT, GOVT, EXPORTS).
    weights
        Observed C/I/G/X shares of *mapped* dollars; sum to 1 (unit: 1).
    coverage
        Share of source dollars that landed on a mapped industry (unit: 1).
    """
    composed = compose_concordance(concordance, splits, codes=codes, normalize=True)
    assert_concordance_normalized(composed, codes=codes)
    idx = {c: i for i, c in enumerate(codes)}
    n = len(codes)
    fd = np.zeros((len(FD_KEYS), n), dtype=float)
    mapped = 0.0
    total = 0.0
    sources = (
        _as_source_map(pce, codes),
        _as_source_map(investment, codes),
        _as_source_map(government, codes),
        _as_source_map(exports, codes),
    )
    for row, src_map in enumerate(sources):
        for label, dollars in src_map.items():
            val = float(dollars)
            total += val
            dests = composed.get(label)
            if dests is None:
                dests = destination_weights(label, concordance, splits, codes=codes)
            if not dests:
                continue
            mapped += val
            for dst, w in dests.items():
                fd[row, idx[dst]] += val * w
    coverage = mapped / total if total > 0.0 else 0.0
    row_totals = fd.sum(axis=1)
    weights = fd_component_weights(row_totals) if float(row_totals.sum()) > 0.0 else {k: 0.0 for k in FD_KEYS}
    return fd, weights, coverage


def example_fd_arithmetic() -> dict[str, Any]:
    """Synthetic concordance arithmetic used in ``reports/calibration.md`` (not a BEA year)."""
    pce = {"Food": 60.0, "Autos": 20.0, "Housing": 20.0}
    investment = {"Machinery": 15.0, "Construction": 5.0}
    government = {"Health": 10.0, "Defence": 5.0}
    exports = {"Autos": 5.0}
    concordance = {
        "Food": "STAPLES",
        "Autos": "AUTOS",
        "Housing": "REALESTATE",
        "Machinery": "CAPGOODS",
        "Construction": "CONSTRUCT",
        "Health": "HEALTH",
        "Defence": "CAPGOODS",
    }
    fd, weights, coverage = aggregate_final_demand(
        pce, investment, government, exports, concordance
    )
    return {
        "pce": pce,
        "investment": investment,
        "government": government,
        "exports": exports,
        "concordance": concordance,
        "grand_total": 140.0,
        "fd_matrix": fd,
        "weights": weights,
        "coverage": coverage,
        "n_sectors": fd.shape[1],
    }


def extract_nipa_final_demand(
    table: Any,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Pull PCE / I / G / X source maps from workbook attrs or named columns.

    Attrs win when present: ``pce``, ``investment``, ``government``, ``exports``.
    Otherwise NIPA column aliases on ``table`` are used. No network.
    """
    attrs = getattr(table, "attrs", {}) or {}
    attr_keys = {
        "HOUSEHOLD": ("pce", "HOUSEHOLD"),
        "INVESTMENT": ("investment", "INVESTMENT"),
        "GOVT": ("government", "GOVT"),
        "EXPORTS": ("exports", "EXPORTS"),
    }
    from_attrs: dict[str, dict[str, float]] = {}
    for fd_key, names in attr_keys.items():
        for name in names:
            if name in attrs and attrs[name] is not None:
                from_attrs[fd_key] = {str(k): float(v) for k, v in dict(attrs[name]).items()}
                break
    if len(from_attrs) == len(FD_KEYS):
        return (
            from_attrs["HOUSEHOLD"],
            from_attrs["INVESTMENT"],
            from_attrs["GOVT"],
            from_attrs["EXPORTS"],
        )

    columns = list(getattr(table, "columns", []))
    index = list(getattr(table, "index", []))
    rows = [r for r in index if str(r) != "Gross Output"]

    def _series_for(fd_key: str) -> dict[str, float] | None:
        aliases = NIPA_FD_COLUMN_ALIASES[fd_key]
        for alias in aliases:
            hit = _lookup_key(alias, {str(c): c for c in columns})
            if hit is None:
                continue
            col = table[hit] if not hasattr(table, "loc") else table.loc[rows, hit]
            return {str(r): float(col[r]) for r in rows}
        parts: list[Any] = []
        for alias in NIPA_FD_COMPONENT_COLUMNS.get(fd_key, ()):
            hit = _lookup_key(alias, {str(c): c for c in columns})
            if hit is not None:
                parts.append(table[hit] if not hasattr(table, "loc") else table.loc[rows, hit])
        if not parts:
            return None
        out: dict[str, float] = {str(r): 0.0 for r in rows}
        for part in parts:
            for r in rows:
                out[str(r)] += float(part[r])
        return out

    extracted: dict[str, dict[str, float]] = {}
    extracted.update(from_attrs)
    for fd_key in FD_KEYS:
        if fd_key in extracted:
            continue
        series = _series_for(fd_key)
        if series is not None:
            extracted[fd_key] = series
    missing = [k for k in FD_KEYS if k not in extracted]
    if missing:
        raise ValueError(
            "final-demand columns not found for "
            + ", ".join(missing)
            + "; pass them via DataFrame.attrs or --fd-xlsx (local file only)"
        )
    return (
        extracted["HOUSEHOLD"],
        extracted["INVESTMENT"],
        extracted["GOVT"],
        extracted["EXPORTS"],
    )


def load_concordance_sidecar(path: Path) -> tuple[dict[str, str | dict[str, float]], dict[str, dict[str, float]]]:
    """Read a local YAML sidecar ``{concordance, splits}``. No network."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"concordance sidecar is not a mapping: {path}")
    conc = raw.get("concordance") or raw.get("fd_concordance") or {}
    splits = raw.get("splits") or {}
    return conc, splits


def aggregate_use_table(
    df,
    concordance: dict[str, str] | Mapping[str, str | Mapping[str, float]],
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


def _is_seed_io(path: Path) -> bool:
    return path.resolve() == SEED_IO_TABLE.resolve()


def _reject_seed_overwrite(path: Path) -> bool:
    """Return True if ``path`` is the shipped seed (must not be written)."""
    if _is_seed_io(path):
        print("refusing to overwrite seed config/io_table.json (ADR-015 / T8.04)", file=sys.stderr)
        return True
    return False


def _print_fd_result(fd: np.ndarray, weights: Mapping[str, float], coverage: float) -> None:
    print("final-demand concordance (proposed; seed io_table.json not written)")
    print(f"  coverage={coverage:.3f}  floor={COVERAGE_FLOOR}")
    for key, w in weights.items():
        print(f"  fd_weights.{key}={w:.6f}")
    print(f"  fd_matrix shape={fd.shape}  (4 × {len(CODES)} CODES)")


def _print_dry_run() -> None:
    ex = example_fd_arithmetic()
    print("T8.04 final-demand concordance — dry-run (no workbook, no network)")
    print("Method: map BEA NIPA PCE / private fixed investment / government /")
    print("exports onto the 18 CODES; fd_weights = observed C/I/G/X shares.")
    print(f"Coverage floor stays {COVERAGE_FLOOR}. Seed io_table.json is not overwritten.")
    print("FD columns: --fd-xlsx, or the same --xlsx attrs/columns (pce, investment,")
    print("government, exports). See claude/plan/reports/bea-howto.md and calibration.md.")
    print()
    print("Seed fd_weights (hand-set, not from this method):")
    for k, v in SEED_FD_WEIGHTS.items():
        print(f"  {k}={v:.2f}")
    print()
    print("Example arithmetic (synthetic dollars; not a BEA year):")
    print(f"  PCE {sum(ex['pce'].values()):.0f} + I {sum(ex['investment'].values()):.0f}")
    print(f"  + G {sum(ex['government'].values()):.0f} + X {sum(ex['exports'].values()):.0f}")
    print(f"  = {ex['grand_total']:.0f}")
    for k, v in ex["weights"].items():
        print(f"  proposed {k}={v:.6f}  (vs seed {SEED_FD_WEIGHTS[k]:.2f})")
    print(f"  coverage={ex['coverage']:.3f}  n_sectors={ex['n_sectors']}")


def _concordance_from_workbook(
    df,
    sidecar: Path | None,
) -> tuple[dict[str, str | dict[str, float]], dict[str, dict[str, float]]]:
    if sidecar is not None:
        return load_concordance_sidecar(sidecar)
    attrs = getattr(df, "attrs", {}) or {}
    conc = attrs.get("fd_concordance") or attrs.get("concordance") or {}
    splits = attrs.get("splits") or {}
    if not conc and "concordance" in getattr(df, "columns", []):
        raise SystemExit("pass concordance via DataFrame.attrs, --concordance sidecar, or defaults")
    if not conc:
        conc = DEFAULT_NIPA_CONCORDANCE
    return conc, splits


def _run_final_demand(
    df,
    sidecar: Path | None,
    coverage_floor: float,
    fd_out: Path | None,
) -> int:
    conc, splits = _concordance_from_workbook(df, sidecar)
    try:
        pce, inv, gov, exp = extract_nipa_final_demand(df)
    except ValueError as exc:
        print(f"final-demand extract failed: {exc}", file=sys.stderr)
        return 2
    fd, weights, coverage = aggregate_final_demand(pce, inv, gov, exp, conc, splits)
    if coverage < coverage_floor:
        print(f"FD coverage {coverage:.3f} below floor {coverage_floor}", file=sys.stderr)
        return 1
    _print_fd_result(fd, weights, coverage)
    if fd_out is not None:
        if _reject_seed_overwrite(fd_out):
            return 2
        np.savez(
            fd_out,
            fd_matrix=fd,
            weights=np.array([weights[k] for k in FD_KEYS]),
            coverage=np.array([coverage]),
            keys=np.array(FD_KEYS),
        )
        print(f"wrote {fd_out} (proposal only; seed not updated)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate a BEA Use table and/or NIPA final-demand columns (local file only). "
            "Does not fetch the network. Does not overwrite config/io_table.json."
        )
    )
    p.add_argument("--xlsx", help="Local Use-table workbook")
    p.add_argument(
        "--fd-xlsx",
        dest="fd_xlsx",
        default=None,
        help=(
            "Local NIPA / Use-table workbook with PCE, private fixed investment, "
            "government, and export columns (or the same maps in DataFrame.attrs). "
            "If omitted, FD is read from --xlsx attrs/columns when present."
        ),
    )
    p.add_argument("--out", default=None, help="Destination *.npz for the Use-table Z (not io_table.json)")
    p.add_argument("--fd-out", dest="fd_out", default=None, help="Destination *.npz for proposed FD (not io_table.json)")
    p.add_argument("--concordance", default=None, help="Local YAML sidecar {concordance, splits}")
    p.add_argument("--validate", action="store_true", help="Run Layer-1 checks through load_io on the seed")
    p.add_argument("--coverage-floor", type=float, default=COVERAGE_FLOOR)
    p.add_argument("--config-dir", default=str(ROOT / "config"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print concordance method + synthetic arithmetic; exit 0; no files, no network",
    )
    args = p.parse_args(argv)

    if args.dry_run:
        _print_dry_run()
        return 0

    if not args.xlsx and not args.fd_xlsx:
        print(
            "No workbook given. Download the BEA Use table from https://apps.bea.gov/ "
            "and re-run with --xlsx PATH and/or --fd-xlsx PATH. FD columns may also "
            "live on the same workbook (named PCE/I/G/X columns or DataFrame.attrs). "
            "No request is made. See claude/plan/reports/bea-howto.md and "
            "claude/plan/reports/calibration.md.",
            file=sys.stderr,
        )
        return 2

    sidecar = Path(args.concordance) if args.concordance else None
    if sidecar is not None and not sidecar.exists():
        print(f"concordance sidecar not found: {sidecar} (no network fetch is attempted)", file=sys.stderr)
        return 2

    rc = 0
    if args.xlsx:
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
        if sidecar is not None:
            concordance, splits = load_concordance_sidecar(sidecar)
        if not concordance and "concordance" in df.columns:
            raise SystemExit("pass concordance via DataFrame.attrs or a sidecar")

        has_fd = any(k in df.attrs for k in ("pce", "investment", "government", "exports", "HOUSEHOLD"))
        has_fd = has_fd or any(
            _lookup_key(alias, {str(c): c for c in getattr(df, "columns", [])}) is not None
            for aliases in NIPA_FD_COLUMN_ALIASES.values()
            for alias in aliases
        )
        # Empty concordance on a Use table still runs so coverage-below-floor exits 1
        # (T0.09). Skip only when this workbook is FD-only (attrs/columns, no industry map).
        if concordance or not has_fd:
            z, go, coverage = aggregate_use_table(df, concordance, splits)
            if coverage < args.coverage_floor:
                print(f"coverage {coverage:.3f} below floor {args.coverage_floor}", file=sys.stderr)
                return 1
            if args.out:
                out_path = Path(args.out)
                if _reject_seed_overwrite(out_path):
                    return 2
                np.savez(out_path, Z=z, go=go, coverage=np.array([coverage]))
                print(f"wrote {out_path}")

        if args.validate:
            from marketsim.core.config import load_config

            cfg = load_config(args.config_dir)
            io = load_io(Path(args.config_dir) / "io_table.json")
            assert_structure(io, cfg)
            print("validated seed table via load_io (seed io_table.json not replaced)")

        if not args.fd_xlsx:
            attrs = getattr(df, "attrs", {}) or {}
            has_fd = any(k in attrs for k in ("pce", "investment", "government", "exports", "HOUSEHOLD"))
            has_fd = has_fd or any(
                _lookup_key(alias, {str(c): c for c in getattr(df, "columns", [])}) is not None
                for aliases in NIPA_FD_COLUMN_ALIASES.values()
                for alias in aliases
            )
            if has_fd:
                rc = _run_final_demand(
                    df,
                    sidecar,
                    args.coverage_floor,
                    Path(args.fd_out) if args.fd_out else None,
                )

    if args.fd_xlsx:
        fd_path = Path(args.fd_xlsx)
        if not fd_path.exists():
            print(f"FD workbook not found: {fd_path} (no network fetch is attempted)", file=sys.stderr)
            return 2
        try:
            fd_df = _load_xlsx(fd_path)
        except Exception as exc:
            print(f"failed to read {fd_path}: {exc}", file=sys.stderr)
            return 2
        rc = _run_final_demand(
            fd_df,
            sidecar,
            args.coverage_floor,
            Path(args.fd_out) if args.fd_out else None,
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
