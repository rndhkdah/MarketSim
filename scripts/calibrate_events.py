#!/usr/bin/env python3
"""Map published historic figures onto ShockBus z in the IRF-safe band.

T8.04 HUMAN GATE. Historic *real* moves are targets, not ShockBus values
(ADR-015). 1973 crude ×4 → z_cost ≈ 0.30–0.40, **not** ln 4. Does not edit
``config/events/*.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ADR-015 / QUESTIONS T4.11 / event-calibration.md
IRF_SAFE_COST_PUSH: tuple[float, float] = (0.30, 0.40)
T224_COST_PUSH_Z = 0.30
T224_CPI_12M = 0.72  # +72 % CPI @12m at z=0.30 (T2.24)
# EIA / BP figures quoted in event-calibration.md (Arabian Light $2.90 → $11.65).
OIL_1973_P0 = 2.90
OIL_1973_P1 = 11.65
ENERGY_2022_CPI_YY = 0.091  # BLS CPI-U 9.1 % y/y Jun 2022


@dataclass(frozen=True)
class EventZRow:
    event_id: str
    primitive: str
    published: str
    source: str
    seed_z: str
    naive_z: str
    proposed_z: str
    note: str
    applied: bool = False


def oil_1973_multiple() -> float:
    """Published Arabian Light multiple (event-calibration.md). Units: 1."""
    return OIL_1973_P1 / OIL_1973_P0


def naive_log_price_z(multiple: float) -> float:
    """Anti-pattern: ``ln(price multiple)`` as ``z_cost``. Do not inject this."""
    if multiple <= 0.0:
        raise ValueError("price multiple must be positive")
    return float(math.log(multiple))


def map_historic_cost_push(
    multiple: float | None = None,
    *,
    seed_z: float | None = None,
    band: tuple[float, float] = IRF_SAFE_COST_PUSH,
) -> tuple[float, float, float]:
    """Map a published price multiple onto the IRF-safe ``z_cost`` band.

    Returns ``(naive_ln, proposed_lo, proposed_hi)``. The naive log is
    recorded and is **not** the proposed ShockBus value (ADR-015).
    Seeds already inside the band (or below the floor, e.g. AGRIFOOD 0.2)
    keep their order of magnitude; oversized cost-push seeds clip to the band.
    """
    lo, hi = band
    if lo <= 0.0 or hi < lo:
        raise ValueError("IRF-safe band must be positive and ordered")
    naive = naive_log_price_z(multiple) if multiple is not None else (
        float("nan") if seed_z is None else float(seed_z)
    )
    if seed_z is not None and 0.0 < seed_z < lo:
        return naive, seed_z, seed_z
    return naive, lo, hi


def linear_cpi_implied_z(
    published_cpi: float,
    *,
    irf_z: float = T224_COST_PUSH_Z,
    irf_cpi: float = T224_CPI_12M,
) -> float:
    """``z`` that would match a published CPI *if* the T2.24 IRF were linear.

    Recorded only. Not proposed: ADR-015 maps 1973 ×4 onto 0.30–0.40, not this
    shrink-to-match-CPI figure. Units: ShockBus z (log-points on ``lp*``).
    """
    if irf_cpi <= 0.0:
        raise ValueError("IRF CPI must be positive")
    return float(irf_z * published_cpi / irf_cpi)


def event_z_table() -> list[EventZRow]:
    """Historic templates + generics from ``reports/event-calibration.md``."""
    oil_mult = oil_1973_multiple()
    oil_naive, oil_lo, oil_hi = map_historic_cost_push(oil_mult, seed_z=1.25)
    _e_naive, e_lo, e_hi = map_historic_cost_push(seed_z=0.70)
    _a_naive, a_lo, a_hi = map_historic_cost_push(seed_z=0.20)
    cpi_linear = linear_cpi_implied_z(ENERGY_2022_CPI_YY)

    return [
        EventZRow(
            event_id="oil_embargo_1973",
            primitive="cost_push ENERGY",
            published=f"Arabian Light ${OIL_1973_P0:.2f} → ${OIL_1973_P1:.2f} (≈×{oil_mult:.2f})",
            source="EIA / BP Statistical Review (event-calibration.md)",
            seed_z="lognormal median 1.25 (min 1.1 max 1.4)",
            naive_z=f"ln({oil_mult:.2f})={oil_naive:.3f}  [DO NOT USE]",
            proposed_z=f"{oil_lo:.2f}–{oil_hi:.2f}",
            note="ADR-015: IRF-safe band, not ln 4. T2.24 z=0.30 still ≈+72% CPI @12m.",
        ),
        EventZRow(
            event_id="energy_inflation_2022",
            primitive="cost_push ENERGY",
            published=f"CPI-U {ENERGY_2022_CPI_YY:.1%} y/y Jun 2022; funds +525 bp / 16m",
            source="BLS CPI, FOMC (event-calibration.md)",
            seed_z="uniform 0.50–0.90",
            naive_z=f"linear CPI match z≈{cpi_linear:.3f}  [not proposed]",
            proposed_z=f"{e_lo:.2f}–{e_hi:.2f}",
            note="Clip seed into IRF-safe band. Linear 9.1/72 × 0.30 is recorded, not applied.",
        ),
        EventZRow(
            event_id="energy_inflation_2022",
            primitive="cost_push AGRIFOOD",
            published="companion food-price leg (seed +0.2)",
            source="event-calibration.md ENERGY/AGRIFOOD composition",
            seed_z="uniform 0.15–0.25",
            naive_z="—",
            proposed_z=f"{a_lo:.2f} (keep; already below IRF-safe floor)",
            note="Order of magnitude of the composition is kept (ADR-015).",
        ),
        EventZRow(
            event_id="cost_push_inflation",
            primitive="cost_push ENERGY+AGRIFOOD",
            published="generic follow-up; no separate historic price multiple",
            source="event-calibration.md generics",
            seed_z="uniform 0.10–0.25",
            naive_z="—",
            proposed_z="0.10–0.25 (keep; already ≤ IRF-safe floor)",
            note="Already SFC-finite on the T2.24 IRF scale.",
        ),
        EventZRow(
            event_id="asian_crisis_1997",
            primitive="world_demand + risk_appetite",
            published="emerging-Asia import collapse 1998; no FX in v1",
            source="IMF WEO 1998",
            seed_z="world_demand −8 %; risk +200–400 bp",
            naive_z="—",
            proposed_z="keep seed (quantity / risk, not z_cost)",
            note="Not a cost-push overflow. T4.11 24m SFC issue is oil/energy.",
        ),
        EventZRow(
            event_id="dotcom_2000",
            primitive="risk_appetite + capex_preference",
            published="NASDAQ −78 % Mar 2000–Oct 2002",
            source="FRED NASDAQCOM",
            seed_z="risk +; SOFTWARE/TELECOM/SEMIS capex −25…−35 %",
            naive_z="do not put −ln(1−0.78) on z_cost",
            proposed_z="keep seed (preference / risk)",
            note="Equity-price path is a target, not a ShockBus z.",
        ),
        EventZRow(
            event_id="gfc_2008",
            primitive="risk + collateral + bank_equity",
            published="CSUSHPINSA ≈ −27 %; S&P 500 −57 %",
            source="FRED CSUSHPINSA, SP500",
            seed_z="collateral −25…−30 %; bank_equity −30…−50 %; risk +300–500 bp",
            naive_z="—",
            proposed_z="keep seed (already a quantity map to −27 % housing)",
            note="Quantity rationing, not a firm-specific spread (D14).",
        ),
        EventZRow(
            event_id="tohoku_2011",
            primitive="catastrophe + link_capacity",
            published="Japan auto output ≈ −50 % Mar–Apr 2011",
            source="METI / JAMA",
            seed_z="catastrophe 1–3 % of K; link_capacity −50 % / 60 d",
            naive_z="do not set z_cost = 0.50",
            proposed_z="keep seed (K / link quantity)",
            note="Output drop is a target for T8.05 directions, not z_cost.",
        ),
        EventZRow(
            event_id="thailand_floods_2011",
            primitive="supply SEMIS",
            published="HDD shipments ≈ −29 % 2011Q4 / 2011Q3",
            source="iSuppli / TrendFocus",
            seed_z="supply −25…−30 %, 2q",
            naive_z="—",
            proposed_z="keep seed (already ≈ published −29 %)",
            note="Supply primitive, SFC-finite.",
        ),
        EventZRow(
            event_id="covid_2020",
            primitive="labour_supply + want_shift + world_demand",
            published="real GDP −31.2 % SAAR Q2 (≈ −9 % q/q); U-3 14.7 % Apr",
            source="BEA NIPA, BLS CPS",
            seed_z="labour −10…−15 %; wants; world_demand −10 %",
            naive_z="do not set z_demand = −0.312",
            proposed_z="keep labour/want seeds (SFC-finite; HEALTH vs DISCRET is T4.11)",
            note="T4.10 storm may re-include after this map is accepted.",
        ),
        EventZRow(
            event_id="chip_shortage_2020",
            primitive="supply SEMIS",
            published="chip lead times peaked ≈ 26 weeks",
            source="Susquehanna / industry surveys",
            seed_z="supply lognormal median 8 %",
            naive_z="—",
            proposed_z="keep seed",
            note="Lead time is not a z_cost.",
        ),
        EventZRow(
            event_id="suez_2021",
            primitive="import_price / link_cost / world_demand pulse",
            published="~12 % of world trade transits Suez",
            source="UNCTAD RMT",
            seed_z="6-day pulse; world_demand −2 %",
            naive_z="do not set z = 0.12",
            proposed_z="keep seed (pulse, not a 12 % output shock)",
            note="Share of trade is exposure, not ShockBus z.",
        ),
        EventZRow(
            event_id="ai_boom_2023",
            primitive="capex_preference + risk",
            published="US data-centre and semiconductor equipment capex surge",
            source="BEA / 10-K capex",
            seed_z="capex_preference +20…+40 % SEMIS/SOFTWARE/UTILITIES",
            naive_z="—",
            proposed_z="keep seed",
            note="Preference shifter, not z_cost.",
        ),
        EventZRow(
            event_id="fiscal_stimulus",
            primitive="fiscal",
            published="IMF discretionary impulse band for AE packages",
            source="event-calibration.md generics",
            seed_z="+2…+5 %, 3q",
            naive_z="—",
            proposed_z="keep seed",
            note="Already a % of GDP on the fiscal primitive.",
        ),
        EventZRow(
            event_id="monetary_tightening",
            primitive="monetary",
            published="FOMC hiking-cycle *steps* (not the 2022 +525 bp cumulative)",
            source="event-calibration.md generics",
            seed_z="+100–300 bp, 3q",
            naive_z="do not put +5.25 on z_monetary",
            proposed_z="keep seed",
            note="2022 cumulative is the energy follow-up path, not this generic.",
        ),
        EventZRow(
            event_id="oil_monetary_tightening / energy_monetary_tightening",
            primitive="monetary",
            published="1974 / 2022 tightening *increments* around the energy spike",
            source="event-calibration.md",
            seed_z="+200–400 bp",
            naive_z="—",
            proposed_z="keep seed",
            note="Follow-up; D13 pressure-news when not autopilot.",
        ),
        EventZRow(
            event_id="world_demand_drop",
            primitive="world_demand",
            published="IMF world-trade growth gaps in 1998 / 2009 / 2020",
            source="event-calibration.md generics",
            seed_z="−5…−10 %",
            naive_z="—",
            proposed_z="keep seed",
            note="ROW quantity, not z_cost.",
        ),
    ]


def format_table(rows: list[EventZRow]) -> str:
    lines = [
        "T8.04 event z map — NOT APPLIED (human must tick T8.04)",
        f"IRF-safe cost-push band: {IRF_SAFE_COST_PUSH[0]:.2f}–{IRF_SAFE_COST_PUSH[1]:.2f} "
        f"(T2.24 z={T224_COST_PUSH_Z:.2f} → CPI @12m ≈ +{T224_CPI_12M:.0%}).",
        f"1973 crude ×{oil_1973_multiple():.2f} → z_cost in that band, NOT "
        f"ln 4 = {naive_log_price_z(4.0):.3f}.",
        "",
        f"{'event':<36} {'primitive':<28} {'seed z':<36} {'proposed z':<28}",
        "-" * 132,
    ]
    for row in rows:
        lines.append(f"{row.event_id:<36} {row.primitive:<28} {row.seed_z:<36} {row.proposed_z:<28}")
        lines.append(f"  published: {row.published}")
        lines.append(f"  naive:     {row.naive_z}")
        lines.append(f"  note:      {row.note}")
    lines.append("")
    lines.append("No event yaml written. See claude/plan/reports/calibration.md.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Map published historic figures onto ShockBus z in the IRF-safe band. "
            "Prints a table. Does not edit event yaml. No network."
        )
    )
    p.add_argument("--out", default=None, help="Optional JSON proposal path (not a shipped yaml)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the table and exit 0 (default behaviour; no writes)",
    )
    args = p.parse_args(argv)
    rows = event_z_table()
    print(format_table(rows))
    if args.out and not args.dry_run:
        out = Path(args.out)
        if out.suffix.lower() in {".yaml", ".yml"}:
            print("refusing to write yaml (ADR-015); use .json or omit --out", file=sys.stderr)
            return 2
        payload = {
            "applied": False,
            "human_gate": "T8.04",
            "irf_safe_cost_push": list(IRF_SAFE_COST_PUSH),
            "oil_1973_multiple": oil_1973_multiple(),
            "naive_ln4": naive_log_price_z(4.0),
            "t2_24_cpi_12m_at_z030": T224_CPI_12M,
            "rows": [asdict(r) for r in rows],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote proposal JSON {out} (not applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
