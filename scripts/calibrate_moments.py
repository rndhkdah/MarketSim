#!/usr/bin/env python3
"""Propose dynamics / household / capex gains from documented FRED/BLS targets.

T8.04 HUMAN GATE. Reads shipped yaml for the *current* column only. Does not
write yaml. Numbers come from ADR-015, T2.24 / T2.35 reports, and the card
targets — no new elasticities (ADR-015 §4).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Card / ADR-015 targets (FRED/BLS stylised facts as stated on T8.04).
TARGET_SD_I_OVER_GDP = (3.0, 4.0)
TARGET_SD_C_OVER_GDP = 1.0  # strict upper bound: sd(C)/sd(GDP) < 1
TARGET_ENVELOPE = 2.0  # last/first 20y |gap| envelope; T2.24 spec (GDP persistence proxy)

# T2.24 100-year stochastic, π*=2 %, seeds 1–3 (`reports/phase2-validation.md`).
T224_SD_I_OVER_GDP = (0.88, 0.96, 1.03)
T224_SD_C_OVER_GDP = (0.78, 0.99, 0.91)
T224_ENVELOPE = (1.43, 1.00, 0.95)
# Prototype (master plan §13) — not the seed; cited for the I/GDP overshoot contrast.
PROTOTYPE_SD_I_OVER_GDP = 7.0
PROTOTYPE_SD_C_OVER_GDP = 1.45

# T2.24 authorized proposals (not applied). Do not invent a linear φ that would
# close the I/GDP gap in one step — that would be a new elasticity.
T224_PHI_CURRENT = 1.2
T224_PHI_PROPOSED = 1.6
T224_ANCHOR_CURRENT = 0.75
T224_ANCHOR_PROPOSED = 0.6


@dataclass(frozen=True)
class MomentRow:
    key: str
    unit: str
    current: float | str
    proposed: float | str
    target: str
    source: str
    applied: bool = False


def _mean(xs: tuple[float, ...]) -> float:
    return float(sum(xs) / len(xs))


def current_yaml_gains(config_dir: Path | None = None) -> dict[str, float]:
    """Read shipped dynamics / capex keys (no writes)."""
    from marketsim.core.config import load_config

    cfg = load_config(str(config_dir or (ROOT / "config")))
    dyn = cfg.dynamics
    cap = cfg.edges.capex
    assert dyn is not None
    return {
        "edges.capex.coefficients.phi_accelerator": float(cap.coefficients.phi_accelerator),
        "edges.capex.coefficients.psi_utilisation": float(cap.coefficients.psi_utilisation),
        "edges.capex.coefficients.chi_cost_of_capital": float(cap.coefficients.chi_cost_of_capital),
        "edges.capex.coefficients.q_tobin": float(cap.coefficients.q_tobin),
        "edges.capex.unit_scale": float(cap.unit_scale),
        "dynamics.expectations.anchor_growth": float(dyn.expectations.anchor_growth),
        "dynamics.expectations.tau_growth_m": float(dyn.expectations.tau_growth_m),
        "dynamics.households.alpha1": float(dyn.households.alpha1),
        "dynamics.households.tau_income_m": float(dyn.households.tau_income_m),
        "dynamics.residential.rate_semi": float(dyn.residential.rate_semi),
        "dynamics.residential.share_of_investment": float(dyn.residential.share_of_investment),
        "dynamics.production.cover_scale": float(dyn.production.cover_scale),
    }


def propose_moment_gains(config_dir: Path | None = None) -> list[MomentRow]:
    """Compute the proposed key table from documented targets + T2.24 proposals.

    I/GDP is far below 3–4 on the seed (mean ≈ 0.96). A linear scale
    ``φ ← 1.2 × (3.5 / 0.96) ≈ 4.4`` is recorded as *rejected* — it invents an
    elasticity. The authorized step is T2.24's ``φ: 1.2 → 1.6`` *or*
    ``anchor_growth: 0.75 → 0.6`` (not stacked). C/GDP already meets ``< 1``.
    GDP persistence is the T2.24 envelope (already ``< 2``).
    """
    gains = current_yaml_gains(config_dir)
    i_mean = _mean(T224_SD_I_OVER_GDP)
    c_mean = _mean(T224_SD_C_OVER_GDP)
    env_mean = _mean(T224_ENVELOPE)
    target_i_mid = 0.5 * (TARGET_SD_I_OVER_GDP[0] + TARGET_SD_I_OVER_GDP[1])
    linear_phi = T224_PHI_CURRENT * (target_i_mid / i_mean)

    phi_now = gains["edges.capex.coefficients.phi_accelerator"]
    anchor_now = gains["dynamics.expectations.anchor_growth"]
    alpha1_now = gains["dynamics.households.alpha1"]

    return [
        MomentRow(
            key="moment.sd_I_over_sd_GDP",
            unit="1 (ratio of s.d.)",
            current=round(i_mean, 3),
            proposed="raise toward 3–4 via φ or anchor (T2.24); not a new elasticity",
            target="3–4 (T8.04 / FRED-style I volatility)",
            source="T2.24 seeds 1–3: 0.88 / 0.96 / 1.03; proto ≈ 7",
        ),
        MomentRow(
            key="moment.sd_C_over_sd_GDP",
            unit="1 (ratio of s.d.)",
            current=round(c_mean, 3),
            proposed="keep (already < 1)",
            target="< 1 (T8.04 / FRED-style C volatility)",
            source="T2.24 seeds 1–3: 0.78 / 0.99 / 0.91; proto ≈ 1.45",
        ),
        MomentRow(
            key="moment.gdp_persistence_envelope",
            unit="last/first 20y |gap| ratio",
            current=round(env_mean, 3),
            proposed="keep; re-check if anchor_growth moves",
            target=f"< {TARGET_ENVELOPE:.0f} (T2.24 spec; documented persistence proxy)",
            source="T2.24 envelopes 1.43 / 1.00 / 0.95",
        ),
        MomentRow(
            key="edges.capex.coefficients.phi_accelerator",
            unit="pct-pts of K per year (ADR-P2)",
            current=phi_now,
            proposed=T224_PHI_PROPOSED,
            target="primary I/GDP step (T2.24); do not use "
            f"linear {linear_phi:.2f} (= {T224_PHI_CURRENT}×{target_i_mid:.1f}/{i_mean:.2f})",
            source="T2.24 tuning proposals; ADR-015 §3",
        ),
        MomentRow(
            key="dynamics.expectations.anchor_growth",
            unit="1 (weight on trend)",
            current=anchor_now,
            proposed=T224_ANCHOR_PROPOSED,
            target="alternative I/GDP step — do not stack with φ without a 100y re-run",
            source="T2.24: 0.75 → 0.6 *or* φ 1.2 → 1.6",
        ),
        MomentRow(
            key="dynamics.households.alpha1",
            unit="1 (MPC out of expected YD)",
            current=alpha1_now,
            proposed=alpha1_now,
            target="no change: seed sd(C)/sd(GDP) already < 1",
            source="dynamics.yaml; T2.24 C/GDP",
        ),
        MomentRow(
            key="dynamics.households.tau_income_m",
            unit="months",
            current=gains["dynamics.households.tau_income_m"],
            proposed=gains["dynamics.households.tau_income_m"],
            target="keep 12-month income smoother (prototype damper)",
            source="master plan §13; dynamics.yaml",
        ),
        MomentRow(
            key="dynamics.expectations.tau_growth_m",
            unit="months",
            current=gains["dynamics.expectations.tau_growth_m"],
            proposed=gains["dynamics.expectations.tau_growth_m"],
            target="keep; envelope already inside spec",
            source="dynamics.yaml; T2.24 envelope",
        ),
        MomentRow(
            key="dynamics.residential.rate_semi",
            unit="percent per 100bp",
            current=gains["dynamics.residential.rate_semi"],
            proposed=gains["dynamics.residential.rate_semi"],
            target="no change this pass (CONSTRUCT ranking is T2.24 xfail)",
            source="T2.24; ADR-015 §4 — do not invent a new elasticity",
        ),
        MomentRow(
            key="edges.capex.unit_scale",
            unit="1",
            current=gains["edges.capex.unit_scale"],
            proposed=gains["edges.capex.unit_scale"],
            target="keep 0.01 (ADR-P2 / ADR-002); 0.0125 was a T2.24 sweep idea only",
            source="ADR-002; T2.24 ranking reopen stands",
        ),
        MomentRow(
            key="rejected.linear_phi_to_close_I_gap",
            unit="pct-pts of K per year",
            current=phi_now,
            proposed=round(linear_phi, 3),
            target="rejected — invents an elasticity (ADR-015 §4)",
            source=f"3.5 / {i_mean:.3f} × {T224_PHI_CURRENT}",
        ),
    ]


def format_table(rows: list[MomentRow]) -> str:
    lines = [
        "T8.04 moment proposals — NOT APPLIED (human must tick T8.04)",
        f"{'key':<48} {'current':<12} {'proposed':<42} applied",
        "-" * 110,
    ]
    for row in rows:
        lines.append(
            f"{row.key:<48} {str(row.current):<12} {str(row.proposed):<42} {row.applied}"
        )
    lines.append("")
    lines.append(
        f"Targets: sd(I)/sd(GDP) {TARGET_SD_I_OVER_GDP[0]:.0f}–{TARGET_SD_I_OVER_GDP[1]:.0f}; "
        f"sd(C)/sd(GDP) < {TARGET_SD_C_OVER_GDP:.0f}; "
        f"GDP persistence envelope < {TARGET_ENVELOPE:.0f}."
    )
    lines.append(
        f"Seed known (T2.24): I/GDP {_mean(T224_SD_I_OVER_GDP):.2f} "
        f"(proto {PROTOTYPE_SD_I_OVER_GDP:.0f}); "
        f"C/GDP {_mean(T224_SD_C_OVER_GDP):.2f} (proto {PROTOTYPE_SD_C_OVER_GDP:.2f})."
    )
    lines.append("No yaml written. See claude/plan/reports/calibration.md.")
    return "\n".join(lines)


def rows_as_json(rows: list[MomentRow]) -> dict[str, Any]:
    return {
        "applied": False,
        "human_gate": "T8.04",
        "targets": {
            "sd_I_over_sd_GDP": list(TARGET_SD_I_OVER_GDP),
            "sd_C_over_sd_GDP_max": TARGET_SD_C_OVER_GDP,
            "gdp_persistence_envelope_max": TARGET_ENVELOPE,
        },
        "t2_24_current": {
            "sd_I_over_sd_GDP": list(T224_SD_I_OVER_GDP),
            "sd_C_over_sd_GDP": list(T224_SD_C_OVER_GDP),
            "envelope": list(T224_ENVELOPE),
        },
        "rows": [asdict(r) for r in rows],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Propose dynamics/household/capex gains from documented FRED/BLS targets. "
            "Prints a table. Does not write yaml. No network."
        )
    )
    p.add_argument("--config-dir", default=str(ROOT / "config"))
    p.add_argument("--out", default=None, help="Optional JSON proposal path (not a shipped yaml)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposal table and exit 0 (default behaviour; no writes)",
    )
    args = p.parse_args(argv)
    rows = propose_moment_gains(Path(args.config_dir))
    print(format_table(rows))
    if args.out and not args.dry_run:
        out = Path(args.out)
        if out.suffix.lower() in {".yaml", ".yml"}:
            print("refusing to write yaml (ADR-015); use .json or omit --out", file=sys.stderr)
            return 2
        out.write_text(json.dumps(rows_as_json(rows), indent=2) + "\n")
        print(f"wrote proposal JSON {out} (not applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
