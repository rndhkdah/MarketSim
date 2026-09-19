"""Monetary and macroprudential instruments (D13 / §2.13). QE/QT is T6.28."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from marketsim.ledger.journal import Entry, Ledger, Tx


@dataclass
class MonetaryLevers:
    """Resolved CENBANK levers. ``None`` means the autopilot rule."""

    rate: float | None = None
    pi_star: float | None = None
    phi_pi: float | None = None
    phi_y: float | None = None
    smoothing: float | None = None
    capital_requirement: float | None = None
    ltv_cap: float | None = None
    lolr: float = 0.0
    guidance_path: list[float] | None = None


def levers_from_merged(merged: dict[str, Any]) -> MonetaryLevers:
    path = merged.get("guidance_path")
    return MonetaryLevers(
        rate=merged.get("rate"),
        pi_star=merged.get("pi_star"),
        phi_pi=merged.get("phi_pi"),
        phi_y=merged.get("phi_y"),
        smoothing=merged.get("smoothing"),
        capital_requirement=merged.get("capital_requirement"),
        ltv_cap=merged.get("ltv_cap"),
        lolr=float(merged.get("lolr") or 0.0),
        guidance_path=list(path) if path is not None else None,
    )


def consume_oneoffs(authority: Any) -> None:
    for name in ("lolr",):
        authority.effective.pop(name, None)
        authority.source_of.pop(name, None)


class QEOperations(Protocol):
    """Asset purchases / sales. Implemented in T6.28 once bonds are market-priced."""

    def purchase(self, amount: float, bucket: str) -> None: ...

    def sale(self, amount: float, bucket: str) -> None: ...


class DeferredQE:
    """T2.29 declaration; body is T6.28."""

    def purchase(self, amount: float, bucket: str) -> None:
        del amount, bucket
        raise NotImplementedError("QE/QT is T6.28")

    def sale(self, amount: float, bucket: str) -> None:
        del amount, bucket
        raise NotImplementedError("QE/QT is T6.28")


def post_lolr(ledger: Ledger, amount: float, *, tick: int) -> None:
    """CB lends ``amount`` of reserves to BANKSYS against a LOAN. Units: cr."""
    if abs(amount) < 1e-14:
        return
    amt = float(amount)
    ledger.post(
        Tx(
            tick,
            "loan_new",
            (
                Entry("BANKSYS", "RES", amt),
                Entry("CB", "RES", -amt),
                Entry("CB", "LOAN", amt),
                Entry("BANKSYS", "LOAN", -amt),
            ),
            memo="lolr",
        )
    )


def apply_requirement_to_midpoint(midpoint: float, baseline: float, requirement: float) -> float:
    """Shift the logistic midpoint by ``requirement − baseline``."""
    return float(midpoint + (requirement - baseline))


@dataclass
class PolicyToolkit:
    """Held on the economy: guidance path and the deferred QE desk."""

    guidance_path: list[float] = field(default_factory=list)
    qe: DeferredQE = field(default_factory=DeferredQE)
    last_lolr: float = 0.0
