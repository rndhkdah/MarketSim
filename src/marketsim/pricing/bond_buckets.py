"""Fungible decaying-coupon bond buckets (T6.24 / §6.11, D12).

Positions are units of remaining face. Coupons and redemptions are ledger
postings (``interest_bonds``, ``bond_redeem``). Price changes are revaluations
and are never income.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.ledger.journal import Entry, Ledger, Tx

MONTHS_PER_YEAR = 12.0  # §6.11 monthly arithmetic
DURATION_REF_YIELD = 0.042  # annual decimal; §6.11 duration table
BUCKET_ORDER: tuple[str, ...] = ("GB_BILL", "GB_NOTE", "GB_BOND", "CORP_POOL")
GOVT_BUCKETS: tuple[str, ...] = ("GB_BILL", "GB_NOTE", "GB_BOND")
# Matches `ledger.yaml` government_mix and opening.GOVT_MIX (par bitwise).
DEFAULT_ISSUANCE_MIX: dict[str, float] = {"GB_BILL": 0.20, "GB_NOTE": 0.40, "GB_BOND": 0.40}

PricingMode = Literal["par", "market"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BucketSpec(FrozenModel):
    issuer: str
    decay: float  # 1/year (δ)

    @field_validator("decay")
    @classmethod
    def _decay(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("bucket decay must be > 0")
        return v

    @field_validator("issuer")
    @classmethod
    def _iss(cls, v: str) -> str:
        if not v:
            raise ValueError("bucket issuer must be non-empty")
        return v


class BondsFile(FrozenModel):
    """Root of `config/bonds.yaml`."""

    pricing: PricingMode = "par"
    duration_ref_yield: float = DURATION_REF_YIELD  # annual decimal
    buckets: dict[str, BucketSpec] = Field(default_factory=dict)
    issuance_mix: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_ISSUANCE_MIX))

    @field_validator("duration_ref_yield")
    @classmethod
    def _y(cls, v: float) -> float:
        if v < 0:
            raise ValueError("duration_ref_yield must be >= 0")
        return v

    @model_validator(mode="after")
    def _buckets_and_mix(self) -> BondsFile:
        missing = set(BUCKET_ORDER) - set(self.buckets)
        extra = set(self.buckets) - set(BUCKET_ORDER)
        if missing or extra:
            raise ValueError(f"buckets must be exactly {BUCKET_ORDER}")
        mix_keys = set(self.issuance_mix)
        if mix_keys != set(GOVT_BUCKETS):
            raise ValueError(f"issuance_mix must be exactly {GOVT_BUCKETS}")
        total = sum(self.issuance_mix.values())
        if abs(total - 1.0) > 1e-12:
            raise ValueError("issuance_mix must sum to 1")
        for share in self.issuance_mix.values():
            if share < 0:
                raise ValueError("issuance_mix shares must be >= 0")
        return self

    def mix_tuple(self) -> tuple[tuple[str, float], ...]:
        """Issuance / opening mix in `GOVT_MIX` order."""
        return tuple((name, float(self.issuance_mix[name])) for name in GOVT_BUCKETS)


def monthly_rate(annual: float) -> float:
    """Annual decimal → per-month decimal (``x / 12``)."""
    return float(annual) / MONTHS_PER_YEAR


def bucket_price(y_annual: float, kappa_annual: float, delta_annual: float) -> float:
    """Price per unit of remaining face: ``P = (κ_m + δ_m) / (y_m + δ_m)``."""
    y_m = monthly_rate(y_annual)
    k_m = monthly_rate(kappa_annual)
    d_m = monthly_rate(delta_annual)
    return float((k_m + d_m) / (y_m + d_m))


def bucket_duration_months(y_annual: float, delta_annual: float) -> float:
    """Macaulay-style bucket duration in months: ``(1 + y_m) / (y_m + δ_m)``."""
    y_m = monthly_rate(y_annual)
    d_m = monthly_rate(delta_annual)
    return float((1.0 + y_m) / (y_m + d_m))


def bucket_duration_years(y_annual: float, delta_annual: float) -> float:
    """Duration in years (duration_months / 12)."""
    return bucket_duration_months(y_annual, delta_annual) / MONTHS_PER_YEAR


def monthly_holding_return(
    p_prev: float,
    p_t: float,
    kappa_annual: float,
    delta_annual: float,
) -> float:
    """``(κ_m + δ_m + (1 − δ_m)·P_t) / P_{t−1} − 1``. Monthly decimal."""
    k_m = monthly_rate(kappa_annual)
    d_m = monthly_rate(delta_annual)
    return float((k_m + d_m + (1.0 - d_m) * p_t) / p_prev - 1.0)


def month_face_flows(face: float, kappa_annual: float, delta_annual: float) -> tuple[float, float, float]:
    """Coupon, redemption and remaining face (cr) for one month on ``face`` units."""
    coupon = float(face) * monthly_rate(kappa_annual)
    redeem = float(face) * monthly_rate(delta_annual)
    return coupon, redeem, float(face) - redeem


def revaluation(face: float, price_prev: float, price_new: float) -> float:
    """Mark-to-market change ``face · ΔP`` (cr). Not income — do not post as a flow tag."""
    return float(face) * (float(price_new) - float(price_prev))


def post_coupon_and_redeem(
    ledger: Ledger,
    *,
    tick: int,
    holder: str,
    issuer: str,
    instrument: str,
    face: float,
    kappa_annual: float,
    delta_annual: float,
    cash_instrument: str = "DEP",
) -> tuple[float, float, float]:
    """Post ``interest_bonds`` and ``bond_redeem``. Returns (coupon, redeem, face_next)."""
    coupon, redeem, face_next = month_face_flows(face, kappa_annual, delta_annual)
    if abs(coupon) > 1e-15:
        ledger.post(
            Tx(
                tick,
                "interest_bonds",
                (
                    Entry(issuer, cash_instrument, -coupon),
                    Entry(holder, cash_instrument, coupon),
                ),
                memo=f"{instrument} coupon",
            )
        )
    if abs(redeem) > 1e-15:
        ledger.post(
            Tx(
                tick,
                "bond_redeem",
                (
                    Entry(issuer, cash_instrument, -redeem),
                    Entry(holder, cash_instrument, redeem),
                    Entry(holder, instrument, -redeem),
                    Entry(issuer, instrument, redeem),
                ),
                memo=f"{instrument} redeem",
            )
        )
    return coupon, redeem, face_next


@dataclass
class BondBooks:
    """Per-bucket face marks. ``pricing='par'`` values face at 1; ``market`` uses ``prices``."""

    pricing: PricingMode
    kappa: dict[str, float]  # annual decimal, fixed at opening SS yield
    decay: dict[str, float]  # 1/year
    prices: dict[str, float] = field(default_factory=dict)
    issuer: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_bonds(cls, bonds: BondsFile, ss_yield: float) -> BondBooks:
        """``ss_yield`` (annual decimal) is κ so P=1 at baseline."""
        prices = {name: 1.0 for name in bonds.buckets}
        decay = {name: spec.decay for name, spec in bonds.buckets.items()}
        issuer = {name: spec.issuer for name, spec in bonds.buckets.items()}
        kappa = {name: float(ss_yield) for name in bonds.buckets}
        return cls(pricing=bonds.pricing, kappa=kappa, decay=decay, prices=prices, issuer=issuer)

    def mark(self, instrument: str, price: float) -> float:
        """Update the market price. Returns the per-unit ΔP; never posts income."""
        prev = self.prices[instrument]
        self.prices[instrument] = float(price)
        return float(price) - prev

    def unit_value(self, instrument: str) -> float:
        if self.pricing == "par":
            return 1.0
        return float(self.prices[instrument])

    def market_value(self, face: float, instrument: str) -> float:
        return float(face) * self.unit_value(instrument)

    def to_state(self) -> dict[str, Any]:
        return {
            "pricing": self.pricing,
            "kappa": dict(self.kappa),
            "decay": dict(self.decay),
            "prices": dict(self.prices),
            "issuer": dict(self.issuer),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BondBooks:
        return cls(
            pricing=state["pricing"],
            kappa={k: float(v) for k, v in state["kappa"].items()},
            decay={k: float(v) for k, v in state["decay"].items()},
            prices={k: float(v) for k, v in state["prices"].items()},
            issuer={k: str(v) for k, v in state["issuer"].items()},
        )


def issuance_mix_tuple(bonds: BondsFile | None) -> tuple[tuple[str, float], ...]:
    """Opening / issue mix. Default matches ``opening.GOVT_MIX`` exactly."""
    if bonds is None:
        return tuple((n, DEFAULT_ISSUANCE_MIX[n]) for n in GOVT_BUCKETS)
    return bonds.mix_tuple()
