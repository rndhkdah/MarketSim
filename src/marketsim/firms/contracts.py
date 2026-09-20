"""Fixed-price / fixed-quantity forward contracts (T8.02).

A priority layer in front of ``goods_market.allocate``: contracted quantity is served
first, leftover seller availability is shared pro-rata on the spot market. Settlement
cash uses only the contract's agreed price (D14: no firm-specific spot premia here).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent

SETTLE_TAG = "contract_settle"
"""Ledger flow tag for contract settlement cash (cr)."""

# Skip empty leftover / cash; same water-fill floor as ``firms.procurement``.
_QTY_EPS = 1e-15


@dataclass(frozen=True)
class ForwardContract:
    """Fixed-price, fixed-quantity forward between a buyer and a supplier cell.

    ``qty`` is units/month; ``price`` is a goods-price index treated as cr/unit at
    settlement; ``remaining_qty`` is units still deliverable on the contract;
    ``remaining_months`` is months of tenor left. ``buyer_id`` / ``seller_id`` are
    ledger entities (``FIRM:<id>`` or ``NPC:<region>:<sector>``).
    """

    buyer_id: str
    seller_id: str
    region: str
    sector: str
    qty: float
    price: float
    remaining_qty: float
    remaining_months: int

    def monthly_claim(self) -> float:
        """This month's contracted quantity (units). ``min(qty, remaining_qty)``."""
        if self.remaining_months <= 0:
            return 0.0
        return min(max(self.qty, 0.0), max(self.remaining_qty, 0.0))


def open_forward(
    *,
    buyer_id: str,
    seller_id: str,
    region: str,
    sector: str,
    qty: float,
    price: float,
    months: int,
) -> ForwardContract:
    """Build a forward with ``remaining_qty = qty * months`` (units) and tenor ``months``."""
    m = int(months)
    q = float(qty)
    return ForwardContract(
        buyer_id=buyer_id,
        seller_id=seller_id,
        region=region,
        sector=sector,
        qty=q,
        price=float(price),
        remaining_qty=q * m,
        remaining_months=m,
    )


@dataclass(frozen=True)
class SettlementResult:
    """One month of contract-then-spot allocation.

    Fills and wanted quantities are units; ``cash_cr`` is cr posted on DEP this month.
    Dict keys are ``(buyer_id, seller_id)``.
    """

    contract_fill: dict[tuple[str, str], float]
    spot_fill: dict[tuple[str, str], float]
    contract_wanted: dict[tuple[str, str], float]
    spot_wanted: dict[tuple[str, str], float]
    avail_left: dict[str, float]
    cash_cr: float

    def fill_rate(self, buyer_id: str, seller_id: str, *, contracted: bool) -> float:
        """Filled / wanted this month (dimensionless). ``0`` when wanted is ``0``."""
        key = (buyer_id, seller_id)
        if contracted:
            want = self.contract_wanted.get(key, 0.0)
            got = self.contract_fill.get(key, 0.0)
        else:
            want = self.spot_wanted.get(key, 0.0)
            got = self.spot_fill.get(key, 0.0)
        if want <= 0.0:
            return 0.0
        return got / want


def _scale(claims: list[float], supply: float) -> list[float]:
    """Pro-rata ``claims`` into ``supply`` (same units). Caps at each claim."""
    n = len(claims)
    if n == 0:
        return []
    vals = [max(c, 0.0) for c in claims]
    tot = sum(vals)
    if tot <= 0.0 or supply <= 0.0:
        return [0.0] * n
    if tot <= supply:
        return vals
    scale = supply / tot
    return [v * scale for v in vals]


def _add(acc: dict[tuple[str, str], float], key: tuple[str, str], amt: float) -> None:
    acc[key] = acc.get(key, 0.0) + amt


def _ensure_settle_tag(ledger: Ledger) -> None:
    if SETTLE_TAG not in ledger.flow_tags:
        ledger.flow_tags = (*ledger.flow_tags, SETTLE_TAG)


@dataclass
class ContractBook:
    """Open forwards. Insertion order is preserved; seller grouping is sorted."""

    contracts: list[ForwardContract] = field(default_factory=list)

    def add(self, contract: ForwardContract) -> ForwardContract:
        """Append an open forward. Returns the contract."""
        self.contracts.append(contract)
        return contract

    def __iter__(self) -> Iterator[ForwardContract]:
        return iter(self.contracts)

    def __len__(self) -> int:
        return len(self.contracts)

    def settle_month(
        self,
        avail_by_seller: Mapping[str, float],
        spot_demand: Mapping[str, Mapping[str, float]],
        *,
        ledger: Ledger | None = None,
        tick: int = 0,
    ) -> SettlementResult:
        """Serve contracts first, then leftover availability on the spot market.

        ``avail_by_seller`` maps seller entity → available units this month.
        ``spot_demand`` maps seller entity → (buyer entity → units wanted on spot).
        Contracted claims take priority: fill = min(remaining claim, avail), pro-rata
        across contracts on the same seller when they jointly exceed avail. Leftover
        avail is shared pro-rata among spot buyers. During a supply cut, a contracted
        buyer who wanted the same qty as a spot buyer has fill rate ≥ the spot buyer's.

        When ``ledger`` is set, each filled contract posts ``price * filled`` cr on DEP
        (buyer pays, seller receives) tagged ``contract_settle``, then ``assert_consistent``.
        """
        avail = {str(s): max(float(q), 0.0) for s, q in avail_by_seller.items()}
        spot: dict[str, dict[str, float]] = {}
        for seller, buyers in spot_demand.items():
            spot[str(seller)] = {str(b): max(float(q), 0.0) for b, q in buyers.items()}

        sellers = sorted(set(avail) | set(spot) | {c.seller_id for c in self.contracts})
        for s in sellers:
            avail.setdefault(s, 0.0)

        contract_wanted: dict[tuple[str, str], float] = {}
        contract_fill: dict[tuple[str, str], float] = {}
        per_index = [0.0] * len(self.contracts)

        for seller in sellers:
            idxs = [
                i
                for i, c in enumerate(self.contracts)
                if c.seller_id == seller and c.monthly_claim() > 0.0
            ]
            claims = [self.contracts[i].monthly_claim() for i in idxs]
            for i, claim in zip(idxs, claims, strict=True):
                c = self.contracts[i]
                _add(contract_wanted, (c.buyer_id, seller), claim)
            filled = _scale(claims, avail[seller])
            used = 0.0
            for i, take in zip(idxs, filled, strict=True):
                c = self.contracts[i]
                per_index[i] = take
                _add(contract_fill, (c.buyer_id, seller), take)
                used += take
            avail[seller] = max(avail[seller] - used, 0.0)

        spot_wanted: dict[tuple[str, str], float] = {}
        spot_fill: dict[tuple[str, str], float] = {}
        for seller in sellers:
            buyers = spot.get(seller, {})
            keys = sorted(buyers)
            claims = [buyers[b] for b in keys]
            for b, claim in zip(keys, claims, strict=True):
                _add(spot_wanted, (b, seller), claim)
            filled = _scale(claims, avail[seller])
            used = 0.0
            for b, take in zip(keys, filled, strict=True):
                _add(spot_fill, (b, seller), take)
                used += take
            avail[seller] = max(avail[seller] - used, 0.0)

        cash_cr = 0.0
        if ledger is not None:
            _ensure_settle_tag(ledger)
            for i, c in enumerate(self.contracts):
                take = per_index[i]
                cash = c.price * take
                if cash <= _QTY_EPS:
                    continue
                ledger.register_entity(c.buyer_id)
                ledger.register_entity(c.seller_id)
                ledger.post(
                    Tx(
                        tick,
                        SETTLE_TAG,
                        (
                            Entry(c.buyer_id, "DEP", -cash),
                            Entry(c.seller_id, "DEP", cash),
                        ),
                        memo="contract settle",
                    )
                )
                cash_cr += cash
            assert_consistent(ledger)

        updated: list[ForwardContract] = []
        for i, c in enumerate(self.contracts):
            nxt = replace(
                c,
                remaining_qty=max(c.remaining_qty - per_index[i], 0.0),
                remaining_months=c.remaining_months - 1,
            )
            if nxt.remaining_months > 0 and nxt.remaining_qty > _QTY_EPS:
                updated.append(nxt)
        self.contracts = updated

        return SettlementResult(
            contract_fill=contract_fill,
            spot_fill=spot_fill,
            contract_wanted=contract_wanted,
            spot_wanted=spot_wanted,
            avail_left=dict(avail),
            cash_cr=cash_cr,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "contracts": [
                {
                    "buyer_id": c.buyer_id,
                    "seller_id": c.seller_id,
                    "region": c.region,
                    "sector": c.sector,
                    "qty": c.qty,
                    "price": c.price,
                    "remaining_qty": c.remaining_qty,
                    "remaining_months": c.remaining_months,
                }
                for c in self.contracts
            ]
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ContractBook:
        rows = []
        for row in state.get("contracts", ()):
            rows.append(
                ForwardContract(
                    buyer_id=str(row["buyer_id"]),
                    seller_id=str(row["seller_id"]),
                    region=str(row["region"]),
                    sector=str(row["sector"]),
                    qty=float(row["qty"]),
                    price=float(row["price"]),
                    remaining_qty=float(row["remaining_qty"]),
                    remaining_months=int(row["remaining_months"]),
                )
            )
        return cls(contracts=rows)
