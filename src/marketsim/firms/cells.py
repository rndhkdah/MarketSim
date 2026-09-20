"""NPC + firm cell aggregates (T5.04 / §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marketsim.firms.firm import FirmRegistry
from marketsim.firms.founding import CellBook, NpcCell


def safe_share(num: float, den: float) -> float:
    """Utilisation or weight. Dimensionless; 0 when ``den`` is 0 (never NaN)."""
    if den <= 0.0:
        return 0.0
    return float(num) / float(den)


def reference_price(
    *,
    npc_price: float,
    npc_sales: float,
    npc_capacity: float,
    firm_prices: np.ndarray,
    firm_sales: np.ndarray,
    firm_capacity: np.ndarray,
) -> float:
    """Sales-weighted posted price; capacity-weighted if no sales; firms only if NPC K = 0."""
    prices = np.asarray(firm_prices, dtype=float)
    sales = np.asarray(firm_sales, dtype=float)
    caps = np.asarray(firm_capacity, dtype=float)
    if npc_capacity <= 0.0:
        w = float(sales.sum())
        if w > 0.0:
            return float((prices * sales).sum() / w)
        wcap = float(caps.sum())
        if wcap > 0.0:
            return float((prices * caps).sum() / wcap)
        return float(prices.mean()) if prices.size else float("nan")
    w = float(npc_sales) + float(sales.sum())
    if w > 0.0:
        return (float(npc_price) * float(npc_sales) + float((prices * sales).sum())) / w
    wcap = float(npc_capacity) + float(caps.sum())
    if wcap > 0.0:
        return (float(npc_price) * float(npc_capacity) + float((prices * caps).sum())) / wcap
    return float(npc_price)


@dataclass
class CellAggregate:
    """One (region, sector) cell. Real quantities are units / month; prices are an index."""

    region: str
    sector: str
    npc_capacity: float
    firm_capacity: float
    npc_output: float
    firm_output: float
    npc_employment: float
    firm_employment: float
    npc_price: float
    p_ref: float
    sales: float

    @property
    def capacity(self) -> float:
        return self.npc_capacity + self.firm_capacity

    @property
    def output(self) -> float:
        return self.npc_output + self.firm_output

    @property
    def employment(self) -> float:
        return self.npc_employment + self.firm_employment

    @property
    def utilisation(self) -> float:
        return safe_share(self.output, self.capacity)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.capacity,
                self.output,
                self.employment,
                self.p_ref,
                self.sales,
                self.utilisation,
            ],
            dtype=float,
        )


def aggregate_cell(
    *,
    region: str,
    sector: str,
    npc_capacity: float,
    npc_output: float,
    npc_employment: float,
    npc_price: float,
    npc_sales: float,
    firm_capacity: np.ndarray,
    firm_output: np.ndarray,
    firm_employment: np.ndarray,
    firm_prices: np.ndarray,
    firm_sales: np.ndarray,
) -> CellAggregate:
    """Sums NPC + firms. ``p_ref`` follows §5.3 (sales-weighted; firms-only if no NPC)."""
    p_ref = reference_price(
        npc_price=npc_price,
        npc_sales=npc_sales,
        npc_capacity=npc_capacity,
        firm_prices=firm_prices,
        firm_sales=firm_sales,
        firm_capacity=firm_capacity,
    )
    return CellAggregate(
        region=region,
        sector=sector,
        npc_capacity=float(npc_capacity),
        firm_capacity=float(np.asarray(firm_capacity, dtype=float).sum()),
        npc_output=float(npc_output),
        firm_output=float(np.asarray(firm_output, dtype=float).sum()),
        npc_employment=float(npc_employment),
        firm_employment=float(np.asarray(firm_employment, dtype=float).sum()),
        npc_price=float(npc_price),
        p_ref=float(p_ref),
        sales=float(npc_sales) + float(np.asarray(firm_sales, dtype=float).sum()),
    )


def firm_cell_arrays(
    registry: FirmRegistry, region: str, sector: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-firm (capacity, output=capacity, employment share, posted price, sales).

    Output defaults to plant capacity (autopilot-full utilisation) until T5.07.
    Employment is the firm's headcount allocated to this cell by capacity share.
    """
    caps: list[float] = []
    prices: list[float] = []
    sales: list[float] = []
    emps: list[float] = []
    for firm in registry:
        cell_cap = 0.0
        total_cap = 0.0
        for plant in firm.plants:
            total_cap += plant.capacity
            if plant.cell == (region, sector):
                cell_cap += plant.capacity
        if cell_cap <= 0.0:
            continue
        caps.append(cell_cap)
        prices.append(float(firm.posted_price.get((region, sector), 1.0)))
        sales.append(float(firm.customer_share.get((region, sector), 0.0)) * cell_cap)
        share = cell_cap / total_cap if total_cap > 0 else 0.0
        emps.append(firm.employees * share)
    if not caps:
        z = np.zeros(0, dtype=float)
        return z, z, z, z, z
    c = np.asarray(caps, dtype=float)
    return c, c.copy(), np.asarray(emps, dtype=float), np.asarray(prices, dtype=float), np.asarray(sales, dtype=float)


def aggregates_for_registry(
    registry: FirmRegistry,
    book: CellBook,
    *,
    npc_output: dict[tuple[str, str], float] | None = None,
    npc_employment: dict[tuple[str, str], float] | None = None,
    npc_price: dict[tuple[str, str], float] | None = None,
    npc_sales: dict[tuple[str, str], float] | None = None,
) -> list[CellAggregate]:
    """One aggregate per NPC cell, sorted by (region, sector)."""
    out: list[CellAggregate] = []
    for key in sorted(book.cells):
        cell = book.cells[key]
        fc, fo, fe, fp, fs = firm_cell_arrays(registry, cell.region, cell.sector)
        out.append(
            aggregate_cell(
                region=cell.region,
                sector=cell.sector,
                npc_capacity=cell.capacity,
                npc_output=float((npc_output or {}).get(key, 0.0)),
                npc_employment=float((npc_employment or {}).get(key, 0.0)),
                npc_price=float((npc_price or {}).get(key, 1.0)),
                npc_sales=float((npc_sales or {}).get(key, 0.0)),
                firm_capacity=fc,
                firm_output=fo,
                firm_employment=fe,
                firm_prices=fp,
                firm_sales=fs,
            )
        )
    return out


def step_zero_npc_cell(
    registry: FirmRegistry,
    *,
    region: str,
    sector: str,
    demand: float,
    months: int,
) -> list[CellAggregate]:
    """Advance a no-NPC cell ``months`` months. Demand is real units / month."""
    hist: list[CellAggregate] = []
    for _ in range(months):
        fc, fo, fe, fp, fs = firm_cell_arrays(registry, region, sector)
        supply = float(fo.sum()) if fo.size else 0.0
        fill = min(1.0, safe_share(demand, supply)) if supply > 0 else 0.0
        firm_sales = fo * fill
        for firm in registry:
            if (region, sector) in {p.cell for p in firm.plants}:
                firm.customer_share[(region, sector)] = fill
        agg = aggregate_cell(
            region=region,
            sector=sector,
            npc_capacity=0.0,
            npc_output=0.0,
            npc_employment=0.0,
            npc_price=1.0,
            npc_sales=0.0,
            firm_capacity=fc,
            firm_output=fo,
            firm_employment=fe,
            firm_prices=fp,
            firm_sales=firm_sales,
        )
        hist.append(agg)
    return hist


def economy_cell_aggregates(economy: Any) -> list[CellAggregate]:
    """Read NPC arrays off ``RealEconomy`` and add attached firms (R=1 uses all regions)."""
    registry: FirmRegistry | None = getattr(economy, "firm_registry", None)
    book: CellBook | None = getattr(economy, "cell_book", None)
    codes: tuple[str, ...] = tuple(economy.codes)
    if book is None:
        book = CellBook()
        k = np.asarray(economy.k, dtype=float).reshape(-1, len(codes))
        x = np.asarray(economy.x, dtype=float).reshape(-1, len(codes))
        n = np.asarray(economy.n, dtype=float).reshape(-1, len(codes))
        p = np.asarray(economy.p, dtype=float).reshape(-1, len(codes))
        sales = np.asarray(economy.sales, dtype=float).reshape(-1, len(codes))
        regions = ("NATIONAL",) if k.shape[0] == 1 else tuple(economy.geom.codes)
        for ri, region in enumerate(regions):
            for si, sector in enumerate(codes):
                book.cells[(region, sector)] = NpcCell(region, sector, float(k[ri, si]), 0.0)
        npc_output = {(r, s): float(x[i, j]) for i, r in enumerate(regions) for j, s in enumerate(codes)}
        npc_emp = {(r, s): float(n[i, j]) for i, r in enumerate(regions) for j, s in enumerate(codes)}
        npc_price = {(r, s): float(p[i, j]) for i, r in enumerate(regions) for j, s in enumerate(codes)}
        npc_sales = {(r, s): float(sales[i, j]) for i, r in enumerate(regions) for j, s in enumerate(codes)}
    else:
        k = np.asarray(economy.k, dtype=float).reshape(-1, len(codes))
        x = np.asarray(economy.x, dtype=float).reshape(-1, len(codes))
        n = np.asarray(economy.n, dtype=float).reshape(-1, len(codes))
        p = np.asarray(economy.p, dtype=float).reshape(-1, len(codes))
        sales = np.asarray(economy.sales, dtype=float).reshape(-1, len(codes))
        npc_output = {}
        npc_emp = {}
        npc_price = {}
        npc_sales = {}
        for region, sector in book.cells:
            si = codes.index(sector)
            npc_output[(region, sector)] = float(x[0, si])
            npc_emp[(region, sector)] = float(n[0, si])
            npc_price[(region, sector)] = float(p[0, si])
            npc_sales[(region, sector)] = float(sales[0, si])
    if registry is None:
        registry = FirmRegistry(cells=book.cells)
    return aggregates_for_registry(
        registry,
        book,
        npc_output=npc_output,
        npc_employment=npc_emp,
        npc_price=npc_price,
        npc_sales=npc_sales,
    )
