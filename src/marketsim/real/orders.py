"""Orders, rationing, deliveries and input stocks (R5–R6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def order_matrix(
    a: np.ndarray,
    plan: np.ndarray,
    s_in: np.ndarray,
    stor_in: np.ndarray,
    n_in: float,
    tau_in: float,
) -> np.ndarray:
    """Buyer-plan orders ``O[i,j]`` (cr/month). Floored at 0."""
    o = a * plan[None, :]
    o = o + np.where(stor_in, (n_in * a * plan[None, :] - s_in) / tau_in, 0.0)
    return np.maximum(o, 0.0)


@dataclass
class SupplyResult:
    """One-month supply / rationing outcome. All real quantities in cr/month or cr."""

    x: np.ndarray
    sales: np.ndarray
    inv: np.ndarray
    backlog: np.ndarray
    deliveries: np.ndarray
    s_in: np.ndarray
    fill: np.ndarray
    dshare: np.ndarray
    final_sales: np.ndarray


def supply_and_ration(
    *,
    x_goods: np.ndarray,
    new_orders: np.ndarray,
    final: np.ndarray,
    orders: np.ndarray,
    a: np.ndarray,
    inv: np.ndarray,
    backlog: np.ndarray,
    s_in: np.ndarray,
    leak: np.ndarray,
    tau_ob: np.ndarray,
    cap: np.ndarray,
    flow_crit: np.ndarray,
    is_stock: np.ndarray,
    is_order: np.ndarray,
    is_flow: np.ndarray,
    stor_in: np.ndarray,
    backlog_loss: float,
) -> SupplyResult:
    """Three-mode supply, pro-rata fill, deliveries and ``S_in`` update."""
    d_stock = new_orders + np.where(is_stock, backlog, 0.0)
    x = np.where(is_stock, x_goods, 0.0)
    ob = backlog + new_orders
    x = np.where(is_order, np.minimum(ob / tau_ob, cap * flow_crit), x)
    x = np.where(is_flow, np.minimum(d_stock, cap * flow_crit), x)
    avail = np.where(is_stock, x + (1.0 - leak) * inv, x)
    dem_eff = np.where(is_order, x, d_stock)
    fill = np.minimum(1.0, avail / np.maximum(dem_eff, 1e-12))
    sales = fill * dem_eff
    inv_n = np.where(is_stock, avail - sales, 0.0)
    backlog_n = np.where(
        is_stock,
        (d_stock - sales) * (1.0 - backlog_loss),
        np.where(is_order, ob - x, 0.0),
    )
    dshare = np.where(is_order, x / np.maximum(ob, 1e-12), fill)
    deliv = orders * dshare[:, None]
    used = a * x[None, :]
    s_in_n = np.where(stor_in, np.maximum(s_in + deliv - used, 0.0), 0.0)
    final_sales = final * dshare
    return SupplyResult(
        x=x,
        sales=sales,
        inv=inv_n,
        backlog=backlog_n,
        deliveries=deliv,
        s_in=s_in_n,
        fill=fill,
        dshare=dshare,
        final_sales=final_sales,
    )
