"""Gymnasium single-agent wrapper (T7.08 / §7.3).

Factored ``Dict`` action: ``K`` order slots × (instrument, side, type, size
fraction, limit offset in spreads) and one normalised ``Box`` per firm lever
plus an autopilot mask. Observations are a fixed-shape encoder over the
published ``LocalClient.observe`` whitelist. Dynamics use the ``World`` seed
only — no global RNG, no wall-clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketsim.api.schemas import OrderRequest
from marketsim.firms.firm import LEVERS, FirmsFile
from marketsim.firms.levers import FirmDecision
from marketsim.layer1.build_io import CODES
from marketsim.market.instruments import NPC_PREFIX, MMCfg
from marketsim.sdk.local import LocalClient
from marketsim.world import FirmAgentModule, World

RewardMode = Literal["operator", "trader"]

# Slot 0 = no order. Mask 0 = leave the lever on autopilot (§5.4 / §7.3).
NO_ORDER = 0
MASK_AUTOPILOT = 0
MASK_AGENT = 1
SIDES: tuple[str, ...] = ("buy", "sell")
ORDER_TYPES: tuple[str, ...] = ("market", "limit", "stop")
# Encoder Box bound (cr / index / count). Caps the *space*, not world prices.
_OBS_ABS = 1.0e9
# Fixed encoder widths (§7.3). Not calibration.
_N_NEWS = 2
_N_MACRO = 2
_N_FIRM_FEATS = 4  # equity cr, cash cr, employees persons, dividend cr
_MACRO_KEYS: tuple[str, ...] = ("cpi", "unemployment")


def default_instruments(n: int = 2) -> tuple[str, ...]:
    """First ``n`` NPC sector equities in ``CODES`` order. Symbols are unitless."""
    return tuple(f"{NPC_PREFIX}{code}" for code in CODES[:n])


def _f32(values: Sequence[float], shape: tuple[int, ...]) -> np.ndarray:
    """Pack ``values`` as ``float32`` with ``shape`` (encoder units)."""
    arr = np.asarray(list(values), dtype=np.float32)
    return np.reshape(arr, shape)


def _as_int_vec(value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    out = np.zeros(n, dtype=np.int64)
    take = min(int(arr.size), n)
    if take:
        out[:take] = arr[:take]
    return out


def _as_f32_vec(value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(n, dtype=np.float32)
    take = min(int(arr.size), n)
    if take:
        out[:take] = arr[:take]
    return out


def _first_number(value: Any) -> float:
    """First finite number in a nested release payload (index or rate)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else 0.0
    if isinstance(value, Mapping):
        for key in sorted(value):
            found = _first_number(value[key])
            if found != 0.0:
                return found
        return 0.0
    if isinstance(value, (list, tuple)) and value:
        return _first_number(value[0])
    return 0.0


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def net_worth_cr(raw: Mapping[str, Any]) -> float:
    """Published net worth (cr): cash + MV of holdings + own-firm equity."""
    port = _mapping(raw.get("portfolio"))
    if "net_worth" in port:
        return float(port["net_worth"] or 0.0)
    cash = float(port.get("cash") or 0.0)
    total = cash
    for row in port.get("positions") or ():
        total += float(_attr(row, "market_value", 0.0) or 0.0)
    return total


def cash_cr(raw: Mapping[str, Any]) -> float:
    """Published cash (cr)."""
    return float(_mapping(raw.get("portfolio")).get("cash") or 0.0)


def position_qty(raw: Mapping[str, Any], symbol: str) -> float:
    """Published holding in ``symbol`` (shares)."""
    for row in _mapping(raw.get("portfolio")).get("positions") or ():
        if str(_attr(row, "instrument", "")) == symbol:
            return float(_attr(row, "qty", 0.0) or 0.0)
    return 0.0


def _own_report(raw: Mapping[str, Any], firm_id: str) -> Any:
    for row in raw.get("reports") or ():
        if str(_attr(row, "firm_id", "")) == firm_id:
            return row
    return None


def firm_equity_cr(raw: Mapping[str, Any], firm_id: str) -> float:
    """Own-firm book equity (cr) when published; else 0."""
    books = _attr(_own_report(raw, firm_id), "books")
    if not isinstance(books, Mapping):
        return 0.0
    for key in ("equity", "equity_value", "book_equity"):
        if key in books:
            return float(books[key] or 0.0)
    return 0.0


def dividends_cr(raw: Mapping[str, Any], firm_id: str) -> float:
    """Dividends this observation (cr). 0 if unpublished."""
    port = _mapping(raw.get("portfolio"))
    if "dividends" in port:
        return float(port["dividends"] or 0.0)
    books = _attr(_own_report(raw, firm_id), "books")
    if isinstance(books, Mapping):
        for key in ("dividends", "dividend"):
            if key in books:
                return float(books[key] or 0.0)
    return 0.0


def is_bankrupt(raw: Mapping[str, Any], *, net_worth: float, firm_id: str) -> bool:
    """True if net worth is strictly negative (cr) or a report status is bankrupt."""
    if net_worth < 0.0:
        return True
    books = _attr(_own_report(raw, firm_id), "books")
    status = ""
    if isinstance(books, Mapping):
        status = str(books.get("status") or "")
    return status in {"bankrupt", "liquidated"}


def encode_observation(
    raw: Mapping[str, Any],
    *,
    instruments: Sequence[str],
    firm_id: str,
    horizon: int,
    n_news: int = _N_NEWS,
    n_macro: int = _N_MACRO,
) -> dict[str, np.ndarray]:
    """Fixed-size encoder over a published observe() dict.

    Units: ``tick`` days (clipped to ``horizon``); ``net_worth`` / books in cr;
    portfolio qty in shares and market value in cr; quotes are price indices;
    news is ``severity_hint`` (ordinal 0–3); macro fields keep release units.
    """
    quotes = _mapping(raw.get("quotes"))
    prices = _mapping(raw.get("prices"))
    port_qty: list[float] = []
    port_mv: list[float] = []
    mids: list[float] = []
    for symbol in instruments:
        qty = position_qty(raw, symbol)
        mv = 0.0
        for row in _mapping(raw.get("portfolio")).get("positions") or ():
            if str(_attr(row, "instrument", "")) == symbol:
                mv = float(_attr(row, "market_value", 0.0) or 0.0)
                break
        port_qty.append(qty)
        port_mv.append(mv)
        mid = quotes.get(symbol, prices.get(symbol, 0.0))
        mids.append(float(mid or 0.0))
    interleaved = [x for pair in zip(port_qty, port_mv, strict=True) for x in pair]

    news_vals = [0.0] * n_news
    for i, item in enumerate(list(raw.get("news") or ())[:n_news]):
        news_vals[i] = float(_attr(item, "severity_hint", 0) or 0.0)

    releases = _mapping(raw.get("releases"))
    macro_vals = [_first_number(releases.get(key)) for key in _MACRO_KEYS[:n_macro]]
    while len(macro_vals) < n_macro:
        macro_vals.append(0.0)

    books = _attr(_own_report(raw, firm_id), "books")
    books_m = books if isinstance(books, Mapping) else {}
    firm_vals = [
        firm_equity_cr(raw, firm_id),
        float(books_m.get("cash") or 0.0),
        float(books_m.get("employees") or 0.0),
        dividends_cr(raw, firm_id),
    ]
    tick = min(float(raw.get("tick") or 0), float(horizon))
    return {
        "tick": _f32([tick], (1,)),
        "net_worth": _f32([net_worth_cr(raw)], (1,)),
        "portfolio": _f32(interleaved, (len(instruments) * 2,)),
        "market": _f32(mids, (len(instruments),)),
        "news": _f32(news_vals, (n_news,)),
        "macro": _f32(macro_vals[:n_macro], (n_macro,)),
        "firm": _f32(firm_vals[:_N_FIRM_FEATS], (_N_FIRM_FEATS,)),
    }


class MarketSimEnv(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Single-agent Gymnasium env over ``LocalClient`` / ``World.create``.

    ``reward_mode='operator'`` is Δ(equity cr) + dividends cr. ``'trader'`` is
    risk-adjusted P&L (Δ net worth / running vol; dimensionless). Episode ends
    on bankruptcy (terminated) or ``horizon`` days (truncated).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_dir: str | Path,
        *,
        seed: int = 0,
        agent_id: str = "alice",
        firm_id: str = "acme",
        n_order_slots: int = 2,
        instruments: Sequence[str] | None = None,
        horizon: int = 8,
        reward_mode: RewardMode = "trader",
        cell: tuple[str, str] = ("INDUSTRIAL", "AUTOS"),
        modules: Sequence[Any] | None = None,
    ) -> None:
        super().__init__()
        if n_order_slots < 1:
            raise ValueError("n_order_slots must be >= 1")
        if horizon < 2:
            raise ValueError("horizon must be >= 2 (days) so a 1-step episode is not truncated")
        if reward_mode not in ("operator", "trader"):
            raise ValueError("reward_mode must be 'operator' or 'trader'")
        self.config_dir = Path(config_dir)
        self._seed = int(seed)
        self.agent_id = str(agent_id)
        self.firm_id = str(firm_id)
        self.n_order_slots = int(n_order_slots)
        self.instruments: tuple[str, ...] = tuple(instruments) if instruments else default_instruments(2)
        self.horizon = int(horizon)  # days
        self.reward_mode: RewardMode = reward_mode
        self.cell: tuple[str, str] = (str(cell[0]), str(cell[1]))
        self._modules: list[Any] = list(modules) if modules is not None else [FirmAgentModule()]
        self._world: World | None = None
        self._client: LocalClient | None = None
        self._raw: dict[str, Any] = {}
        self._prev_nw = 0.0  # cr
        self._prev_equity = 0.0  # cr
        self._pnl_n = 0
        self._pnl_mean = 0.0  # cr
        self._pnl_m2 = 0.0  # cr^2
        self.last_decision: FirmDecision | None = None
        self.last_orders: list[OrderRequest] = []
        self._build_spaces()

    def _build_spaces(self) -> None:
        k = self.n_order_slots
        n_inst = len(self.instruments)
        n_choice = n_inst + 1  # + NO_ORDER
        abs_b = np.float32(_OBS_ABS)
        self.action_space = spaces.Dict(
            {
                "orders": spaces.Dict(
                    {
                        "instrument": spaces.MultiDiscrete([n_choice] * k),
                        "side": spaces.MultiDiscrete([len(SIDES)] * k),
                        "type": spaces.MultiDiscrete([len(ORDER_TYPES)] * k),
                        "size": spaces.Box(0.0, 1.0, shape=(k,), dtype=np.float32),
                        "limit_offset": spaces.Box(-1.0, 1.0, shape=(k,), dtype=np.float32),
                    }
                ),
                "firm": spaces.Dict(
                    {
                        **{name: spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32) for name in LEVERS},
                        "mask": spaces.MultiBinary(len(LEVERS)),
                    }
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "tick": spaces.Box(0.0, float(self.horizon), shape=(1,), dtype=np.float32),
                "net_worth": spaces.Box(-abs_b, abs_b, shape=(1,), dtype=np.float32),
                "portfolio": spaces.Box(-abs_b, abs_b, shape=(n_inst * 2,), dtype=np.float32),
                "market": spaces.Box(-abs_b, abs_b, shape=(n_inst,), dtype=np.float32),
                "news": spaces.Box(-abs_b, abs_b, shape=(_N_NEWS,), dtype=np.float32),
                "macro": spaces.Box(-abs_b, abs_b, shape=(_N_MACRO,), dtype=np.float32),
                "firm": spaces.Box(-abs_b, abs_b, shape=(_N_FIRM_FEATS,), dtype=np.float32),
            }
        )

    def _build(self) -> None:
        self._world = World.create(self.config_dir, seed=self._seed, modules=self._modules)
        self._client = LocalClient(self._world)

    @property
    def world(self) -> World:
        if self._world is None:
            self._build()
        assert self._world is not None
        return self._world

    @property
    def client(self) -> LocalClient:
        if self._client is None:
            self._build()
        assert self._client is not None
        return self._client

    def _firms_cfg(self) -> FirmsFile:
        cfg = self.world.cfg.firms
        return cfg if cfg is not None else FirmsFile()

    def _half_spread(self) -> float:
        """Engine MM half-spread floor (dimensionless). Shipped ``markets.mm.s0``."""
        markets = self.world.cfg.markets
        if markets is not None:
            return float(markets.mm.s0)
        return float(MMCfg().s0)

    def _clear_episode(self) -> None:
        self._prev_nw = 0.0
        self._prev_equity = 0.0
        self._pnl_n = 0
        self._pnl_mean = 0.0
        self._pnl_m2 = 0.0
        self.last_decision = None
        self.last_orders = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Rewind to tick 0 (days). ``seed`` is the ``World`` root seed (unitless)."""
        super().reset(seed=seed)
        del options
        if seed is not None:
            new_seed = int(seed)
            if self._world is None or new_seed != self._world.seed:
                self._seed = new_seed
                self._build()
            else:
                self.world.reset()
                self._client = LocalClient(self.world)
        elif self._world is None:
            self._build()
        else:
            self.world.reset()
            self._client = LocalClient(self.world)
        self._clear_episode()
        raw = self.client.observe(self.agent_id)
        self._raw = raw
        self._prev_nw = net_worth_cr(raw)
        self._prev_equity = firm_equity_cr(raw, self.firm_id)
        obs = self.encode_observation(raw)
        return obs, self._info(raw)

    def encode_observation(self, raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Fixed-size observation. See :func:`encode_observation` for units."""
        return encode_observation(
            raw,
            instruments=self.instruments,
            firm_id=self.firm_id,
            horizon=self.horizon,
        )

    def idle_action(self) -> dict[str, Any]:
        """No-op orders; every lever masked (autopilot). Deterministic, no RNG."""
        k = self.n_order_slots
        firm = {name: np.zeros(1, dtype=np.float32) for name in LEVERS}
        firm["mask"] = np.zeros(len(LEVERS), dtype=np.int8)
        return {
            "orders": {
                "instrument": np.zeros(k, dtype=np.int64),
                "side": np.zeros(k, dtype=np.int64),
                "type": np.zeros(k, dtype=np.int64),
                "size": np.zeros(k, dtype=np.float32),
                "limit_offset": np.zeros(k, dtype=np.float32),
            },
            "firm": firm,
        }

    def decode_firm(self, action: Mapping[str, Any]) -> FirmDecision:
        """Map the firm ``Dict`` to a ``FirmDecision``. Masked levers stay ``None`` (autopilot)."""
        block = _mapping(action.get("firm"))
        mask = _as_int_vec(block.get("mask"), len(LEVERS))
        decision = FirmDecision(firm_id=self.firm_id, operator=self.agent_id)
        firms = self._firms_cfg()
        cell = self.cell
        for i, name in enumerate(LEVERS):
            if int(mask[i]) == MASK_AUTOPILOT:
                continue
            value = float(_as_f32_vec(block.get(name), 1)[0])
            unit = 0.5 * (value + 1.0)  # Box [-1, 1] → [0, 1]
            if name == "pricing":
                # index; 1.0 at baseline; ±step_max_month log-points (§5.4)
                decision.posted_price = {cell: float(math.exp(value * firms.pricing.step_max_month))}
            elif name == "production":
                decision.target_output = {cell: unit}  # units / month
            elif name == "labour":
                decision.vacancies = unit  # persons
            elif name == "procurement":
                decision.input_cover_m = unit * float(firms.procurement.input_cover_max_m)  # months
            elif name == "capex":
                decision.rnd_spend = unit  # cr / month
            elif name == "financing":
                decision.dividend = unit  # cr
            elif name == "treasury":
                decision.treasury_enabled = value > 0.0
            elif name == "exit":
                decision.liquidate = value > 0.0
        return decision

    def decode_orders(self, action: Mapping[str, Any]) -> list[OrderRequest]:
        """Map ``K`` slots to ``OrderRequest``. ``qty`` is shares; empty slots are dropped."""
        block = _mapping(action.get("orders"))
        k = self.n_order_slots
        inst = _as_int_vec(block.get("instrument"), k)
        side_i = _as_int_vec(block.get("side"), k)
        type_i = _as_int_vec(block.get("type"), k)
        size = _as_f32_vec(block.get("size"), k)
        offset = _as_f32_vec(block.get("limit_offset"), k)
        spread = self._half_spread()
        quotes = _mapping(self._raw.get("quotes"))
        prices = _mapping(self._raw.get("prices"))
        orders: list[OrderRequest] = []
        for slot in range(k):
            idx = int(inst[slot])
            frac = float(size[slot])
            if idx <= NO_ORDER or idx > len(self.instruments) or frac <= 0.0:
                continue
            symbol = self.instruments[idx - 1]
            side = SIDES[int(side_i[slot]) % len(SIDES)]
            otype = ORDER_TYPES[int(type_i[slot]) % len(ORDER_TYPES)]
            mid = float(quotes.get(symbol, prices.get(symbol, 1.0)) or 1.0)
            if mid <= 0.0:
                mid = 1.0  # baseline price index
            limit_px = mid * (1.0 + float(offset[slot]) * spread)
            if limit_px <= 0.0:
                limit_px = mid
            qty = self._order_qty(side, symbol, frac, mid if otype == "market" else limit_px)
            if qty < 1:
                continue
            price = limit_px if otype == "limit" else None
            stop = limit_px if otype == "stop" else None
            side_name: Literal["buy", "sell"] = "buy" if side == "buy" else "sell"
            type_name: Literal["market", "limit", "stop"]
            if otype == "market":
                type_name = "market"
            elif otype == "limit":
                type_name = "limit"
            else:
                type_name = "stop"
            orders.append(
                OrderRequest(
                    agent_id=self.agent_id,
                    side=side_name,
                    qty=qty,
                    symbol=symbol,
                    order_type=type_name,
                    price=price,
                    stop_price=stop,
                )
            )
        return orders

    def _order_qty(self, side: str, symbol: str, frac: float, px: float) -> int:
        """Size as a fraction of buying power (cr) or position (shares) → shares."""
        price = px if px > 0.0 else 1.0
        if side == "buy":
            return int(frac * cash_cr(self._raw) / price)
        return int(frac * max(position_qty(self._raw, symbol), 0.0))

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one tick (day). Reward units: operator cr; trader dimensionless."""
        orders = self.decode_orders(action)
        decision = self.decode_firm(action)
        self.last_orders = orders
        self.last_decision = decision
        if orders:
            self.client.submit_orders(self.agent_id, orders)
        self.client.submit_decisions(self.agent_id, decision)
        self.client.step(1)
        raw = self.client.observe(self.agent_id)
        self._raw = raw
        nw = net_worth_cr(raw)
        equity = firm_equity_cr(raw, self.firm_id)
        div = dividends_cr(raw, self.firm_id)
        reward = self._reward(nw, equity, div)
        self._prev_nw = nw
        self._prev_equity = equity
        terminated = is_bankrupt(raw, net_worth=nw, firm_id=self.firm_id)
        truncated = int(raw.get("tick") or 0) >= self.horizon
        return self.encode_observation(raw), float(reward), bool(terminated), bool(truncated), self._info(raw)

    def _reward(self, nw: float, equity: float, div: float) -> float:
        """Operator: Δ equity + dividends (cr). Trader: ΔNW / (σ + 1 cr)."""
        if self.reward_mode == "operator":
            if equity != 0.0 or self._prev_equity != 0.0:
                return (equity - self._prev_equity) + div
            return nw - self._prev_nw
        pnl = nw - self._prev_nw  # cr
        self._pnl_n += 1
        delta = pnl - self._pnl_mean
        self._pnl_mean += delta / float(self._pnl_n)
        self._pnl_m2 += delta * (pnl - self._pnl_mean)
        var = self._pnl_m2 / float(self._pnl_n)
        vol = math.sqrt(max(var, 0.0))
        return pnl / (vol + 1.0)  # 1 cr floor so σ=0 is defined

    def _info(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        nw = net_worth_cr(raw)
        return {
            "tick": int(raw.get("tick") or 0),  # days
            "net_worth": float(nw),  # cr
            "bankrupt": bool(is_bankrupt(raw, net_worth=nw, firm_id=self.firm_id)),
        }

    def close(self) -> None:
        self._world = None
        self._client = None
