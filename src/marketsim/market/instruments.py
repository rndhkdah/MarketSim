"""Tradable-instrument registry and `markets.yaml` schema (T6.01 / §6.2)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.core.errors import ConfigError, StateError

KIND = Literal["eq_npc", "govt_bond", "corp_pool", "cash", "commodity", "eq_firm", "index"]
VENUE = Literal["engine_mm", "auction_mm", "clob", "none"]
PRICING_MODE = Literal["structural", "factor_lite"]

KINDS: tuple[str, ...] = ("eq_npc", "govt_bond", "corp_pool", "cash", "commodity", "eq_firm", "index")
VENUES: tuple[str, ...] = ("engine_mm", "auction_mm", "clob", "none")
FIRM_PREFIX = "EQ:FIRM:"
NPC_PREFIX = "EQ:NPC:"

COMMODITY_SECTORS: dict[str, str] = {
    "OIL": "ENERGY",
    "METALS": "MATERIALS",
    "GRAINS": "AGRIFOOD",
}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _positive(name: str, v: float) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _nonneg(name: str, v: float) -> float:
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


def _unit_interval(name: str, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return v


class PricingCfg(FrozenModel):
    """§6.3 curve / earnings constants used by later pricing cards."""

    mode: PRICING_MODE = "structural"
    lambda_m: float = 0.978  # monthly; E[r_{t+h}] decay
    horizon_m: int = 120  # months in the 10-year average
    g_lr_anchor: float = 0.1  # g_lr = this × published growth
    tau_ee_m: float = 12.0  # months; published-profit EMA

    @field_validator("lambda_m")
    @classmethod
    def _lam(cls, v: float) -> float:
        return _unit_interval("lambda_m", v)

    @field_validator("horizon_m")
    @classmethod
    def _hor(cls, v: int) -> int:
        if v < 1:
            raise ValueError("horizon_m must be >= 1")
        return v

    @field_validator("g_lr_anchor", "tau_ee_m")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("pricing param", v)


class NpcEquityCfg(FrozenModel):
    prefix: str = NPC_PREFIX
    venue: Literal["engine_mm"] = "engine_mm"

    @field_validator("prefix")
    @classmethod
    def _pref(cls, v: str) -> str:
        if not v:
            raise ValueError("npc_equity.prefix must be non-empty")
        return v


class InstrumentSpec(FrozenModel):
    symbol: str
    kind: KIND
    venue: VENUE
    tradable: bool = True
    sector: str | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        if not v or v != v.strip():
            raise ValueError("symbol must be a non-empty stripped name")
        return v

    @model_validator(mode="after")
    def _commodity_sector(self) -> InstrumentSpec:
        if self.kind == "commodity" and not self.sector:
            raise ValueError(f"{self.symbol} commodity requires a sector")
        return self


class IndicesCfg(FrozenModel):
    market: str = "IDX:MARKET"
    bond: str = "IDX:BOND"
    sector_prefix: str = "IDX:"

    @field_validator("market", "bond", "sector_prefix")
    @classmethod
    def _name(cls, v: str) -> str:
        if not v:
            raise ValueError("index name must be non-empty")
        return v


class ImpactCfg(FrozenModel):
    """§6.4 power-law kernel. Weights are fitted in T6.07."""

    delta: float = 0.5
    beta: float = 0.5
    Y: float = 0.8
    tau0_d: float = 1.0  # days
    half_lives_d: tuple[float, ...] = (1.0, 5.0, 20.0, 60.0, 250.0)

    @field_validator("delta", "beta", "Y", "tau0_d")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("impact param", v)

    @field_validator("half_lives_d")
    @classmethod
    def _hl(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("half_lives_d must be non-empty")
        if any(h <= 0 for h in v):
            raise ValueError("half_lives_d must be > 0")
        return v


class MMCfg(FrozenModel):
    s0: float = 0.0005  # dimensionless half-spread floor
    k_sigma: float = 0.5
    k_inv: float = 0.2
    k_skew: float = 0.1
    participation_cap: float = 0.25  # fill budget / ADV / tick

    @field_validator("s0", "k_sigma", "k_inv", "k_skew")
    @classmethod
    def _nn(cls, v: float) -> float:
        return _nonneg("mm param", v)

    @field_validator("participation_cap")
    @classmethod
    def _part(cls, v: float) -> float:
        return _unit_interval("participation_cap", v)


class FlowCfg(FrozenModel):
    n_components: int = 3
    persistences: tuple[float, ...] = (0.50, 0.90, 0.99)
    mean_reversion: float = 0.10
    momentum: float = 0.10

    @field_validator("n_components")
    @classmethod
    def _n(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_components must be >= 1")
        return v

    @field_validator("persistences")
    @classmethod
    def _p(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        for x in v:
            _unit_interval("flow persistence", x)
        return v

    @field_validator("mean_reversion", "momentum")
    @classmethod
    def _nn(cls, v: float) -> float:
        return _nonneg("flow loading", v)

    @model_validator(mode="after")
    def _len(self) -> FlowCfg:
        if len(self.persistences) != self.n_components:
            raise ValueError("flow.persistences length must equal n_components")
        return self


class ClobCfg(FrozenModel):
    tick: float = 0.01
    lot: int = 1
    halt_band: float = 0.20  # ± daily
    thin_quote_w: float = 0.03  # V·(1 ± w)
    thin_quote_size: float = 0.002  # share of float

    @field_validator("tick", "thin_quote_w", "thin_quote_size")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("clob param", v)

    @field_validator("lot")
    @classmethod
    def _lot(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lot must be >= 1")
        return v

    @field_validator("halt_band")
    @classmethod
    def _halt(cls, v: float) -> float:
        return _unit_interval("halt_band", v)


class FeesCfg(FrozenModel):
    commission: float = 0.0  # share of notional
    transaction_tax: float = 0.0  # share of notional

    @field_validator("commission", "transaction_tax")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("fee", v)


class MarginCfg(FrozenModel):
    equity_initial: float = 0.50
    equity_maintenance: float = 0.25
    commodity_initial: float = 0.10
    commodity_maintenance: float = 0.07

    @field_validator(
        "equity_initial",
        "equity_maintenance",
        "commodity_initial",
        "commodity_maintenance",
    )
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("margin", v)

    @model_validator(mode="after")
    def _maint_below_init(self) -> MarginCfg:
        if self.equity_maintenance > self.equity_initial:
            raise ValueError("equity_maintenance must be <= equity_initial")
        if self.commodity_maintenance > self.commodity_initial:
            raise ValueError("commodity_maintenance must be <= commodity_initial")
        return self


class SurveillanceCfg(FrozenModel):
    wash_window_ticks: int = 1
    circular_n: int = 3
    pump_volume_share: float = 0.50
    pump_runup: float = 0.20
    pump_window_d: int = 5

    @field_validator("wash_window_ticks", "circular_n", "pump_window_d")
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("surveillance window must be >= 1")
        return v

    @field_validator("pump_volume_share", "pump_runup")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("surveillance share", v)


class MarketsFile(FrozenModel):
    """Root of `config/markets.yaml`."""

    turnover: float = 0.004  # 1/day; ADV = turnover × cap
    pricing: PricingCfg = Field(default_factory=PricingCfg)
    npc_equity: NpcEquityCfg = Field(default_factory=NpcEquityCfg)
    static: tuple[InstrumentSpec, ...] = ()
    indices: IndicesCfg = Field(default_factory=IndicesCfg)
    impact: ImpactCfg = Field(default_factory=ImpactCfg)
    mm: MMCfg = Field(default_factory=MMCfg)
    flow: FlowCfg = Field(default_factory=FlowCfg)
    clob: ClobCfg = Field(default_factory=ClobCfg)
    fees: FeesCfg = Field(default_factory=FeesCfg)
    margin: MarginCfg = Field(default_factory=MarginCfg)
    surveillance: SurveillanceCfg = Field(default_factory=SurveillanceCfg)

    @field_validator("turnover")
    @classmethod
    def _to(cls, v: float) -> float:
        return _positive("turnover", v)

    @model_validator(mode="after")
    def _unique_static(self) -> MarketsFile:
        seen: set[str] = set()
        for spec in self.static:
            if spec.symbol in seen:
                raise ValueError(f"duplicate static symbol {spec.symbol!r}")
            seen.add(spec.symbol)
        return self


def average_daily_volume(cap: float, turnover: float) -> float:
    """ADV (cr/day) = ``turnover`` (1/day) × ``cap`` (cr). §6.2."""
    return float(turnover) * float(cap)


def npc_symbol(code: str, prefix: str = NPC_PREFIX) -> str:
    """``EQ:NPC:<SECTOR>`` claim on the national NPC mass."""
    return f"{prefix}{code}"


def firm_symbol(firm_id: str) -> str:
    """``EQ:FIRM:<id>`` listed agent-firm share."""
    if not firm_id or ":" in firm_id or "/" in firm_id:
        raise ConfigError(f"invalid firm id for listing: {firm_id!r}")
    return f"{FIRM_PREFIX}{firm_id}"


def sector_index_symbol(code: str, prefix: str = "IDX:") -> str:
    return f"{prefix}{code}"


@dataclass(frozen=True)
class Instrument:
    """One listed or published instrument. ``cap`` lives outside; ADV uses the registry turnover."""

    symbol: str
    kind: str
    venue: str
    tradable: bool
    sector: str | None = None

    def to_state(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Instrument:
        return cls(
            symbol=str(state["symbol"]),
            kind=str(state["kind"]),
            venue=str(state["venue"]),
            tradable=bool(state["tradable"]),
            sector=None if state.get("sector") is None else str(state["sector"]),
        )


def _spec_to_instrument(spec: InstrumentSpec) -> Instrument:
    return Instrument(
        symbol=spec.symbol,
        kind=spec.kind,
        venue=spec.venue,
        tradable=spec.tradable,
        sector=spec.sector,
    )


class InstrumentRegistry:
    """Name → instrument. Iteration is sorted by symbol (deterministic)."""

    def __init__(
        self,
        instruments: Iterable[Instrument] | None = None,
        *,
        turnover: float = 0.004,
    ) -> None:
        if turnover <= 0:
            raise ConfigError("turnover must be > 0")
        self.turnover = float(turnover)
        self._by_symbol: dict[str, Instrument] = {}
        for inst in instruments or ():
            self._add(inst)

    def _add(self, inst: Instrument) -> None:
        if inst.symbol in self._by_symbol:
            raise ConfigError(f"duplicate instrument {inst.symbol!r}")
        if inst.kind not in KINDS:
            raise ConfigError(f"unknown kind {inst.kind!r}")
        if inst.venue not in VENUES:
            raise ConfigError(f"unknown venue {inst.venue!r}")
        self._by_symbol[inst.symbol] = inst

    @classmethod
    def from_markets(
        cls,
        markets: MarketsFile,
        codes: Sequence[str] | None = None,
    ) -> InstrumentRegistry:
        """Build the §6.2 book: EQ:NPC × |codes|, static names, published indices."""
        if codes is None:
            from marketsim.layer1.build_io import CODES as _CODES

            codes = _CODES
        codes = tuple(codes)
        code_set = set(codes)
        insts: list[Instrument] = []
        seen: set[str] = set()

        def _push(inst: Instrument) -> None:
            if inst.symbol in seen:
                raise ConfigError(f"duplicate instrument {inst.symbol!r}")
            seen.add(inst.symbol)
            insts.append(inst)

        for code in codes:
            _push(
                Instrument(
                    symbol=npc_symbol(code, markets.npc_equity.prefix),
                    kind="eq_npc",
                    venue=markets.npc_equity.venue,
                    tradable=True,
                    sector=code,
                )
            )
            _push(
                Instrument(
                    symbol=sector_index_symbol(code, markets.indices.sector_prefix),
                    kind="index",
                    venue="none",
                    tradable=False,
                    sector=code,
                )
            )
        for spec in markets.static:
            if spec.sector is not None and spec.sector not in code_set:
                raise ConfigError(f"instrument {spec.symbol} sector {spec.sector!r} not in codes")
            expected = COMMODITY_SECTORS.get(spec.symbol)
            if expected is not None and spec.sector != expected:
                raise ConfigError(f"{spec.symbol} must map to {expected}, got {spec.sector}")
            _push(_spec_to_instrument(spec))
        _push(Instrument(symbol=markets.indices.market, kind="index", venue="none", tradable=False))
        _push(Instrument(symbol=markets.indices.bond, kind="index", venue="none", tradable=False))
        return cls(insts, turnover=markets.turnover)

    def list_firm(self, firm_id: str) -> Instrument:
        """Dynamically list ``EQ:FIRM:<id>`` on the CLOB. Idempotent."""
        symbol = firm_symbol(firm_id)
        existing = self._by_symbol.get(symbol)
        if existing is not None:
            return existing
        inst = Instrument(symbol=symbol, kind="eq_firm", venue="clob", tradable=True, sector=None)
        self._add(inst)
        return inst

    def adv(self, cap: float, *, symbol: str | None = None) -> float:
        """ADV (cr/day) = turnover × cap. ``symbol`` is reserved for per-name overrides."""
        if symbol is not None and symbol not in self._by_symbol:
            raise ConfigError(f"unknown instrument {symbol!r}")
        return average_daily_volume(cap, self.turnover)

    def get(self, symbol: str) -> Instrument:
        try:
            return self._by_symbol[symbol]
        except KeyError as exc:
            raise ConfigError(f"unknown instrument {symbol!r}") from exc

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_symbol))

    def tradable_symbols(self) -> tuple[str, ...]:
        return tuple(s for s in self.symbols() if self._by_symbol[s].tradable)

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and symbol in self._by_symbol

    def __len__(self) -> int:
        return len(self._by_symbol)

    def __iter__(self) -> Iterator[Instrument]:
        for symbol in self.symbols():
            yield self._by_symbol[symbol]

    def to_state(self) -> dict[str, Any]:
        return {
            "turnover": self.turnover,
            "instruments": [self._by_symbol[s].to_state() for s in self.symbols()],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> InstrumentRegistry:
        if "turnover" not in state or "instruments" not in state:
            raise StateError("instrument registry state needs turnover and instruments")
        insts = [Instrument.from_state(row) for row in state["instruments"]]
        return cls(insts, turnover=float(state["turnover"]))
