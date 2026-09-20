"""T8.07 — game-layer campaign client (pacing, slots, score, narrative).

This module is a client of the public v1 API (``create_app`` + ``HttpClient``).
It does not add engine hooks. Seeds go through ``WorldSpec`` / ``HttpClient``.
Money is cr; ticks are simulated days; NPC scale and event severity are
dimensionless.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketsim.api.lockstep import DEFAULT_TICKS_PER_SECOND
from marketsim.api.replay import load_snapshot, write_snapshot
from marketsim.api.rest import create_app
from marketsim.api.schemas import (
    AgentRegistration,
    FirmDecision,
    FoundFirmRequest,
    Observation,
    OrderRequest,
    WorldSpec,
)
from marketsim.api.schemas import (
    NewsItem as ApiNewsItem,
)
from marketsim.core.errors import ConfigError, StateError
from marketsim.equity.control import NEWS_CATEGORY
from marketsim.events.news import NewsItem
from marketsim.sdk.client import HttpClient
from marketsim.world import World

PacingMode = Literal["lockstep", "realtime"]
WorldModeName = Literal["game", "professional"]

# Short demo horizon (days). A playable campaign must not require a 50k-tick run.
DEMO_DURATION_TICKS = 3
# cr — same order as ``examples/quickstart_trader.py`` starting capital.
DEMO_FOUNDING_CAPITAL_CR = 100.0
# Dimensionless NPC-mass and event-severity knobs (difficulty factors).
DEMO_NPC_SCALE = 1.0
DEMO_EVENT_SEVERITY = 1.0
DEMO_SEED = 7
DEMO_ORDER_QTY = 10  # shares
DEMO_ORDER_PRICE_CR = 1.0  # cr / share
DEMO_VACANCIES = 2.0  # persons
# Catalog knobs (cr / dimensionless). Named so the catalog has no bare literals.
OPERATOR_FOUNDING_CAPITAL_CR = 50.0
OPERATOR_DURATION_TICKS = 4
OPERATOR_SEED = 11
STORM_FOUNDING_CAPITAL_CR = 80.0
STORM_NPC_SCALE = 1.2
STORM_EVENT_SEVERITY = 1.5
STORM_SEED = 13
DEFAULT_SYMBOL = "EQ:NPC:AUTOS"
SLOT_SUFFIX = ".campaign.json"
ENGINE_SLOT_SUFFIX = ".world.json"

# Narrative uses only published ``NewsItem`` fields (no magnitudes; T4.06).
NARRATIVE_TEMPLATE = (
    "{rumour}{headline} [{category} · day {tick} · hint {severity_hint}{where}]"
)
TAKEOVER_TEMPLATE = "Takeover: {headline} ({firm_id}; day {tick})"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CampaignSpec(FrozenModel):
    """One playable campaign. Difficulty = founding capital × NPC scale × event severity.

    Units: ``founding_capital`` cr; ``npc_scale`` and ``event_severity`` dimensionless;
    ``duration_ticks`` and seed-driven ticks are days; ``ticks_per_second`` is ticks/s.
    """

    id: str
    seed: int
    duration_ticks: int = DEMO_DURATION_TICKS
    run_mode: PacingMode = "lockstep"
    ticks_per_second: float = DEFAULT_TICKS_PER_SECOND
    founding_capital: float = DEMO_FOUNDING_CAPITAL_CR
    npc_scale: float = DEMO_NPC_SCALE
    event_severity: float = DEMO_EVENT_SEVERITY
    agent_id: str = "player"
    firm_id: str = "campaign_co"
    symbol: str = DEFAULT_SYMBOL
    order_qty: int = DEMO_ORDER_QTY
    order_price: float = DEMO_ORDER_PRICE_CR
    vacancies: float = DEMO_VACANCIES
    slot_dir: str | None = None
    mode: WorldModeName = "game"

    @field_validator("id", "agent_id", "firm_id", "symbol")
    @classmethod
    def _id(cls, v: str) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError("campaign id fields must be non-empty")
        return text

    @field_validator("seed")
    @classmethod
    def _seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be >= 0")
        return v

    @field_validator("duration_ticks")
    @classmethod
    def _days(cls, v: int) -> int:
        if v < 1:
            raise ValueError("duration_ticks must be >= 1 (days)")
        return v

    @field_validator("ticks_per_second")
    @classmethod
    def _rate(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("ticks_per_second must be > 0 (ticks/s)")
        return v

    @field_validator("founding_capital")
    @classmethod
    def _capital(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("founding_capital must be > 0 (cr)")
        return v

    @field_validator("npc_scale", "event_severity")
    @classmethod
    def _positive_scale(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("npc_scale and event_severity must be > 0 (dimensionless)")
        return v

    @field_validator("order_qty")
    @classmethod
    def _qty(cls, v: int) -> int:
        if v < 1:
            raise ValueError("order_qty must be >= 1 (shares)")
        return v

    @field_validator("order_price")
    @classmethod
    def _px(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("order_price must be >= 0 (cr / share)")
        return v

    def difficulty(self) -> float:
        """Composite difficulty (cr): ``founding_capital × npc_scale × event_severity``."""
        return float(self.founding_capital) * float(self.npc_scale) * float(self.event_severity)

    def world_spec(self) -> WorldSpec:
        """``POST /v1/worlds`` body. Seed is the RNG root; no yaml retune."""
        return WorldSpec(
            seed=self.seed,
            scenario=self.id,
            overrides={},
            mode=self.mode,
            run_mode=self.run_mode,
        )

    def agent_registration(self) -> AgentRegistration:
        """``POST …/agents``. ``starting_capital`` is founding capital (cr)."""
        return AgentRegistration(agent_id=self.agent_id, starting_capital=self.founding_capital)


class Scorecard(FrozenModel):
    """Published score. ``net_worth`` and ``valuation`` are cr; ``market_share`` is dimensionless."""

    net_worth: float = 0.0
    valuation: float = 0.0
    market_share: float = 0.0


class CampaignSlot(FrozenModel):
    """Game-layer save slot. Tick is days; ``state_hash`` is a hex digest; money in the score is cr."""

    spec: CampaignSpec
    world_id: str
    seed: int
    tick: int
    state_hash: str
    score: Scorecard
    agent_id: str
    firm_id: str
    notifications: tuple[str, ...] = Field(default_factory=tuple)
    takeovers: tuple[str, ...] = Field(default_factory=tuple)
    finished: bool = False


@dataclass
class PacingControls:
    """Pause / speed. ``ticks_per_second`` is ticks/s; lockstep ignores wall-clock.

    The core is never sampled: ``due_ticks(elapsed_s)`` takes elapsed seconds from the caller.
    """

    mode: PacingMode = "lockstep"
    ticks_per_second: float = DEFAULT_TICKS_PER_SECOND
    paused: bool = False

    def pause(self) -> None:
        """Stop advancing. Subsequent ``due_ticks`` return 0 until ``resume``."""
        self.paused = True

    def resume(self) -> None:
        """Clear the pause flag. Does not step the world."""
        self.paused = False

    def set_speed(self, ticks_per_second: float) -> None:
        """Set realtime speed (ticks/s). ``ticks_per_second`` must be > 0."""
        rate = float(ticks_per_second)
        if rate <= 0.0:
            raise ConfigError("ticks_per_second must be > 0 (ticks/s)")
        self.ticks_per_second = rate

    def interval_s(self) -> float:
        """Wall-clock period (s) at ``ticks_per_second``. Unused by the engine."""
        return 1.0 / float(self.ticks_per_second)

    def due_ticks(self, elapsed_s: float) -> int:
        """Ticks due after ``elapsed_s`` seconds (s). 0 while paused. Caller supplies the clock."""
        if self.paused:
            return 0
        return int(float(elapsed_s) * float(self.ticks_per_second))


@dataclass(frozen=True)
class CampaignResult:
    """Finished (or mid-run) campaign. Tick is days; difficulty and cash scores are cr."""

    campaign_id: str
    world_id: str
    agent_id: str
    firm_id: str
    tick: int
    seed: int
    state_hash: str
    difficulty: float
    score: Scorecard
    narrative: tuple[str, ...]
    takeovers: tuple[str, ...]
    slot_path: str | None
    finished: bool
    ticks_played: int


@dataclass
class Campaign:
    """Live campaign bound to one ``HttpClient``. All world I/O is REST."""

    client: HttpClient
    spec: CampaignSpec
    pacing: PacingControls = field(init=False)
    notifications: list[str] = field(default_factory=list)
    takeovers: list[str] = field(default_factory=list)
    scores: list[Scorecard] = field(default_factory=list)
    finished: bool = False
    world_id: str | None = None
    account: str | None = None
    last_hash: str | None = None
    ticks_played: int = 0

    def __post_init__(self) -> None:
        self.pacing = PacingControls(
            mode=self.spec.run_mode,
            ticks_per_second=self.spec.ticks_per_second,
        )

    def start(self) -> None:
        """Create the world, register the player, found the campaign firm (cr)."""
        status = self.client.create_world(self.spec.world_spec())
        self.world_id = status.world_id
        self.last_hash = status.state_hash
        acct = self.client.register_agent(self.spec.agent_registration())
        self.account = acct.account
        self.client.bind(token=acct.token)
        _found_firm(self.client, self.spec)

    def observe_and_score(self) -> tuple[Observation, Scorecard]:
        """GET …/observe and score published net worth / valuation / share."""
        obs = self.client.observe()
        score = score_observation(
            obs,
            firm_id=self.spec.firm_id,
            unpublished_cash_cr=self.spec.founding_capital,
        )
        self.scores.append(score)
        for item in obs.news:
            line = narrate(item)
            self.notifications.append(line)
            note = takeover_notification(item)
            if note is not None:
                self.takeovers.append(note)
        return obs, score

    def act_scripted(self) -> None:
        """One scripted operator decision plus one limit buy (shares, cr / share)."""
        decision = FirmDecision(
            firm_id=self.spec.firm_id,
            operator=self.spec.agent_id,
            vacancies=self.spec.vacancies,
        )
        self.client.submit_decisions(self.spec.agent_id, decision)
        order = OrderRequest(
            agent_id=self.spec.agent_id,
            side="buy",
            qty=self.spec.order_qty,
            symbol=self.spec.symbol,
            price=self.spec.order_price,
        )
        self.client.submit_orders(self.spec.agent_id, order)

    def step_days(self, n: int = 1) -> None:
        """POST …/step. ``n`` is days. Refuses while paused."""
        if self.pacing.paused:
            raise StateError("campaign is paused")
        status = self.client.step(n)
        self.last_hash = status.state_hash
        self.ticks_played += int(n)

    def save_slot(self, name: str = "autosave") -> Path:
        """Write the game overlay slot (JSON). ``name`` is a unitless slot id."""
        dest = campaign_slot_path(self._slot_dir(), name)
        status = self.client.status()
        score = self.scores[-1] if self.scores else Scorecard()
        slot = CampaignSlot(
            spec=self.spec,
            world_id=status.world_id,
            seed=status.seed,
            tick=status.tick,
            state_hash=status.state_hash,
            score=score,
            agent_id=self.spec.agent_id,
            firm_id=self.spec.firm_id,
            notifications=tuple(self.notifications),
            takeovers=tuple(self.takeovers),
            finished=self.finished,
        )
        write_campaign_slot(dest, slot)
        return dest

    def load_slot(self, name: str = "autosave") -> CampaignSlot:
        """Read a game overlay slot. Does not replace the live world (T7.10 load does)."""
        return read_campaign_slot(campaign_slot_path(self._slot_dir(), name))

    def finish(self) -> CampaignResult:
        """Mark the campaign complete and return the public scorecard."""
        self.finished = True
        status = self.client.status()
        self.last_hash = status.state_hash
        slot_path: str | None = None
        if self.spec.slot_dir:
            slot_path = str(self.save_slot("final"))
        score = self.scores[-1] if self.scores else Scorecard()
        return CampaignResult(
            campaign_id=self.spec.id,
            world_id=status.world_id,
            agent_id=self.spec.agent_id,
            firm_id=self.spec.firm_id,
            tick=status.tick,
            seed=status.seed,
            state_hash=status.state_hash,
            difficulty=self.spec.difficulty(),
            score=score,
            narrative=tuple(self.notifications),
            takeovers=tuple(self.takeovers),
            slot_path=slot_path,
            finished=True,
            ticks_played=self.ticks_played,
        )

    def _slot_dir(self) -> Path:
        if not self.spec.slot_dir:
            raise ConfigError("slot_dir is required to save or load a campaign slot")
        return Path(self.spec.slot_dir)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _text_list(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(part) for part in value if str(part))


def is_takeover_news(item: NewsItem | ApiNewsItem | Mapping[str, Any]) -> bool:
    """True for T6.16 control-transfer headlines (``corporate`` / ``control:``)."""
    category = str(_field(item, "category", ""))
    nid = str(_field(item, "id", ""))
    headline = str(_field(item, "headline", ""))
    if category != NEWS_CATEGORY:
        return False
    return nid.startswith("control:") or headline.startswith("Operating control of")


def takeover_firm_id(item: NewsItem | ApiNewsItem | Mapping[str, Any]) -> str:
    """Firm id from ``control:<firm_id>:<tick>``, or empty. Unitless."""
    nid = str(_field(item, "id", ""))
    if nid.startswith("control:"):
        rest = nid[len("control:") :]
        if ":" in rest:
            return rest.rsplit(":", 1)[0]
        return rest
    return ""


def narrate(item: NewsItem | ApiNewsItem | Mapping[str, Any]) -> str:
    """Player-facing line from published ``NewsItem`` fields only. Tick is days."""
    regions = _text_list(_field(item, "regions", ()))
    sectors = _text_list(_field(item, "sectors", ()))
    where_parts: list[str] = []
    if regions:
        where_parts.append(" · " + ", ".join(regions))
    if sectors:
        where_parts.append(" · " + ", ".join(sectors))
    rumour = "Rumour — " if bool(_field(item, "is_rumour", False)) else ""
    return NARRATIVE_TEMPLATE.format(
        rumour=rumour,
        headline=str(_field(item, "headline", "")),
        category=str(_field(item, "category", "")),
        tick=int(_field(item, "tick", 0) or 0),
        severity_hint=int(_field(item, "severity_hint", 0) or 0),
        where="".join(where_parts),
    )


def takeover_notification(item: NewsItem | ApiNewsItem | Mapping[str, Any]) -> str | None:
    """Takeover banner, or None when the item is not a control transfer."""
    if not is_takeover_news(item):
        return None
    firm_id = takeover_firm_id(item) or str(_field(item, "id", ""))
    return TAKEOVER_TEMPLATE.format(
        headline=str(_field(item, "headline", "")),
        firm_id=firm_id,
        tick=int(_field(item, "tick", 0) or 0),
    )


def sample_takeover_news(*, firm_id: str, operator: str, tick: int) -> NewsItem:
    """``NewsItem`` matching ``ControlDesk.apply_month_boundary`` (no magnitude).

    ``tick`` is days; ``firm_id`` / ``operator`` are unitless ids.
    """
    return NewsItem(
        id=f"control:{firm_id}:{int(tick)}",
        tick=int(tick),
        category=NEWS_CATEGORY,
        headline=f"Operating control of {firm_id} passed to {operator}",
        regions=(),
        sectors=(),
        severity_hint=0,
        is_rumour=False,
    )


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _published_market_share(obs: Observation, firm_id: str) -> float:
    """Published share (dimensionless). 0 if unpublished."""
    goods = obs.goods
    raw_goods = goods.model_dump() if hasattr(goods, "model_dump") else _mapping(goods)
    for key in ("market_share", "share"):
        value = raw_goods.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for report in obs.reports:
        if firm_id and report.firm_id != firm_id:
            continue
        books = report.books or {}
        for key in ("market_share", "share"):
            if key in books and isinstance(books[key], (int, float)):
                return float(books[key])
    return 0.0


def _valuation_cr(obs: Observation, firm_id: str) -> float:
    """Own-firm published valuation (cr). 0 if unpublished."""
    for report in obs.reports:
        if firm_id and report.firm_id != firm_id:
            continue
        books = report.books or {}
        for key in ("equity", "equity_value", "book_equity", "valuation"):
            if key in books and isinstance(books[key], (int, float)):
                return float(books[key] or 0.0)
    return 0.0


def score_observation(
    obs: Observation,
    *,
    firm_id: str = "",
    unpublished_cash_cr: float = 0.0,
) -> Scorecard:
    """Score from published observe fields. Cash / net worth / valuation are cr."""
    port = obs.portfolio
    net_worth = float(port.net_worth)
    cash = float(port.cash)
    if net_worth == 0.0 and cash == 0.0 and not port.positions:
        net_worth = float(unpublished_cash_cr)
    return Scorecard(
        net_worth=net_worth,
        valuation=_valuation_cr(obs, firm_id),
        market_share=_published_market_share(obs, firm_id),
    )


def campaign_slot_path(slot_dir: str | Path, name: str) -> Path:
    """Filesystem path for a game overlay slot. ``name`` is a unitless id."""
    safe = name.strip()
    if not safe or "/" in safe or "\\" in safe:
        raise ConfigError("slot name must be a single path segment")
    return Path(slot_dir) / f"{safe}{SLOT_SUFFIX}"


def engine_slot_path(slot_dir: str | Path, name: str) -> Path:
    """Filesystem path for a T7.10 World snapshot beside the overlay slot."""
    safe = name.strip()
    if not safe or "/" in safe or "\\" in safe:
        raise ConfigError("slot name must be a single path segment")
    return Path(slot_dir) / f"{safe}{ENGINE_SLOT_SUFFIX}"


def write_campaign_slot(path: str | Path, slot: CampaignSlot) -> Path:
    """Write overlay JSON. ``path`` is a filesystem path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(slot.model_dump(mode="json"), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


def read_campaign_slot(path: str | Path) -> CampaignSlot:
    """Load overlay JSON. ``path`` is a filesystem path."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("campaign slot must be a JSON object")
    return CampaignSlot.model_validate(raw)


def save_engine_slot(path: str | Path, world: World) -> Path:
    """Write a T7.10 ``World`` snapshot (``write_snapshot``). ``path`` is a filesystem path."""
    return write_snapshot(path, world)


def load_engine_slot(
    path: str | Path,
    config_dir: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> World:
    """Restore a T7.10 snapshot (``load_snapshot``). ``config_dir`` is a filesystem path."""
    return load_snapshot(path, config_dir, overrides=overrides)


def demo_campaign(*, slot_dir: str | Path | None = None) -> CampaignSpec:
    """Built-in short demo (a handful of days). ``slot_dir`` is a filesystem path."""
    return CampaignSpec(
        id="demo",
        seed=DEMO_SEED,
        duration_ticks=DEMO_DURATION_TICKS,
        founding_capital=DEMO_FOUNDING_CAPITAL_CR,
        npc_scale=DEMO_NPC_SCALE,
        event_severity=DEMO_EVENT_SEVERITY,
        slot_dir=None if slot_dir is None else str(slot_dir),
    )


def campaign_scenarios(*, slot_dir: str | Path | None = None) -> dict[str, CampaignSpec]:
    """Built-in campaign catalog. Keys are unitless ids; slot dir is a filesystem path."""
    dest = None if slot_dir is None else str(slot_dir)
    return {
        "demo": demo_campaign(slot_dir=dest),
        "operator": CampaignSpec(
            id="operator",
            seed=OPERATOR_SEED,
            duration_ticks=OPERATOR_DURATION_TICKS,
            founding_capital=OPERATOR_FOUNDING_CAPITAL_CR,
            npc_scale=DEMO_NPC_SCALE,
            event_severity=DEMO_EVENT_SEVERITY,
            slot_dir=dest,
            vacancies=DEMO_VACANCIES,
        ),
        "storm_watch": CampaignSpec(
            id="storm_watch",
            seed=STORM_SEED,
            duration_ticks=DEMO_DURATION_TICKS,
            founding_capital=STORM_FOUNDING_CAPITAL_CR,
            npc_scale=STORM_NPC_SCALE,
            event_severity=STORM_EVENT_SEVERITY,
            slot_dir=dest,
        ),
    }


def _found_firm(client: HttpClient, spec: CampaignSpec) -> FoundFirmRequest:
    """POST …/firms. ``capital`` is founding capital (cr)."""
    resp = client._http.post(
        client._world_url("/firms"),
        json=FoundFirmRequest(
            firm_id=spec.firm_id,
            operator=spec.agent_id,
            capital=spec.founding_capital,
        ).model_dump(mode="json"),
        headers=client._headers(),
    )
    resp.raise_for_status()
    return FoundFirmRequest.model_validate(resp.json())


def play_scripted_campaign(client: HttpClient, spec: CampaignSpec) -> CampaignResult:
    """Drive one scripted player through REST on an already-bound client."""
    campaign = Campaign(client=client, spec=spec)
    campaign.start()
    campaign.pacing.pause()
    try:
        campaign.step_days(1)
    except StateError:
        pass
    else:
        raise StateError("paused campaign must not step")
    campaign.pacing.resume()
    campaign.pacing.set_speed(2.0)
    # Observe + slot before submit: GET … status hashes the inbox, which still
    # holds typed FirmDecision / OrderRequest objects until PUBLISH.
    campaign.observe_and_score()
    mid_slot: Path | None = None
    if spec.slot_dir:
        mid_slot = campaign.save_slot("mid")
        loaded = campaign.load_slot("mid")
        if loaded.world_id != campaign.world_id:
            raise StateError("loaded slot world_id does not match live campaign")
    campaign.act_scripted()
    for _day in range(int(spec.duration_ticks)):
        campaign.step_days(1)
        campaign.observe_and_score()
    result = campaign.finish()
    if mid_slot is not None and not mid_slot.is_file():
        raise StateError("mid-campaign slot was not written")
    return result


def run_scripted_player(
    config_dir: str | Path,
    *,
    spec: CampaignSpec | None = None,
    slot_dir: str | Path | None = None,
) -> CampaignResult:
    """Finish a short campaign through ``create_app`` + ``HttpClient`` only.

    ``config_dir`` and ``slot_dir`` are filesystem paths. Seed is ``spec.seed``.
    """
    chosen = spec or demo_campaign(slot_dir=slot_dir)
    if slot_dir is not None and chosen.slot_dir != str(slot_dir):
        chosen = chosen.model_copy(update={"slot_dir": str(slot_dir)})
    app = create_app(config_dir=config_dir)
    client = HttpClient(app=app)
    try:
        return play_scripted_campaign(client, chosen)
    finally:
        client.close()


__all__ = [
    "Campaign",
    "CampaignResult",
    "CampaignSlot",
    "CampaignSpec",
    "PacingControls",
    "Scorecard",
    "campaign_scenarios",
    "campaign_slot_path",
    "demo_campaign",
    "engine_slot_path",
    "is_takeover_news",
    "load_engine_slot",
    "narrate",
    "play_scripted_campaign",
    "read_campaign_slot",
    "run_scripted_player",
    "sample_takeover_news",
    "save_engine_slot",
    "score_observation",
    "takeover_notification",
    "write_campaign_slot",
]
