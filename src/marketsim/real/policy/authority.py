"""Policy-authority framework: levers, arbitration, lags, news (D13 / §2.13)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

SOURCE_RANK = {"autopilot": 0, "event": 1, "scripted": 2, "agent": 3}
Source = Literal["autopilot", "event", "scripted", "agent"]

GOVT_LEVERS = (
    "purchases_level",
    "purchases_mix",
    "tau_y",
    "tau_c",
    "vat",
    "tariff",
    "excise",
    "benefit_replacement",
    "transfer_oneoff",
    "capex_subsidy",
    "rescue_banksys",
    "debt_target",
    "kappa_debt",
    "fiscal_rule_on",
)
CB_LEVERS = (
    "rate",
    "pi_star",
    "phi_pi",
    "phi_y",
    "smoothing",
    "capital_requirement",
    "ltv_cap",
    "lolr",
    "guidance_path",
)


@dataclass
class NewsItem:
    """Published when a decision is announced. Units depend on the payload."""

    tick: int
    authority: str
    headline: str
    payload: dict[str, Any]


@dataclass
class ClipReport:
    lever: str
    requested: Any
    applied: Any
    reason: str


@dataclass
class LeverSpec:
    name: str
    lo: float | None = None
    hi: float | None = None
    max_abs: float | None = None


@dataclass
class PolicyDecision:
    """Every economic field is optional; omitted levers stay on autopilot."""

    source: Source = "agent"
    purchases_level: float | None = None
    purchases_mix: dict[str, float] | None = None
    tau_y: float | None = None
    tau_c: float | None = None
    vat: float | None = None
    tariff: float | None = None
    excise: dict[str, float] | None = None
    benefit_replacement: float | None = None
    transfer_oneoff: float | None = None
    capex_subsidy: dict[str, float] | None = None
    rescue_banksys: float | None = None
    debt_target: float | None = None
    kappa_debt: float | None = None
    fiscal_rule_on: bool | None = None
    rate: float | None = None
    pi_star: float | None = None
    phi_pi: float | None = None
    phi_y: float | None = None
    smoothing: float | None = None
    capital_requirement: float | None = None
    ltv_cap: float | None = None
    lolr: float | None = None
    guidance_path: list[float] | None = None

    def set_fields(self) -> dict[str, Any]:
        skip = {"source"}
        return {k: v for k, v in asdict(self).items() if k not in skip and v is not None}


def _clip_value(name: str, value: Any, spec: LeverSpec | None) -> tuple[Any, ClipReport | None]:
    if spec is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return value, None
    applied = float(value)
    reason = None
    if spec.max_abs is not None and abs(applied) > spec.max_abs:
        applied = spec.max_abs if applied > 0 else -spec.max_abs
        reason = "max_abs"
    if spec.lo is not None and applied < spec.lo:
        applied = spec.lo
        reason = "lo"
    if spec.hi is not None and applied > spec.hi:
        applied = spec.hi
        reason = "hi"
    if reason is None:
        return applied, None
    return applied, ClipReport(name, float(value), applied, reason)


class PolicyAuthority:
    """One authority (GOVT or CENBANK). Arbitration: agent > scripted > event > autopilot."""

    def __init__(self, name: str, control: str, specs: dict[str, LeverSpec]) -> None:
        self.name = name
        self.control = control
        self.specs = specs
        self.effective: dict[str, Any] = {}
        self.source_of: dict[str, str] = {}
        self.pending: list[dict[str, Any]] = []
        self.news: list[dict[str, Any]] = []

    @classmethod
    def government(cls, tax_bounds: tuple[float, float] = (0.0, 0.6), tariff_max: float = 0.5) -> PolicyAuthority:
        lo, hi = tax_bounds
        specs = {
            "tau_y": LeverSpec("tau_y", lo, hi),
            "tau_c": LeverSpec("tau_c", lo, hi),
            "vat": LeverSpec("vat", lo, hi),
            "tariff": LeverSpec("tariff", 0.0, tariff_max),
            "benefit_replacement": LeverSpec("benefit_replacement", 0.0, 1.0),
            "purchases_level": LeverSpec("purchases_level", 0.0, None),
            "debt_target": LeverSpec("debt_target", 0.0, 2.0),
            "kappa_debt": LeverSpec("kappa_debt", 0.0, 1.0),
            "rescue_banksys": LeverSpec("rescue_banksys", 0.0, None),
            "transfer_oneoff": LeverSpec("transfer_oneoff", 0.0, None),
        }
        return cls("GOVT", "autopilot", specs)

    @classmethod
    def cenbank(cls, rate_move: float = 0.02) -> PolicyAuthority:
        specs = {
            "rate": LeverSpec("rate", 0.0, 0.20, max_abs=None),
            "pi_star": LeverSpec("pi_star", 0.0, 0.10),
            "phi_pi": LeverSpec("phi_pi", 0.0, 5.0),
            "phi_y": LeverSpec("phi_y", 0.0, 5.0),
            "smoothing": LeverSpec("smoothing", 0.0, 1.0),
            "capital_requirement": LeverSpec("capital_requirement", 0.05, 0.25),
            "ltv_cap": LeverSpec("ltv_cap", 0.0, 1.0),
            "lolr": LeverSpec("lolr", 0.0, None),
        }
        del rate_move
        return cls("CENBANK", "autopilot", specs)

    def submit(self, decision: PolicyDecision, month: int, lag_m: int, tick: int = 0) -> list[ClipReport]:
        fields = decision.set_fields()
        reports: list[ClipReport] = []
        clipped: dict[str, Any] = {}
        for name, value in fields.items():
            applied, rep = _clip_value(name, value, self.specs.get(name))
            clipped[name] = applied
            if rep is not None:
                reports.append(rep)
        payload = {**asdict(decision), **clipped, "source": decision.source}
        self.pending.append({"month": int(month + lag_m), "decision": payload, "source": decision.source})
        item = NewsItem(tick, self.name, f"{self.name} decision", clipped)
        self.news.append(asdict(item))
        if lag_m <= 0:
            self.activate_due(month)
        return reports

    def activate_due(self, month: int) -> list[dict[str, Any]]:
        due, keep = [], []
        for item in self.pending:
            if item["month"] <= month:
                due.append(item)
            else:
                keep.append(item)
        self.pending = keep
        for item in due:
            src = item["source"]
            for name, value in item["decision"].items():
                if name == "source" or value is None:
                    continue
                prev = self.source_of.get(name, "autopilot")
                if SOURCE_RANK.get(src, 0) >= SOURCE_RANK.get(prev, 0):
                    self.effective[name] = value
                    self.source_of[name] = src
        return due

    def merged(self) -> dict[str, Any]:
        """Levers currently off autopilot."""
        return dict(self.effective)

    def to_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "control": self.control,
            "effective": dict(self.effective),
            "source_of": dict(self.source_of),
            "pending": list(self.pending),
            "news": list(self.news),
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.name = state["name"]
        self.control = state["control"]
        self.effective = dict(state["effective"])
        self.source_of = dict(state["source_of"])
        self.pending = list(state["pending"])
        self.news = list(state["news"])


class PolicyDesk:
    """GOVT + CENBANK pair attached to a RealEconomy."""

    def __init__(self, govt: PolicyAuthority, cenbank: PolicyAuthority) -> None:
        self.govt = govt
        self.cenbank = cenbank

    @classmethod
    def default(cls) -> PolicyDesk:
        return cls(PolicyAuthority.government(), PolicyAuthority.cenbank())

    @classmethod
    def from_config(cls, cfg: Any) -> PolicyDesk:
        pol = getattr(cfg, "policy", None)
        if pol is None:
            return cls.default()
        govt = PolicyAuthority.government(
            tax_bounds=tuple(pol.govt.limits.tax_rate),
            tariff_max=pol.govt.limits.tariff,
        )
        govt.control = pol.govt.control
        cb = PolicyAuthority.cenbank(rate_move=pol.cenbank.limits.rate_move)
        cb.control = pol.cenbank.control
        return cls(govt, cb)

    def tick_month(self, month: int) -> None:
        self.govt.activate_due(month)
        self.cenbank.activate_due(month)

    def to_state(self) -> dict[str, Any]:
        return {"govt": self.govt.to_state(), "cenbank": self.cenbank.to_state()}

    def from_state(self, state: dict[str, Any]) -> None:
        self.govt.from_state(state["govt"])
        self.cenbank.from_state(state["cenbank"])
