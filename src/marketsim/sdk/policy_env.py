"""Gymnasium policymaker environment (T7.16 / §7.3, D13).

Observation is published macro, budget, debt and the government curve.
Actions are any subset of the §2.13 levers (masked levers stay on autopilot).
Reward is the quadratic policy loss
``−[(π − π*)² + λ_y·gap² + λ_b·(b − b*)²]`` (dimensionless).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marketsim.real.economy import make_real_world
from marketsim.real.policy.authority import CB_LEVERS, GOVT_LEVERS, PolicyDecision
from marketsim.world import World

AuthorityName = Literal["GOVT", "CENBANK"]

# §7.3 quadratic weights. Not calibration — named so they are not magic.
LAMBDA_Y = 1.0
LAMBDA_B = 0.25
MASK_AUTOPILOT = 0
MASK_AGENT = 1
_OBS_ABS = 1.0e3
# Scalar §2.13 levers the Box can represent (dicts/paths stay on autopilot).
_GOVT_SCALAR: tuple[str, ...] = (
    "purchases_level",
    "tau_y",
    "tau_c",
    "vat",
    "tariff",
    "benefit_replacement",
    "transfer_oneoff",
    "rescue_banksys",
    "debt_target",
    "kappa_debt",
    "fiscal_rule_on",
)
_CB_SCALAR: tuple[str, ...] = (
    "rate",
    "pi_star",
    "phi_pi",
    "phi_y",
    "smoothing",
    "capital_requirement",
    "ltv_cap",
    "lolr",
)


def policy_loss(
    pi: float,
    pi_star: float,
    gap: float,
    debt_gdp: float,
    b_star: float,
    *,
    lambda_y: float = LAMBDA_Y,
    lambda_b: float = LAMBDA_B,
) -> float:
    """§7.3 score. ``pi`` / ``pi_star`` are annual decimals; ``gap`` / debts are fractions."""
    return -((pi - pi_star) ** 2 + lambda_y * gap * gap + lambda_b * (debt_gdp - b_star) ** 2)


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class PolicyEnv(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Single-authority Gymnasium env over ``make_real_world``.

    ``authority`` binds GOVT or CENBANK (D13). Episode is ``horizon`` days.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_dir: str | Path,
        *,
        seed: int = 0,
        authority: AuthorityName = "GOVT",
        horizon: int = 8,
        lambda_y: float = LAMBDA_Y,
        lambda_b: float = LAMBDA_B,
    ) -> None:
        super().__init__()
        if horizon < 2:
            raise ValueError("horizon must be >= 2 (days)")
        if authority not in ("GOVT", "CENBANK"):
            raise ValueError("authority must be GOVT or CENBANK")
        self.config_dir = Path(config_dir)
        self._seed = int(seed)
        self.authority: AuthorityName = authority
        self.horizon = int(horizon)  # days
        self.lambda_y = float(lambda_y)
        self.lambda_b = float(lambda_b)
        self.levers: tuple[str, ...] = _GOVT_SCALAR if authority == "GOVT" else _CB_SCALAR
        self._world: World | None = None
        self.last_decision: PolicyDecision | None = None
        self._build_spaces()

    def _build_spaces(self) -> None:
        n = len(self.levers)
        abs_b = np.float32(_OBS_ABS)
        self.action_space = spaces.Dict(
            {
                **{name: spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32) for name in self.levers},
                "mask": spaces.MultiBinary(n),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "tick": spaces.Box(0.0, float(self.horizon), shape=(1,), dtype=np.float32),
                "pi": spaces.Box(-abs_b, abs_b, shape=(1,), dtype=np.float32),
                "gap": spaces.Box(-abs_b, abs_b, shape=(1,), dtype=np.float32),
                "debt_gdp": spaces.Box(-abs_b, abs_b, shape=(1,), dtype=np.float32),
                "pi_star": spaces.Box(-abs_b, abs_b, shape=(1,), dtype=np.float32),
                "curve": spaces.Box(-abs_b, abs_b, shape=(3,), dtype=np.float32),
            }
        )

    @property
    def world(self) -> World:
        if self._world is None:
            self._world = make_real_world(self.config_dir, seed=self._seed)
        return self._world

    def _economy(self) -> Any:
        for mod in self.world.modules:
            if getattr(mod, "name", None) == "real_economy":
                return mod
        raise RuntimeError("PolicyEnv requires a RealEconomy module")

    def _b_star(self) -> float:
        """Fiscal-rule debt target (fraction of annual GDP)."""
        eco = self._economy()
        merged = eco.policy.govt.merged()
        if merged.get("debt_target") is not None:
            return float(merged["debt_target"])
        dyn = self.world.cfg.dynamics
        if dyn is None:
            return 0.6  # fallback debt/GDP fraction when dynamics.yaml is absent
        return float(dyn.fiscal.debt_to_gdp)

    def _macro(self) -> dict[str, float]:
        eco = self._economy()
        agg = getattr(eco, "last_agg", None)
        cb = eco.cb
        y_fair = float(self.world.cfg.bonds.duration_ref_yield) if self.world.cfg.bonds else 0.042
        return {
            "tick": float(self.world.clock.tick),
            "pi": float(getattr(agg, "pi12", 0.0) or 0.0) if agg is not None else 0.0,
            "gap": float(getattr(agg, "gdp_prod_real", 0.0) / max(float(eco.fin.gdp0), 1e-12) - 1.0)
            if agg is not None
            else 0.0,
            "debt_gdp": float(getattr(agg, "debt_gdp", 0.0) or 0.0) if agg is not None else 0.0,
            "pi_star": float(cb.pi_star),
            "y_fair": y_fair,
        }

    def encode_observation(self, macro: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """Fixed-size encoder. See :meth:`_macro` for units."""
        row = dict(macro) if macro is not None else self._macro()
        y = float(row["y_fair"])
        return {
            "tick": np.asarray([min(row["tick"], float(self.horizon))], dtype=np.float32),
            "pi": np.asarray([row["pi"]], dtype=np.float32),
            "gap": np.asarray([row["gap"]], dtype=np.float32),
            "debt_gdp": np.asarray([row["debt_gdp"]], dtype=np.float32),
            "pi_star": np.asarray([row["pi_star"]], dtype=np.float32),
            "curve": np.asarray([y, y, y], dtype=np.float32),
        }

    def idle_action(self) -> dict[str, Any]:
        """Every lever masked (autopilot). Deterministic, no RNG."""
        act = {name: np.zeros(1, dtype=np.float32) for name in self.levers}
        act["mask"] = np.zeros(len(self.levers), dtype=np.int8)
        return act

    def decode(self, action: Mapping[str, Any]) -> PolicyDecision:
        """Map the ``Dict`` to a ``PolicyDecision``. Masked levers stay ``None``."""
        mask = _as_int_vec(action.get("mask"), len(self.levers))
        fields: dict[str, Any] = {"source": "agent"}
        eco = self._economy()
        specs = eco.policy.govt.specs if self.authority == "GOVT" else eco.policy.cenbank.specs
        for i, name in enumerate(self.levers):
            if int(mask[i]) == MASK_AUTOPILOT:
                continue
            unit = float(_as_f32_vec(action.get(name), 1)[0])
            spec = specs.get(name)
            if name == "fiscal_rule_on":
                fields[name] = unit > 0.0
                continue
            lo = 0.0 if spec is None or spec.lo is None else float(spec.lo)
            hi = 1.0 if spec is None or spec.hi is None else float(spec.hi)
            fields[name] = lo + 0.5 * (unit + 1.0) * (hi - lo)
        return PolicyDecision(**fields)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Rewind to tick 0 (days). ``seed`` is the ``World`` root seed."""
        super().reset(seed=seed)
        del options
        if seed is not None:
            self._seed = int(seed)
        self._world = make_real_world(self.config_dir, seed=self._seed)
        desk = self._economy().policy
        if self.authority == "GOVT":
            desk.govt.control = "agent"
        else:
            desk.cenbank.control = "agent"
        self.last_decision = None
        obs = self.encode_observation()
        return obs, self._info()

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one tick (day). Reward is the §7.3 loss (dimensionless)."""
        decision = self.decode(action)
        self.last_decision = decision
        eco = self._economy()
        auth = eco.policy.govt if self.authority == "GOVT" else eco.policy.cenbank
        month = int(self.world.clock.tick) // 21
        auth.submit(decision, month=month, lag_m=0, tick=int(self.world.clock.tick))
        self.world.step(1)
        macro = self._macro()
        reward = policy_loss(
            macro["pi"],
            macro["pi_star"],
            macro["gap"],
            macro["debt_gdp"],
            self._b_star(),
            lambda_y=self.lambda_y,
            lambda_b=self.lambda_b,
        )
        terminated = False
        truncated = int(self.world.clock.tick) >= self.horizon
        return self.encode_observation(macro), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        macro = self._macro()
        return {
            "tick": int(macro["tick"]),
            "pi": float(macro["pi"]),
            "gap": float(macro["gap"]),
            "debt_gdp": float(macro["debt_gdp"]),
        }

    def close(self) -> None:
        self._world = None


def lever_names(authority: AuthorityName) -> Sequence[str]:
    """Scalar lever names for ``authority`` (unitless ids)."""
    return _GOVT_SCALAR if authority == "GOVT" else _CB_SCALAR


assert set(_GOVT_SCALAR) <= set(GOVT_LEVERS) | {"fiscal_rule_on"}
assert set(_CB_SCALAR) <= set(CB_LEVERS)
