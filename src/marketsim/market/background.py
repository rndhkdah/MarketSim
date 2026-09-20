"""Background NPC order flow (T6.09 / §6.5).

Signed volume is a sum of AR(1)s (long-memory sign autocorrelation) with
intensity proportional to ADV and loadings on ``ξ`` (mean-reversion) and
``Δξ`` (momentum). On engine-MM venues this is NPC-on-NPC: it feeds the
impact kernel and posts nothing on the ledger.

Innovations come from ``core.rng.stream("flow")`` as a ``(n, K)`` draw
(instrument axis in constructor order). Per-name CLOB streams are T6.11.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from marketsim.core.errors import StateError
from marketsim.core.rng import RngHub
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FlowCfg

# Isolated from mispricing ``noise`` / ``sentiment`` (T6.06).
STREAM_FLOW = "flow"
# E[|N(0,1)|] = √(2/π); scale so E[|q|] = ADV when ξ terms are off.
GAUSS_L1 = float(np.sqrt(np.pi / 2.0))


def sign_acf(signs: np.ndarray, lag: int) -> float:
    """Lag-``k`` autocorrelation of a ±1 series (dimensionless)."""
    x = np.asarray(signs, dtype=float).reshape(-1)
    k = int(lag)
    if k < 1 or k >= x.size:
        raise ValueError("lag must be in 1 .. n-1")
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 0.0:
        return 0.0
    return float(np.dot(x[:-k], x[k:]) / den)


def feed_impact(
    kernel: ImpactKernel,
    q: float | np.ndarray,
    adv: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Apply signed NPC volume to the impact kernel. No ledger posting.

    ``q`` and ``adv`` are cr (cr/day); ``sigma`` is daily vol (decimal).
    Returns post-update ``ξ_impact`` (log-price, dimensionless).
    """
    return kernel.step(q, adv, sigma)


class BackgroundFlow:
    """Vectorised NPC flow for ``n`` names.

    ``adv`` is cr/day, shape ``(n,)``. ``xi`` / returned ``q`` are
    dimensionless log-mispricing and cr respectively.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        adv: np.ndarray,
        cfg: FlowCfg | None = None,
        *,
        stream: str = STREAM_FLOW,
    ) -> None:
        names = tuple(str(s) for s in symbols)
        if not names:
            raise ValueError("symbols must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("symbols must be unique")
        self.symbols = names
        self.cfg = cfg if cfg is not None else FlowCfg()
        self.stream = str(stream)
        self.phi = np.asarray(self.cfg.persistences, dtype=float)
        self.k = int(self.cfg.n_components)
        if self.phi.shape != (self.k,):
            raise ValueError("persistences must have length n_components")
        self.adv = np.asarray(adv, dtype=float).reshape(-1)
        if self.adv.shape != (len(names),):
            raise ValueError(f"adv must have shape {(len(names),)}")
        if np.any(self.adv < 0.0):
            raise ValueError("ADV must be >= 0 (cr/day)")
        # Stationary std of Σ_k x_k with unit-variance innovations.
        var = float(np.sum(1.0 / np.maximum(1.0 - self.phi**2, 1e-12)))
        self.scale = float(np.sqrt(var))
        self.x = np.zeros((len(names), self.k), dtype=float)
        self.prev_xi = np.zeros(len(names), dtype=float)
        self._have_prev = False

    @property
    def n_names(self) -> int:
        return len(self.symbols)

    def step(self, rng: RngHub, xi: np.ndarray) -> np.ndarray:
        """One tick of signed volume ``q`` (cr), shape ``(n,)``.

        ``xi`` is current log-mispricing (dimensionless), shape ``(n,)``.
        """
        z = np.asarray(xi, dtype=float).reshape(-1)
        if z.shape != (self.n_names,):
            raise ValueError(f"xi must have shape {(self.n_names,)}")
        eps = rng.stream(self.stream).standard_normal((self.n_names, self.k))
        self.x = self.phi * self.x + eps
        latent = self.x.sum(axis=1) / self.scale
        dxi = z - self.prev_xi if self._have_prev else np.zeros(self.n_names)
        drive = (
            latent
            - float(self.cfg.mean_reversion) * z
            + float(self.cfg.momentum) * dxi
        )
        q = self.adv * GAUSS_L1 * drive
        self.prev_xi = z.copy()
        self._have_prev = True
        return q

    def to_state(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "adv": self.adv.tolist(),
            "cfg": {
                "n_components": self.cfg.n_components,
                "persistences": list(self.cfg.persistences),
                "mean_reversion": self.cfg.mean_reversion,
                "momentum": self.cfg.momentum,
            },
            "stream": self.stream,
            "x": self.x.tolist(),
            "prev_xi": self.prev_xi.tolist(),
            "have_prev": self._have_prev,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BackgroundFlow:
        needed = ("symbols", "adv", "x")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"background flow state missing {missing}")
        raw = dict(state.get("cfg") or {})
        if "persistences" in raw:
            raw["persistences"] = tuple(raw["persistences"])
        obj = cls(
            tuple(state["symbols"]),
            np.asarray(state["adv"], dtype=float),
            FlowCfg(**raw) if raw else FlowCfg(),
            stream=str(state.get("stream", STREAM_FLOW)),
        )
        x = np.asarray(state["x"], dtype=float)
        if x.shape != obj.x.shape:
            raise StateError(f"background x must have shape {obj.x.shape}")
        obj.x = x.copy()
        obj.prev_xi = np.asarray(state.get("prev_xi", np.zeros(obj.n_names)), dtype=float)
        obj._have_prev = bool(state.get("have_prev", False))
        return obj
