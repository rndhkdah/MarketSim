"""Mispricing ``ξ_j = I_j + s_j + n_j`` with limits to arbitrage (T6.06 / §6.3).

``I`` is the impact state (accepted as an array; the §6.4 kernel is out of scope).
Sentiment is the specified AR; ``n`` is OU noise with ``σ ∝ idio_vol_j``, scaled
so calm-regime cap-weighted index volatility is 15–18 % a year. Arbitrage
capital ``A`` sets the OU speed and loses when ``|ξ|`` widens against it.
Randomness is only via ``core.rng.stream`` on the named streams ``flow``,
``sentiment`` and ``noise``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marketsim.core.errors import StateError
from marketsim.core.rng import RngHub

# §6.3 daily sentiment persistence ``φ_s`` in ``s ← φ_s·s + …``.
PHI_S = 0.98
# §6.3 loading on ``r̄_20d − r̄_250d`` (log-return gap, 1/day → log ξ).
KAPPA_MOM = 0.15
# §6.3 loading on the news input (news units → log ξ).
KAPPA_NEWS = 1.0
# §6.3 common risk-appetite loading on ``z_risk`` (annual decimal → log ξ).
KAPPA_RISK = 1.0
# §6.3 baseline daily OU mean-reversion ``θ_0`` (1/day) at ``A = A_0``.
THETA_0 = 0.04
# §6.3 baseline arbitrageur capital (normalised units).
A_0 = 1.0
# §6.3 ``clip(A_t/A_0, 0.2, 2)``.
A_RATIO_MIN = 0.2
A_RATIO_MAX = 2.0
# Master plan §6: 252 ticks / year; used to annualise calm-regime index vol.
DAYS_PER_YEAR = 252
# §6.3 / rule 15: daily leak of ``A`` toward ``A_0``.
A_LEAK = 1.0 / DAYS_PER_YEAR
# §6.3 loading of short-ξ P&L onto ``A`` (dimensionless).
KAPPA_A = 4.0
# §6.3 trailing windows for ``r̄_20d`` and ``r̄_250d``.
WINDOW_FAST_D = 20
WINDOW_SLOW_D = 250
# §6.3 calm-regime index-vol band 15–18 % a year; scale target is the midpoint.
CALM_INDEX_VOL_MIN = 0.15
CALM_INDEX_VOL_MAX = 0.18
CALM_INDEX_VOL = 0.165
# Isolated RNG stream names (T6.06). ``noise`` drives the OU; the others stay
# independent so background flow / sentiment draws cannot leak into ``n``.
STREAM_FLOW = "flow"
STREAM_SENTIMENT = "sentiment"
STREAM_NOISE = "noise"
STREAM_NAMES = (STREAM_FLOW, STREAM_SENTIMENT, STREAM_NOISE)


def arb_speed(
    capital: float,
    *,
    a0: float = A_0,
    theta_0: float = THETA_0,
    ratio_min: float = A_RATIO_MIN,
    ratio_max: float = A_RATIO_MAX,
) -> float:
    """``θ_t = θ_0 · clip(A_t/A_0, 0.2, 2)`` (§6.3). Units: 1/day."""
    if a0 == 0.0:
        ratio = float(ratio_min)
    else:
        ratio = float(np.clip(capital / a0, ratio_min, ratio_max))
    return float(theta_0 * ratio)


def ou_decay(theta: float) -> float:
    """Daily OU coefficient ``exp(−θ)`` (§6.3). ``theta`` is 1/day."""
    return float(np.exp(-float(theta)))


def calm_sigma_scale(
    idio_vol: np.ndarray,
    weights: np.ndarray,
    *,
    target_ann_vol: float = CALM_INDEX_VOL,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Scale ``k`` so ``σ_j = k · idio_vol_j`` hits calm index vol (§6.3).

    Independent daily innovations; cap-weighted innovation std is
    ``k · sqrt(Σ w_j² idio_vol_j²)``. ``k`` has units 1/√day per unit of
    ``idio_vol``. ``weights`` must be a simplex. ``idio_vol`` is annual decimal.
    """
    idio = np.asarray(idio_vol, dtype=float)
    w = np.asarray(weights, dtype=float)
    inner = float(np.sqrt(np.sum((w * idio) ** 2)))
    if inner <= 0.0:
        return 0.0
    daily_target = float(target_ann_vol) / np.sqrt(float(days_per_year))
    return float(daily_target / inner)


def trailing_mean(history: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` rows of ``history`` ``(T, n)``. Log return / day."""
    hist = np.asarray(history, dtype=float)
    if hist.size == 0:
        raise ValueError("history is empty")
    k = min(int(window), hist.shape[0])
    return hist[-k:].mean(axis=0)


def cap_weighted_index(xi: np.ndarray, weights: np.ndarray) -> float:
    """Cap-weighted price index at ``V = 1``: ``Σ w_j exp(ξ_j)`` (index points)."""
    return float(np.dot(np.asarray(weights, dtype=float), np.exp(np.asarray(xi, dtype=float))))


def annualised_vol(returns: np.ndarray, days_per_year: int = DAYS_PER_YEAR) -> float:
    """Sample stdev of daily simple returns, annualised (annual decimal)."""
    r = np.asarray(returns, dtype=float)
    return float(r.std(ddof=1) * np.sqrt(float(days_per_year)))


def capital_pnl(xi: np.ndarray, xi_prev: np.ndarray, weights: np.ndarray) -> float:
    """Short-ξ P&L ``−Σ w_j ξ_{j,t−1} Δξ_j`` (§6.3). Dimensionless.

    Negative when the existing dislocation widens against the book.
    """
    x = np.asarray(xi, dtype=float)
    x0 = np.asarray(xi_prev, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(-np.dot(w, x0 * (x - x0)))


def step_capital(
    capital: float,
    pnl: float,
    *,
    a0: float = A_0,
    leak: float = A_LEAK,
    kappa_a: float = KAPPA_A,
    ratio_min: float = A_RATIO_MIN,
    ratio_max: float = A_RATIO_MAX,
) -> float:
    """Leak ``A`` toward ``A_0``, apply P&L, clip to ``[0.2, 2] · A_0`` (§6.3)."""
    a = float(capital) + float(leak) * (float(a0) - float(capital)) + float(kappa_a) * float(pnl)
    return float(np.clip(a, float(ratio_min) * float(a0), float(ratio_max) * float(a0)))


def _as_vec(x: float | np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape {(n,)}, got {arr.shape}")
    return arr.copy()


class Mispricing:
    """Daily ``ξ = I + s + n`` state (§6.3).

    Parameters are arguments with documented defaults (no config keys). Arrays
    are length-``n`` (instruments). ``capital`` is aggregate arb capital.
    """

    def __init__(
        self,
        idio_vol: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        capital: float | None = None,
        a0: float = A_0,
        phi_s: float = PHI_S,
        kappa_mom: float = KAPPA_MOM,
        kappa_news: float = KAPPA_NEWS,
        kappa_risk: float = KAPPA_RISK,
        theta_0: float = THETA_0,
        kappa_a: float = KAPPA_A,
        a_leak: float = A_LEAK,
        ratio_min: float = A_RATIO_MIN,
        ratio_max: float = A_RATIO_MAX,
        window_fast_d: int = WINDOW_FAST_D,
        window_slow_d: int = WINDOW_SLOW_D,
        noise_scale: float | None = None,
        target_ann_vol: float = CALM_INDEX_VOL,
        days_per_year: int = DAYS_PER_YEAR,
        stream_noise: str = STREAM_NOISE,
        stream_sentiment: str = STREAM_SENTIMENT,
        stream_flow: str = STREAM_FLOW,
    ) -> None:
        idio = np.asarray(idio_vol, dtype=float)
        if idio.ndim != 1 or idio.size == 0:
            raise ValueError("idio_vol must be a non-empty 1-d vector")
        n = int(idio.size)
        if weights is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (n,):
                raise ValueError(f"weights must have shape {(n,)}, got {w.shape}")
            total = float(w.sum())
            if total <= 0.0:
                raise ValueError("weights must sum to a positive value")
            w = w / total
        self.n_names = n
        self.idio_vol = idio.copy()
        self.weights = w
        self.a0 = float(a0)
        self.capital = float(self.a0 if capital is None else capital)
        self.phi_s = float(phi_s)
        self.kappa_mom = float(kappa_mom)
        self.kappa_news = float(kappa_news)
        self.kappa_risk = float(kappa_risk)
        self.theta_0 = float(theta_0)
        self.kappa_a = float(kappa_a)
        self.a_leak = float(a_leak)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.window_fast_d = int(window_fast_d)
        self.window_slow_d = int(window_slow_d)
        self.target_ann_vol = float(target_ann_vol)
        self.days_per_year = int(days_per_year)
        self.stream_noise = str(stream_noise)
        self.stream_sentiment = str(stream_sentiment)
        self.stream_flow = str(stream_flow)
        if noise_scale is None:
            self.noise_scale = calm_sigma_scale(
                self.idio_vol,
                self.weights,
                target_ann_vol=self.target_ann_vol,
                days_per_year=self.days_per_year,
            )
        else:
            self.noise_scale = float(noise_scale)
        self.s = np.zeros(n, dtype=float)
        self.n = np.zeros(n, dtype=float)
        self.last_impact = np.zeros(n, dtype=float)
        self._ret = np.zeros((self.window_slow_d, n), dtype=float)
        self._n_ret = 0
        self._ret_i = 0

    @property
    def xi(self) -> np.ndarray:
        """Log-mispricing ``ξ = I + s + n`` (dimensionless), shape ``(n,)``."""
        return self.last_impact + self.s + self.n

    @property
    def theta(self) -> float:
        """Current ``θ_t`` (1/day)."""
        return arb_speed(
            self.capital,
            a0=self.a0,
            theta_0=self.theta_0,
            ratio_min=self.ratio_min,
            ratio_max=self.ratio_max,
        )

    def _chrono_returns(self) -> np.ndarray:
        if self._n_ret == 0:
            return np.zeros((0, self.n_names), dtype=float)
        if self._n_ret < self.window_slow_d:
            return self._ret[: self._n_ret]
        i = self._ret_i
        return np.concatenate([self._ret[i:], self._ret[:i]], axis=0)

    def _push_return(self, ret: np.ndarray) -> None:
        self._ret[self._ret_i] = ret
        self._ret_i = (self._ret_i + 1) % self.window_slow_d
        self._n_ret = min(self._n_ret + 1, self.window_slow_d)

    def _momentum(self) -> np.ndarray:
        """``r̄_20d − r̄_250d`` (log return / day), or 0 before the first return."""
        hist = self._chrono_returns()
        if hist.shape[0] == 0:
            return np.zeros(self.n_names, dtype=float)
        r_fast = trailing_mean(hist, self.window_fast_d)
        r_slow = trailing_mean(hist, self.window_slow_d)
        return r_fast - r_slow

    def step(
        self,
        rng: RngHub,
        impact: float | np.ndarray = 0.0,
        *,
        news: float | np.ndarray = 0.0,
        z_risk: float = 0.0,
    ) -> np.ndarray:
        """Advance one tick (one day). Returns ``ξ`` (dimensionless), shape ``(n,)``.

        ``impact`` is ``I_j`` (log-mispricing). ``news`` is the news impulse
        (same units as ``ξ``, scalar or ``(n,)``). ``z_risk`` is the existing
        risk-appetite primitive (annual decimal) — not a new shock.
        """
        xi_prev = self.xi
        mom = self._momentum()
        news_v = _as_vec(news, self.n_names, "news")
        # OU innovations use ``stream_noise`` only; ``flow`` / ``sentiment``
        # are isolated names (T6.06) and must not be consumed here.
        eps = rng.stream(self.stream_noise).standard_normal(self.n_names)
        self.s = (
            self.phi_s * self.s
            + self.kappa_mom * mom
            + self.kappa_news * news_v
            + self.kappa_risk * float(z_risk)
        )
        theta = self.theta
        sigma = self.noise_scale * self.idio_vol
        self.n = ou_decay(theta) * self.n + sigma * eps
        self.last_impact = _as_vec(impact, self.n_names, "impact")
        xi = self.xi
        self._push_return(xi - xi_prev)
        self.capital = step_capital(
            self.capital,
            capital_pnl(xi, xi_prev, self.weights),
            a0=self.a0,
            leak=self.a_leak,
            kappa_a=self.kappa_a,
            ratio_min=self.ratio_min,
            ratio_max=self.ratio_max,
        )
        return xi.copy()

    def to_state(self) -> dict[str, Any]:
        """JSON-safe snapshot of parameters and dynamic state."""
        return {
            "idio_vol": self.idio_vol.tolist(),
            "weights": self.weights.tolist(),
            "s": self.s.tolist(),
            "n": self.n.tolist(),
            "capital": float(self.capital),
            "last_impact": self.last_impact.tolist(),
            "returns": self._chrono_returns().tolist(),
            "a0": self.a0,
            "phi_s": self.phi_s,
            "kappa_mom": self.kappa_mom,
            "kappa_news": self.kappa_news,
            "kappa_risk": self.kappa_risk,
            "theta_0": self.theta_0,
            "kappa_a": self.kappa_a,
            "a_leak": self.a_leak,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "window_fast_d": self.window_fast_d,
            "window_slow_d": self.window_slow_d,
            "noise_scale": self.noise_scale,
            "target_ann_vol": self.target_ann_vol,
            "days_per_year": self.days_per_year,
            "stream_noise": self.stream_noise,
            "stream_sentiment": self.stream_sentiment,
            "stream_flow": self.stream_flow,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Mispricing:
        needed = ("idio_vol", "s", "n", "capital")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"mispricing state missing {missing}")
        obj = cls(
            np.asarray(state["idio_vol"], dtype=float),
            np.asarray(state.get("weights", np.ones(len(state["idio_vol"]))), dtype=float),
            capital=float(state["capital"]),
            a0=float(state.get("a0", A_0)),
            phi_s=float(state.get("phi_s", PHI_S)),
            kappa_mom=float(state.get("kappa_mom", KAPPA_MOM)),
            kappa_news=float(state.get("kappa_news", KAPPA_NEWS)),
            kappa_risk=float(state.get("kappa_risk", KAPPA_RISK)),
            theta_0=float(state.get("theta_0", THETA_0)),
            kappa_a=float(state.get("kappa_a", KAPPA_A)),
            a_leak=float(state.get("a_leak", A_LEAK)),
            ratio_min=float(state.get("ratio_min", A_RATIO_MIN)),
            ratio_max=float(state.get("ratio_max", A_RATIO_MAX)),
            window_fast_d=int(state.get("window_fast_d", WINDOW_FAST_D)),
            window_slow_d=int(state.get("window_slow_d", WINDOW_SLOW_D)),
            noise_scale=float(state["noise_scale"]) if "noise_scale" in state else None,
            target_ann_vol=float(state.get("target_ann_vol", CALM_INDEX_VOL)),
            days_per_year=int(state.get("days_per_year", DAYS_PER_YEAR)),
            stream_noise=str(state.get("stream_noise", STREAM_NOISE)),
            stream_sentiment=str(state.get("stream_sentiment", STREAM_SENTIMENT)),
            stream_flow=str(state.get("stream_flow", STREAM_FLOW)),
        )
        s = np.asarray(state["s"], dtype=float)
        ou = np.asarray(state["n"], dtype=float)
        if s.shape != (obj.n_names,) or ou.shape != (obj.n_names,):
            raise StateError("mispricing s/n must match idio_vol length")
        obj.s = s.copy()
        obj.n = ou.copy()
        if "last_impact" in state:
            obj.last_impact = _as_vec(state["last_impact"], obj.n_names, "last_impact")
        raw_ret = np.asarray(state.get("returns", []), dtype=float)
        if raw_ret.size:
            if raw_ret.ndim != 2 or raw_ret.shape[1] != obj.n_names:
                raise StateError("mispricing returns must have shape (T, n)")
            for row in raw_ret[-obj.window_slow_d :]:
                obj._push_return(row)
        return obj
