"""Daily-resolution impact kernel (T6.07 / §6.4).

Concave instantaneous impact times a power-law propagator, realised as a
mixture of ``K`` exponentials. ``ξ_impact`` is log-mispricing (dimensionless);
``P = V · exp(ξ)`` with ``ξ = ξ_impact + sentiment + noise`` (sentiment and
noise are out of scope here).

::

    ΔI_j   = Y · σ_j · sign(q) · (|q| / ADV_j)^δ
    I_k   ← ρ_k · I_k + w_k · ΔI_j
    ξ_impact = Σ_k I_k

``ρ_k = 2^(-1/h_k)`` with half-lives ``h_k`` in days. Weights ``w_k`` are a
non-negative least-squares fit of the impulse response ``Σ_k w_k ρ_k^τ`` to
``G(τ) = (1 + τ/τ0)^(-β)`` on a log-spaced grid out to 500 days. Weights are
**not** renormalised to sum to one — raw NNLS stays inside the card's 10 %
band on 1–250 days.

Fills pay the full post-impact mid ± half-spread (nobody trades ahead of
their own impact). State is ``O(K)`` per instrument.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import nnls

from marketsim.core.errors import StateError
from marketsim.market.instruments import ImpactCfg

# §6.4 / T6.07: NNLS lag grid is log-spaced out to 500 days.
FIT_HORIZON_D = 500.0
# §6.4 / T6.07: number of log-spaced nodes on [1, FIT_HORIZON_D] days.
FIT_GRID_N = 64


def _as_out(x: np.ndarray) -> float | np.ndarray:
    """Return a Python float for 0-d results; otherwise the array."""
    if x.ndim == 0:
        return float(x)
    return x


def power_law_kernel(
    tau_d: float | np.ndarray,
    tau0_d: float,
    beta: float,
) -> float | np.ndarray:
    """Target propagator ``G(τ) = (1 + τ/τ0)^(-β)`` (§6.4).

    ``tau_d`` and ``tau0_d`` are days; ``beta`` is dimensionless. ``G`` is a
    dimensionless remaining-impact factor (``G(0) = 1``).
    """
    tau = np.asarray(tau_d, dtype=float)
    return _as_out(np.power(1.0 + tau / float(tau0_d), -float(beta)))


def daily_decays(half_lives_d: Sequence[float]) -> np.ndarray:
    """Daily decay factors ``ρ_k = 2^(-1/h_k)`` (§6.4).

    ``half_lives_d`` are days; each ``ρ_k`` is dimensionless in ``(0, 1)``.
    A day with ``q = 0`` applies ``I_k ← ρ_k I_k``.
    """
    h = np.asarray(half_lives_d, dtype=float)
    if h.size == 0 or np.any(h <= 0.0):
        raise ValueError("half_lives_d must be non-empty and > 0 (days)")
    return np.power(2.0, -1.0 / h)


def impulse_response(
    tau_d: float | np.ndarray,
    weights: np.ndarray,
    rho: np.ndarray,
) -> float | np.ndarray:
    """Mixture impulse response ``Σ_k w_k ρ_k^τ`` (§6.4). Dimensionless.

    ``tau_d`` is days; ``weights`` and ``rho`` are length-``K`` (dimensionless).
    """
    tau = np.asarray(tau_d, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(rho, dtype=float)
    return _as_out(np.sum(w * np.power(r, tau[..., None]), axis=-1))


def fit_kernel_weights(
    half_lives_d: Sequence[float],
    tau0_d: float,
    beta: float,
    *,
    horizon_d: float = FIT_HORIZON_D,
    n_grid: int = FIT_GRID_N,
) -> np.ndarray:
    """Non-negative least squares of ``Σ_k w_k ρ_k^τ ≈ G(τ)`` (§6.4 / T6.07).

    Fitted on a log-spaced grid from 1 day to ``horizon_d`` days (default
    ``FIT_HORIZON_D`` = 500). Returns ``w`` of length ``K`` (dimensionless,
    ``w_k ≥ 0``). Does not force ``Σ w_k = 1``.
    """
    if horizon_d <= 0.0:
        raise ValueError("horizon_d must be > 0 (days)")
    if n_grid < 1:
        raise ValueError("n_grid must be >= 1")
    rho = daily_decays(half_lives_d)
    # log-spaced inclusive endpoints; 1 day is the first on-grid lag (§6.4).
    tau = np.geomspace(1.0, float(horizon_d), int(n_grid))
    target = np.asarray(power_law_kernel(tau, tau0_d, beta), dtype=float)
    design = np.power(rho, tau[:, None])
    weights, _rnorm = nnls(design, target)
    return np.asarray(weights, dtype=float)


def delta_impact(
    q: float | np.ndarray,
    adv: float | np.ndarray,
    sigma: float | np.ndarray,
    *,
    Y: float,
    delta: float,
) -> float | np.ndarray:
    """Instantaneous increment ``ΔI = Y · σ · sign(q) · (|q|/ADV)^δ`` (§6.4).

    Parameters
    ----------
    q:
        Net signed executed volume this tick (same units as ``adv``, typically cr).
    adv:
        Average daily volume (cr/day). Required ``> 0`` wherever ``q ≠ 0``.
    sigma:
        Daily return volatility (decimal, e.g. ``0.01``). Must be ``>= 0``.
    Y:
        Dimensionless impact scale.
    delta:
        Concave participation exponent (dimensionless; ``δ < 1``).

    Returns
    -------
    ``ΔI`` in log-price units (dimensionless). ``q = 0 ⇒ ΔI = 0``.
    """
    if Y <= 0.0 or delta <= 0.0:
        raise ValueError("Y and delta must be > 0 (§6.4)")
    q_arr = np.asarray(q, dtype=float)
    adv_arr = np.asarray(adv, dtype=float)
    sig_arr = np.asarray(sigma, dtype=float)
    traded = q_arr != 0.0
    if np.any(traded & (adv_arr <= 0.0)):
        raise ValueError("ADV must be > 0 (cr/day) where q != 0")
    if np.any(sig_arr < 0.0):
        raise ValueError("sigma must be >= 0 (daily vol, decimal)")
    ratio = np.zeros(np.broadcast_shapes(q_arr.shape, adv_arr.shape, sig_arr.shape), dtype=float)
    # 0^δ is 0 for δ > 0; avoid 0/0 when q = 0 and ADV is unused.
    np.divide(np.abs(q_arr), adv_arr, out=ratio, where=traded)
    out = float(Y) * sig_arr * np.sign(q_arr) * np.power(ratio, float(delta))
    return _as_out(np.asarray(out, dtype=float))


def post_impact_mid(
    mid: float | np.ndarray,
    xi_pre: float | np.ndarray,
    xi_post: float | np.ndarray,
) -> float | np.ndarray:
    """Post-impact mid from the current mid and the ξ move (§6.3–§6.4).

    ``mid_post = mid · exp(ξ_post − ξ_pre)``. ``mid`` is the pre-impact mid
    (cr/unit); ``ξ`` are dimensionless log-mispricing. Apply ``ΔI`` to ``ξ``
    *before* quoting the fill.
    """
    m = np.asarray(mid, dtype=float)
    if np.any(m <= 0.0):
        raise ValueError("mid must be > 0 (cr/unit)")
    moved = m * np.exp(np.asarray(xi_post, dtype=float) - np.asarray(xi_pre, dtype=float))
    return _as_out(np.asarray(moved, dtype=float))


def fill_price(
    mid_post: float | np.ndarray,
    spread: float | np.ndarray,
    side: float | np.ndarray,
) -> float | np.ndarray:
    """Fill at post-impact mid ± half-spread (§6.4).

    ``p_fill = mid_post · (1 + sign(side) · s/2)``. ``mid_post`` and ``p_fill``
    are cr/unit; ``spread`` is the dimensionless quoted spread ``s`` (half-spread
    is ``s/2``). ``side > 0`` is a buy (pays above mid); ``side < 0`` is a sell.
    """
    mid = np.asarray(mid_post, dtype=float)
    spr = np.asarray(spread, dtype=float)
    if np.any(mid <= 0.0):
        raise ValueError("mid_post must be > 0 (cr/unit)")
    if np.any(spr < 0.0):
        raise ValueError("spread must be >= 0 (dimensionless)")
    out = mid * (1.0 + np.sign(np.asarray(side, dtype=float)) * spr / 2.0)
    return _as_out(np.asarray(out, dtype=float))


class ImpactKernel:
    """Fitted §6.4 kernel plus ``O(K)`` impact state per instrument.

    ``I`` has shape ``(*shape, K)``. Default ``shape=()`` is one name, so
    ``I`` is the length-``K`` vector. A multi-name book uses ``shape=(N,)``;
    ``step`` / ``fill`` are numpy-broadcast over the leading axes (no Python
    loop over names). A loop over ``K = 5`` would be fine; the update is
    fully vectorised over ``K`` as well.

    Parameters
    ----------
    cfg:
        ``ImpactCfg`` (defaults match ``config/markets.yaml``).
    shape:
        Leading instrument axes; state length along the last axis is ``K``.
    """

    def __init__(self, cfg: ImpactCfg | None = None, *, shape: tuple[int, ...] = ()) -> None:
        self.cfg = cfg if cfg is not None else ImpactCfg()
        self.half_lives_d = tuple(float(h) for h in self.cfg.half_lives_d)
        self.delta = float(self.cfg.delta)
        self.beta = float(self.cfg.beta)
        self.Y = float(self.cfg.Y)
        self.tau0_d = float(self.cfg.tau0_d)
        self.rho = daily_decays(self.half_lives_d)
        self.weights = fit_kernel_weights(self.half_lives_d, self.tau0_d, self.beta)
        self.I = np.zeros(tuple(shape) + (len(self.half_lives_d),), dtype=float)

    @property
    def n_components(self) -> int:
        """``K`` — number of exponential components (dimensionless)."""
        return len(self.half_lives_d)

    @property
    def xi(self) -> float | np.ndarray:
        """``ξ_impact = Σ_k I_k`` (log-price, dimensionless)."""
        return _as_out(self.I.sum(axis=-1))

    def step(
        self,
        q: float | np.ndarray,
        adv: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> float | np.ndarray:
        """One daily update: decay every component, then add ``w_k · ΔI``.

        ``q`` is net signed volume this tick (cr); ``adv`` is cr/day; ``sigma``
        is daily vol (decimal). Returns the post-update ``ξ_impact``.
        ``q = 0`` applies decay only.
        """
        dI = np.asarray(delta_impact(q, adv, sigma, Y=self.Y, delta=self.delta), dtype=float)
        try:
            dI = np.broadcast_to(dI, self.I.shape[:-1])
        except ValueError as exc:
            raise ValueError(
                f"q/ADV/sigma shape {dI.shape} must broadcast to {self.I.shape[:-1]}"
            ) from exc
        # I_k ← ρ_k I_k + w_k ΔI  (§6.4); leading axes are instruments.
        self.I = self.I * self.rho + dI[..., None] * self.weights
        return self.xi

    def fill(
        self,
        mid: float | np.ndarray,
        spread: float | np.ndarray,
        q: float | np.ndarray,
        adv: float | np.ndarray,
        sigma: float | np.ndarray,
    ) -> float | np.ndarray:
        """Apply this tick's net ``q``, then fill at post-impact mid ± half-spread.

        ``mid`` is the *pre-impact* mid (cr/unit). Impact is applied to ``ξ``
        first; the fill uses ``mid_post = mid · exp(Δξ)`` (§6.3–§6.4). Buy
        (``q > 0``) pays above the post-impact mid.
        """
        xi_pre = np.asarray(self.xi, dtype=float)
        self.step(q, adv, sigma)
        mid_post = post_impact_mid(mid, xi_pre, np.asarray(self.xi, dtype=float))
        return fill_price(mid_post, spread, q)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe snapshot. ``I`` is the length-``K`` (or ``N×K``) state."""
        return {
            "I": self.I.tolist(),
            "delta": self.delta,
            "beta": self.beta,
            "Y": self.Y,
            "tau0_d": self.tau0_d,
            "half_lives_d": list(self.half_lives_d),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ImpactKernel:
        needed = ("I", "delta", "beta", "Y", "tau0_d", "half_lives_d")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"impact state missing {missing}")
        half = tuple(float(h) for h in state["half_lives_d"])
        i_vec = np.asarray(state["I"], dtype=float)
        if i_vec.ndim == 0 or i_vec.shape[-1] != len(half):
            raise StateError("impact I last axis must equal len(half_lives_d)")
        obj = cls(
            ImpactCfg(
                delta=float(state["delta"]),
                beta=float(state["beta"]),
                Y=float(state["Y"]),
                tau0_d=float(state["tau0_d"]),
                half_lives_d=half,
            ),
            shape=i_vec.shape[:-1],
        )
        obj.I = i_vec.copy()
        if "weights" in state:
            w = np.asarray(state["weights"], dtype=float)
            if w.shape != (len(half),):
                raise StateError("impact weights must have shape (K,)")
            obj.weights = w
        return obj
