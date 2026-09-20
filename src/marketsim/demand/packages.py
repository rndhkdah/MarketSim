"""Need shapes and budget scaling (§3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def survival(y: np.ndarray | float, v_max: float, y_s: float) -> np.ndarray:
    """``v_max·(1 − exp(−y/y_s))``."""
    yy = np.asarray(y, dtype=float)
    return float(v_max) * (1.0 - np.exp(-yy / max(float(y_s), 1e-12)))


def plateau(y: np.ndarray | float, v_max: float, y_p: float) -> np.ndarray:
    """``v_max·min(1, y/y_p)``."""
    yy = np.asarray(y, dtype=float)
    return float(v_max) * np.minimum(1.0, yy / max(float(y_p), 1e-12))


def vanish(y: np.ndarray | float, v_pk: float, y_pk: float) -> np.ndarray:
    """``v_pk·(y/y_pk)·exp(1 − y/y_pk)``. Peaks at ``y_pk``, → 0 as ``y→∞``."""
    yy = np.asarray(y, dtype=float)
    yp = max(float(y_pk), 1e-12)
    return float(v_pk) * (yy / yp) * np.exp(1.0 - yy / yp)


def normal(y: np.ndarray | float, b: float) -> np.ndarray:
    """``b·y``."""
    return float(b) * np.asarray(y, dtype=float)


def luxury(y: np.ndarray | float, b: float, y_th: float, gamma: float) -> np.ndarray:
    """``b·max(0, y − y_th)^γ``, ``γ ∈ [1.3, 2]``."""
    yy = np.asarray(y, dtype=float)
    return float(b) * np.power(np.maximum(0.0, yy - float(y_th)), float(gamma))


SHAPE_FNS = {
    "survival": lambda y, p: survival(y, p["v_max"], p["y_s"]),
    "plateau": lambda y, p: plateau(y, p["v_max"], p["y_p"]),
    "vanish": lambda y, p: vanish(y, p["v_pk"], p["y_pk"]),
    "normal": lambda y, p: normal(y, p["b"]),
    "luxury": lambda y, p: luxury(y, p["b"], p["y_th"], p["gamma"]),
}


@dataclass
class PackageParams:
    """Per-want shape parameters (generated / edited in ``buy_packages.yaml``)."""

    shape: str
    values: dict[str, float]


def evaluate_shape(shape: str, params: dict[str, float], y: np.ndarray | float) -> np.ndarray:
    """Package value at real income ``y`` (index). Units: budget share before scaling."""
    if shape not in SHAPE_FNS:
        raise ValueError(f"unknown shape {shape}")
    return np.asarray(SHAPE_FNS[shape](y, params), dtype=float)


def default_params(shape: str) -> dict[str, float]:
    """Deterministic start values for the calibrator."""
    if shape == "survival":
        return {"v_max": 0.12, "y_s": 0.35}
    if shape == "plateau":
        return {"v_max": 0.10, "y_p": 1.00}
    if shape == "vanish":
        return {"v_pk": 0.04, "y_pk": 0.40}
    if shape == "normal":
        return {"b": 0.08}
    if shape == "luxury":
        return {"b": 0.06, "y_th": 0.80, "gamma": 1.5}
    raise ValueError(shape)


def scale_budget(
    values: np.ndarray,
    shapes: tuple[str, ...],
    budget: float,
) -> np.ndarray:
    """Survival served first; remainder pro-rata; surplus over non-survival.

    ``values`` is ``(Q,)`` package value; return spends ``(Q,)`` summing to ``budget``.
    """
    v = np.maximum(np.asarray(values, dtype=float), 0.0)
    surv = np.array([s == "survival" for s in shapes])
    rest = ~surv
    out = np.zeros_like(v)
    b = float(budget)
    surv_need = float(v[surv].sum()) if np.any(surv) else 0.0
    if surv_need >= b and surv_need > 0:
        out[surv] = v[surv] * (b / surv_need)
        return out
    out[surv] = v[surv]
    leftover = b - surv_need
    rest_v = float(v[rest].sum())
    if rest_v > leftover and rest_v > 0:
        out[rest] = v[rest] * (leftover / rest_v)
        return out
    out[rest] = v[rest]
    surplus = leftover - rest_v
    if surplus > 0 and np.any(rest):
        w = v[rest]
        if float(w.sum()) <= 0:
            w = np.ones(int(rest.sum()))
        out[rest] = out[rest] + surplus * w / w.sum()
    return out
