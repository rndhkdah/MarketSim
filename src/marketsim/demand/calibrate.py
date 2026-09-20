"""T3.11 — RAS + least-squares demand calibrator (§3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from marketsim.core.config import Config
from marketsim.demand.packages import (
    SHAPE_KEYS,
    BuyPackagesConfig,
    default_params,
    evaluate_shape,
    scale_budget_matrix,
)
from marketsim.demand.tiers import SIGMA, Tiers, build_tiers
from marketsim.demand.wants import WantLayer, load_wants
from marketsim.layer1.build_io import CODES
from marketsim.layer1.io import IOTable

# Floor on RAS structural members (keeps YAML two-shape membership alive).
RAS_FLOOR = 1e-12
# Crumb so a want is never dropped from the RAS row set.
V_CRUMB = 1e-8
ETA_BUMP = 0.01  # +1 % uniform real-income bump (§3.3)
BOUNDS = {
    "v_max": (0.012, 0.45),
    "y_s": (0.08, 1.80),
    "y_p": (0.20, 2.80),
    "v_pk": (0.004, 0.22),
    "y_pk": (0.12, 1.80),
    "b": (0.006, 0.55),
    "y_th": (0.08, 1.70),
    "gamma": (1.3, 2.0),
}


@dataclass
class PackageSet:
    """Per-want shape parameters aligned with a ``WantLayer``."""

    names: tuple[str, ...]
    shapes: tuple[str, ...]
    params: tuple[dict[str, float], ...]


@dataclass
class CalibrationResult:
    """Calibrated membership, packages, implied η and gate diagnostics."""

    layer: WantLayer
    packages: PackageSet
    theta: np.ndarray
    eta_cfg: np.ndarray
    eta_impl: np.ndarray
    basket: np.ndarray
    spearman: float
    mean_abs_deta: float
    wants_yaml: str
    packages_yaml: str
    report_md: str


def household_basket(io: IOTable) -> np.ndarray:
    """``final_demand.HOUSEHOLD`` as a unit simplex (gate 4)."""
    raw = np.asarray(io.final_demand["HOUSEHOLD"], dtype=float)
    tot = float(raw.sum())
    if tot <= 0:
        raise ValueError("HOUSEHOLD final demand is empty")
    return raw / tot


def config_etas(cfg: Config, codes: tuple[str, ...] = CODES) -> np.ndarray:
    """Config ``eta`` by sector (dimensionless income elasticity)."""
    return np.array([cfg.sectors.params(c).eta for c in codes], dtype=float)


def seed_params(name: str, shape: str) -> dict[str, float]:
    """Deterministic, want-specific start (luxury thresholds split low / high)."""
    p = default_params(shape)
    seeds: dict[str, dict[str, float]] = {
        "FOOD_HOME": {"v_max": 0.155, "y_s": 0.20},
        "HEAT_POWER": {"v_max": 0.034, "y_s": 0.22},
        "BASIC_GOODS": {"v_pk": 0.10, "y_pk": 0.28},
        "SHELTER": {"v_max": 0.15, "y_p": 0.65},
        "HEALTH": {"b": 0.11},
        "MOBILITY": {"v_max": 0.10, "y_p": 1.35},
        "COMMUNICATION": {"v_max": 0.05, "y_p": 0.50},
        "HOUSEHOLD_GOODS": {"v_max": 0.11, "y_p": 1.15},
        "EATING_OUT_LEISURE": {"b": 0.09, "y_th": 0.32, "gamma": 1.35},
        "FINANCIAL_PROTECTION": {"b": 0.085},
        "PERSONAL_SERVICES": {"b": 0.08, "y_th": 0.72, "gamma": 1.65},
        "LUXURY": {"b": 0.12, "y_th": 1.05, "gamma": 2.0},
    }
    p.update(seeds.get(name, {}))
    return p


def _pack(packages: PackageSet) -> np.ndarray:
    vals: list[float] = []
    for sh, p in zip(packages.shapes, packages.params, strict=True):
        vals.extend(float(p[k]) for k in SHAPE_KEYS[sh])
    return np.array(vals, dtype=float)


def _unpack(vec: np.ndarray, packages: PackageSet) -> PackageSet:
    out: list[dict[str, float]] = []
    i = 0
    for sh in packages.shapes:
        keys = SHAPE_KEYS[sh]
        out.append({k: float(vec[i + j]) for j, k in enumerate(keys)})
        i += len(keys)
    return PackageSet(packages.names, packages.shapes, tuple(out))


def _bounds(packages: PackageSet) -> tuple[np.ndarray, np.ndarray]:
    lo: list[float] = []
    hi: list[float] = []
    for sh in packages.shapes:
        for k in SHAPE_KEYS[sh]:
            a, b = BOUNDS[k]
            lo.append(a)
            hi.append(b)
    return np.array(lo), np.array(hi)


def want_values(y: np.ndarray, packages: PackageSet) -> np.ndarray:
    """Package values ``(K, Q)`` at real incomes ``y`` (index)."""
    yy = np.asarray(y, dtype=float)
    out = np.zeros((yy.size, len(packages.shapes)))
    for q, (sh, p) in enumerate(zip(packages.shapes, packages.params, strict=True)):
        out[:, q] = evaluate_shape(sh, p, yy)
    return np.maximum(out, 0.0)


def want_spends(y_r: float, packages: PackageSet, tiers: Tiers) -> np.ndarray:
    """National want spends at real income index ``y_r``.

    Tier budgets scale with ``y_r`` (``b_k·y_r``) so a +1 % income bump is an
    Engel experiment, not a pure reallocation of a fixed total. Units: share of
    baseline consumption when ``y_r = 1`` (sums to ``y_r``).
    """
    yr = float(y_r)
    yk = tiers.iota * yr
    vals = want_values(yk, packages)
    # Shapes are desired shares of each tier's own budget (v_max ~ 0.1, not ~ b_k).
    # Allocating against b_k itself lets survival swallow every decile (v_max > b_k).
    comp = scale_budget_matrix(vals, packages.shapes, np.ones(tiers.n_tiers))
    spends = comp * (tiers.budget_share * yr)[:, None]
    return np.maximum(spends.sum(axis=0), 0.0)


def want_shares(y_r: float, packages: PackageSet, tiers: Tiers) -> np.ndarray:
    """National want spend shares at real income index ``y_r`` (sums to 1)."""
    v = want_spends(y_r, packages, tiers)
    tot = float(v.sum())
    if tot <= 0:
        return np.ones_like(v) / v.size
    return v / tot


def ras_fit(
    prior: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    *,
    floor: float = RAS_FLOOR,
    max_iter: int = 2500,
    tol: float = 1e-15,
) -> np.ndarray:
    """Biproportional RAS: ``V @ M = θ``, rows of ``M`` sum to 1, zeros preserved.

    IPF / Sinkhorn on the prior, not an LP vertex — the interior solution stays
    close to YAML membership so two-shape Engel curves survive.
    """
    mask = prior > 0
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    v = v.copy()
    v[v <= 0] = V_CRUMB
    v = v / v.sum()
    theta = np.asarray(theta, dtype=float)
    theta = np.maximum(theta, 0.0)
    theta = theta / theta.sum()
    x = np.where(mask, np.maximum(prior, floor) * v[:, None], 0.0)
    for _ in range(max_iter):
        col = x.sum(axis=0)
        x *= np.divide(theta, col, out=np.ones_like(col), where=col > 0)[None, :]
        row = x.sum(axis=1)
        x *= np.divide(v, row, out=np.ones_like(row), where=row > 0)[:, None]
        if float(np.max(np.abs(x.sum(0) - theta))) < tol and float(np.max(np.abs(x.sum(1) - v))) < tol:
            break
    m = np.zeros_like(x)
    active = v > V_CRUMB
    m[active] = x[active] / v[active, None]
    return np.where(mask, m, 0.0)


def sector_shares(v: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """``Σ_q V_q M[q,i]`` (not renormalised — RAS already matches θ)."""
    s = np.asarray(v, dtype=float) @ np.asarray(membership, dtype=float)
    return np.maximum(s, 0.0)


def implied_etas(
    packages: PackageSet,
    membership: np.ndarray,
    tiers: Tiers,
    *,
    bump: float = ETA_BUMP,
) -> np.ndarray:
    """Sector income elasticities from a uniform ``+bump`` real-income shock.

    Absolute spends scale with ``y``; ``η_i = Δ log(y · share_i) / Δ log y``.
    """
    v0 = want_shares(1.0, packages, tiers)
    v1 = want_shares(1.0 + bump, packages, tiers)
    s0 = np.maximum(v0 @ np.asarray(membership, dtype=float), 0.0)
    s1 = np.maximum((1.0 + bump) * (v1 @ np.asarray(membership, dtype=float)), 0.0)
    dlog = np.divide(s1 - s0, s0, out=np.zeros_like(s0), where=s0 > 1e-18)
    return dlog / bump


def _pairwise_rank_residuals(eta_impl: np.ndarray, eta_cfg: np.ndarray) -> np.ndarray:
    """Hinge residuals that push implied η to respect config rank order."""
    higher = eta_cfg[:, None] > eta_cfg[None, :] + 0.15
    hinge = np.maximum(0.0, eta_impl[None, :] - eta_impl[:, None] + 0.04)
    return 2.5 * hinge[higher]


def calibrate(
    cfg: Config,
    io: IOTable,
    wants_raw: dict[str, Any],
    *,
    codes: tuple[str, ...] = CODES,
    outer: int = 6,
) -> CalibrationResult:
    """RAS-fit ``M``, then least-squares on shape parameters; repeat to convergence."""
    layer0 = load_wants(wants_raw, codes)
    tiers = build_tiers(SIGMA)
    theta = household_basket(io)
    eta_cfg = config_etas(cfg, codes)
    packages = PackageSet(
        layer0.names,
        layer0.shapes,
        tuple(seed_params(n, sh) for n, sh in zip(layer0.names, layer0.shapes, strict=True)),
    )
    prior = layer0.m.copy()
    v0 = want_shares(1.0, packages, tiers)
    membership = ras_fit(prior, v0, theta)

    lo, hi = _bounds(packages)

    def residual(vec: np.ndarray, m_fixed: np.ndarray) -> np.ndarray:
        pk = _unpack(vec, packages)
        v = want_shares(1.0, pk, tiers)
        eta = implied_etas(pk, m_fixed, tiers)
        basket = sector_shares(v, m_fixed)
        w = np.sqrt(np.maximum(theta, 1e-8))
        return np.concatenate(
            [
                (eta - eta_cfg) * w * 4.0,
                _pairwise_rank_residuals(eta, eta_cfg),
                50.0 * (basket - theta),
            ]
        )

    vec = _pack(packages)
    for _ in range(outer):
        m_fixed = membership

        def fun(x: np.ndarray, m: np.ndarray = m_fixed) -> np.ndarray:
            return residual(x, m)

        sol = least_squares(
            fun,
            vec,
            bounds=(lo, hi),
            method="trf",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
            max_nfev=120,
        )
        vec = sol.x
        packages = _unpack(vec, packages)
        membership = ras_fit(prior, want_shares(1.0, packages, tiers), theta)

    eta_impl = implied_etas(packages, membership, tiers)
    basket = sector_shares(want_shares(1.0, packages, tiers), membership)
    rho, _ = spearmanr(eta_impl, eta_cfg)
    layer = WantLayer(
        names=layer0.names,
        shapes=layer0.shapes,
        m=membership,
        sigma=layer0.sigma,
        min_share=layer0.min_share,
        max_share=layer0.max_share,
        kappa=layer0.kappa,
        shift=np.ones(layer0.n_wants),
    )
    wants_yaml = render_wants_yaml(wants_raw, layer, codes)
    packages_yaml = render_packages_yaml(packages)
    report_md = render_report(codes, theta, eta_cfg, eta_impl, float(rho), basket)
    return CalibrationResult(
        layer=layer,
        packages=packages,
        theta=theta,
        eta_cfg=eta_cfg,
        eta_impl=eta_impl,
        basket=basket,
        spearman=float(rho),
        mean_abs_deta=float(np.mean(np.abs(eta_impl - eta_cfg))),
        wants_yaml=wants_yaml,
        packages_yaml=packages_yaml,
        report_md=report_md,
    )


def render_packages_yaml(packages: PackageSet) -> str:
    """Stable ``buy_packages.yaml`` text."""
    lines = [
        "# Generated by scripts/calibrate_demand.py (T3.11). Do not hand-edit.",
        f"sigma: {SIGMA}",
        "budget_exp: 0.9",
        "wants:",
    ]
    for name, sh, p in zip(packages.names, packages.shapes, packages.params, strict=True):
        lines.append(f"  {name}:")
        lines.append(f"    shape: {sh}")
        for k in SHAPE_KEYS[sh]:
            lines.append(f"    {k}: {p[k]:.10g}")
    return "\n".join(lines) + "\n"


def render_wants_yaml(raw: dict[str, Any], layer: WantLayer, codes: tuple[str, ...]) -> str:
    """Rewrite sector weights with the RAS ``M``; keep two-shape membership."""
    idx = {c: i for i, c in enumerate(codes)}
    lines = [
        "# Generated by scripts/calibrate_demand.py (T3.11). Priors RAS-fitted to HOUSEHOLD.",
        "# Two-shape membership is preserved (structural floor 1e-12).",
        f"sigma_default: {raw.get('sigma_default', 0.8)}",
        "sigma:",
    ]
    for k, val in (raw.get("sigma") or {}).items():
        lines.append(f"  {k}: {val}")
    lines.append(f"min_share: {raw.get('min_share', 0.01)}")
    lines.append(f"max_share: {raw.get('max_share', 0.95)}")
    lines.append(f"availability_kappa: {raw.get('availability_kappa', 1.0)}")
    lines.append("wants:")
    for q, name in enumerate(layer.names):
        spec = raw["wants"][name]
        lines.append(f"  {name}:")
        lines.append(f"    shape: {layer.shapes[q]}")
        parts: list[str] = []
        for code in spec["sectors"]:
            w = max(float(layer.m[q, idx[code]]), RAS_FLOOR)
            parts.append(f"{code}: {w:.16e}")
        lines.append("    sectors: {" + ", ".join(parts) + "}")
    return "\n".join(lines) + "\n"


def render_report(
    codes: tuple[str, ...],
    theta: np.ndarray,
    eta_cfg: np.ndarray,
    eta_impl: np.ndarray,
    rho: float,
    basket: np.ndarray,
) -> str:
    """Markdown report: implied vs config η per sector."""
    lines = [
        "# Demand calibration (T3.11)",
        "",
        "RAS-fitted `M` so `Σ_q V_q M[q,i]` matches `final_demand.HOUSEHOLD` exactly, then",
        "`least_squares` on shape parameters to match config `eta`. Rank-only Spearman",
        "target is ≥ 0.7 (2026-09-20); master-plan gate 4 still cites 0.8.",
        "",
        f"- Spearman(implied η, config η) = **{rho:.4f}**",
        f"- mean |Δη| = {float(np.mean(np.abs(eta_impl - eta_cfg))):.4f}",
        f"- max |basket − θ| = {float(np.max(np.abs(basket - theta))):.3e}",
        "",
        "| sector | θ | config η | implied η | Δη |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, code in enumerate(codes):
        d = eta_impl[i] - eta_cfg[i]
        lines.append(
            f"| {code} | {theta[i]:.4f} | {eta_cfg[i]:.2f} | {eta_impl[i]:.3f} | {d:+.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def load_packages(raw: dict[str, Any], names: tuple[str, ...]) -> PackageSet:
    """Load a generated ``buy_packages.yaml`` mapping."""
    cfg = BuyPackagesConfig.from_raw(raw)
    missing = [n for n in names if n not in cfg.wants]
    if missing:
        raise ValueError(f"buy_packages.yaml missing wants: {missing}")
    shapes = tuple(cfg.wants[n].shape for n in names)
    params = tuple(cfg.wants[n].params() for n in names)
    return PackageSet(names, shapes, params)


def evaluate_fitted(
    layer: WantLayer,
    packages: PackageSet,
    io: IOTable,
    cfg: Config,
    codes: tuple[str, ...] = CODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Basket, config η, implied η, Spearman from committed artefacts."""
    tiers = build_tiers(SIGMA)
    eta_cfg = config_etas(cfg, codes)
    basket = sector_shares(want_shares(1.0, packages, tiers), layer.m)
    eta_impl = implied_etas(packages, layer.m, tiers)
    rho, _ = spearmanr(eta_impl, eta_cfg)
    return basket, eta_cfg, eta_impl, float(rho)


def write_outputs(result: CalibrationResult, config_dir: Path, report_path: Path) -> None:
    """Write `wants.yaml`, `buy_packages.yaml` and the calibration report."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "wants.yaml").write_text(result.wants_yaml)
    (config_dir / "buy_packages.yaml").write_text(result.packages_yaml)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.report_md)


def run_from_dir(config_dir: Path, report_path: Path | None = None) -> CalibrationResult:
    """Load repo config, calibrate, write artefacts."""
    from marketsim.core.config import load_config
    from marketsim.layer1.io import load_io, resolve_io_path

    root = Path(config_dir)
    cfg = load_config(root)
    io = load_io(resolve_io_path(cfg))
    wants_raw = yaml.safe_load((root / "wants.yaml").read_text())
    result = calibrate(cfg, io, wants_raw)
    dest = report_path or root.parent / "claude" / "plan" / "reports" / "demand-calibration.md"
    write_outputs(result, root, dest)
    return result


def main() -> int:
    from marketsim.core.config import default_config_dir

    root = default_config_dir()
    report = root.parent / "claude" / "plan" / "reports" / "demand-calibration.md"
    result = run_from_dir(root, report)
    print(f"Spearman={result.spearman:.4f}  max|basket-θ|={np.max(np.abs(result.basket - result.theta)):.3e}")
    print(f"wrote {root / 'wants.yaml'}, {root / 'buy_packages.yaml'}, {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
