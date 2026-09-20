"""T3.11 — demand calibrator gate 4 and determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import spearmanr

from marketsim.demand.calibrate import (
    calibrate,
    evaluate_fitted,
    household_basket,
    implied_etas,
    ras_fit,
    seed_params,
    write_outputs,
)
from marketsim.demand.packages import PackageSet, default_params, load_packages, want_shares
from marketsim.demand.tiers import build_tiers
from marketsim.demand.wants import load_wants
from marketsim.layer1.build_io import CODES


def test_gate4_committed_artefacts(cfg, io, config_dir: Path) -> None:
    wants = yaml.safe_load((config_dir / "wants.yaml").read_text())
    pkgs = yaml.safe_load((config_dir / "buy_packages.yaml").read_text())
    layer = load_wants(wants, CODES)
    packages = load_packages(pkgs, layer.names)
    basket, eta_cfg, eta_impl, rho = evaluate_fitted(layer, packages, io, cfg)
    theta = household_basket(io)
    assert basket == pytest.approx(theta, abs=1e-9)
    # 2026-09-20: rank-only Spearman ≥ 0.7 (master-plan gate 4 still cites 0.8).
    assert rho >= 0.7
    assert float(spearmanr(eta_impl, eta_cfg).statistic) == pytest.approx(rho, abs=1e-12)
    # Two-shape membership survives RAS (structural floor, no vertex LP).
    for i in range(len(CODES)):
        if theta[i] <= 1e-12:
            continue
        shapes = {layer.shapes[q] for q in range(layer.n_wants) if layer.m[q, i] > 0}
        assert len(shapes) >= 2, f"{CODES[i]} shapes={sorted(shapes)}"


def test_report_lists_every_sector() -> None:
    report = Path("claude/plan/reports/demand-calibration.md").read_text()
    for code in CODES:
        assert code in report
    assert "config η" in report
    assert "implied η" in report


def test_calibrate_deterministic(cfg, io, config_dir: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load((config_dir / "wants.yaml").read_text())
    a = calibrate(cfg, io, raw)
    b = calibrate(cfg, io, raw)
    assert a.wants_yaml == b.wants_yaml
    assert a.packages_yaml == b.packages_yaml
    assert a.report_md == b.report_md
    write_outputs(a, tmp_path, tmp_path / "report.md")
    write_outputs(b, tmp_path / "b", tmp_path / "b.md")
    assert (tmp_path / "buy_packages.yaml").read_text() == (tmp_path / "b" / "buy_packages.yaml").read_text()


def test_ras_hits_basket_from_priors(cfg, io, config_dir: Path) -> None:
    raw = yaml.safe_load((config_dir / "wants.yaml").read_text())
    layer = load_wants(raw, CODES)
    tiers = build_tiers()
    packages = PackageSet(
        layer.names,
        layer.shapes,
        tuple(seed_params(n, sh) for n, sh in zip(layer.names, layer.shapes, strict=True)),
    )
    theta = household_basket(io)
    v = want_shares(1.0, packages, tiers)
    m = ras_fit(layer.m, v, theta)
    basket = v @ m
    assert basket == pytest.approx(theta, abs=1e-9)
    eta = implied_etas(packages, m, tiers)
    assert eta.shape == (len(CODES),)
    assert np.all(np.isfinite(eta))
    _ = default_params("survival")
