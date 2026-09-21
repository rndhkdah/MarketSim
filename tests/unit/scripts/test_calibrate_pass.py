"""T8.04 calibrate_* CLIs — importable, dry-run / --help, no network, no yaml writes."""

from __future__ import annotations

from pathlib import Path

import pytest

import calibrate_events
import calibrate_moments
from calibrate_events import (
    IRF_SAFE_COST_PUSH,
    T224_COST_PUSH_Z,
    T224_CPI_12M,
    event_z_table,
    linear_cpi_implied_z,
    map_historic_cost_push,
    naive_log_price_z,
    oil_1973_multiple,
)
from calibrate_moments import (
    T224_ANCHOR_PROPOSED,
    T224_PHI_PROPOSED,
    T224_SD_C_OVER_GDP,
    T224_SD_I_OVER_GDP,
    propose_moment_gains,
)


def _cli(fn, argv: list[str]) -> int:
    try:
        return int(fn(argv))
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
        return int(code)


def test_scripts_importable() -> None:
    assert callable(calibrate_moments.main)
    assert callable(calibrate_events.main)
    assert callable(propose_moment_gains)
    assert callable(event_z_table)


def test_help_and_dry_run_exit_zero_without_network(monkeypatch) -> None:
    def _block(*_a, **_k):
        raise AssertionError("network must not be used")

    monkeypatch.setattr("socket.create_connection", _block)
    for fn in (calibrate_moments.main, calibrate_events.main):
        assert _cli(fn, ["--help"]) == 0
        assert _cli(fn, ["--dry-run"]) == 0


def test_oil_map_is_irf_safe_band_not_ln4() -> None:
    multiple = oil_1973_multiple()
    assert multiple == pytest.approx(11.65 / 2.90)
    naive = naive_log_price_z(4.0)
    assert naive == pytest.approx(1.386, abs=5e-3)
    naive_m, lo, hi = map_historic_cost_push(multiple, seed_z=1.25)
    assert naive_m == pytest.approx(naive_log_price_z(multiple))
    assert (lo, hi) == IRF_SAFE_COST_PUSH
    assert lo == pytest.approx(0.30)
    assert hi == pytest.approx(0.40)
    assert naive_m > hi  # ln 4 is outside the safe band
    assert T224_COST_PUSH_Z == pytest.approx(0.30)
    assert T224_CPI_12M == pytest.approx(0.72)


def test_small_cost_push_seed_kept() -> None:
    _n, lo, hi = map_historic_cost_push(seed_z=0.20)
    assert lo == pytest.approx(0.20)
    assert hi == pytest.approx(0.20)


def test_linear_cpi_match_is_recorded_not_the_oil_proposal() -> None:
    z = linear_cpi_implied_z(0.091)
    assert z == pytest.approx(0.30 * 0.091 / 0.72)
    assert z < IRF_SAFE_COST_PUSH[0]


def test_event_table_oil_row() -> None:
    rows = {r.event_id: r for r in event_z_table()}
    oil = rows["oil_embargo_1973"]
    assert "1.25" in oil.seed_z
    assert "DO NOT USE" in oil.naive_z
    assert "0.30" in oil.proposed_z and "0.40" in oil.proposed_z
    assert oil.applied is False


def test_moment_proposals_match_t224_and_are_not_applied() -> None:
    rows = {r.key: r for r in propose_moment_gains()}
    assert rows["edges.capex.coefficients.phi_accelerator"].current == pytest.approx(1.2)
    assert rows["edges.capex.coefficients.phi_accelerator"].proposed == T224_PHI_PROPOSED
    assert rows["dynamics.expectations.anchor_growth"].proposed == T224_ANCHOR_PROPOSED
    assert rows["dynamics.households.alpha1"].proposed == pytest.approx(0.70)
    assert rows["rejected.linear_phi_to_close_I_gap"].proposed != T224_PHI_PROPOSED
    assert all(r.applied is False for r in rows.values())
    assert min(T224_SD_I_OVER_GDP) < 3.0
    assert max(T224_SD_C_OVER_GDP) < 1.0


def test_refuse_yaml_out(tmp_path: Path) -> None:
    yaml_path = tmp_path / "proposed.yaml"
    assert calibrate_moments.main(["--out", str(yaml_path)]) == 2
    assert not yaml_path.exists()
    assert calibrate_events.main(["--out", str(tmp_path / "events.yml")]) == 2
    json_path = tmp_path / "proposed.json"
    assert calibrate_moments.main(["--out", str(json_path)]) == 0
    assert json_path.exists()
    assert "phi_accelerator" in json_path.read_text(encoding="utf-8")
