"""T8.04 final-demand concordance — local maps only, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fetch_bea_io import (
    CODES,
    COVERAGE_FLOOR,
    DEFAULT_NIPA_CONCORDANCE,
    FD_KEYS,
    SEED_FD_WEIGHTS,
    SEED_IO_TABLE,
    aggregate_final_demand,
    assert_concordance_normalized,
    compose_concordance,
    example_fd_arithmetic,
    extract_nipa_final_demand,
    fd_component_weights,
    identity_concordance,
    main,
)
from marketsim.layer1.io import load_io


def test_default_concordance_rows_sum_to_one_and_land_on_codes() -> None:
    composed = compose_concordance(DEFAULT_NIPA_CONCORDANCE)
    assert_concordance_normalized(composed, codes=CODES)
    assert composed
    for _src, weights in composed.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert set(weights).issubset(set(CODES))


def test_identity_concordance_is_18_by_18_simplex() -> None:
    ident = identity_concordance()
    assert list(ident) == list(CODES)
    assert len(ident) == 18
    assert_concordance_normalized(ident)
    for code, weights in ident.items():
        assert weights == {code: 1.0}


def test_aggregate_final_demand_shape_weights_and_example_arithmetic() -> None:
    ex = example_fd_arithmetic()
    fd = ex["fd_matrix"]
    assert fd.shape == (4, 18)
    assert list(ex["weights"]) == list(FD_KEYS)
    assert abs(sum(ex["weights"].values()) - 1.0) < 1e-12
    assert ex["coverage"] == pytest.approx(1.0)
    assert ex["weights"]["HOUSEHOLD"] == pytest.approx(100.0 / 140.0)
    assert ex["weights"]["INVESTMENT"] == pytest.approx(20.0 / 140.0)
    assert ex["weights"]["GOVT"] == pytest.approx(15.0 / 140.0)
    assert ex["weights"]["EXPORTS"] == pytest.approx(5.0 / 140.0)
    staples = list(CODES).index("STAPLES")
    autos = list(CODES).index("AUTOS")
    capgoods = list(CODES).index("CAPGOODS")
    assert fd[0, staples] == pytest.approx(60.0)
    assert fd[0, autos] == pytest.approx(20.0)
    assert fd[3, autos] == pytest.approx(5.0)
    assert fd[1, capgoods] == pytest.approx(15.0)
    assert fd[2, capgoods] == pytest.approx(5.0)


def test_seed_identity_recovers_hand_set_fd_weights(io) -> None:
    scale = 100.0
    pce = {c: scale * SEED_FD_WEIGHTS["HOUSEHOLD"] * float(io.final_demand["HOUSEHOLD"][i]) for i, c in enumerate(CODES)}
    inv = {c: scale * SEED_FD_WEIGHTS["INVESTMENT"] * float(io.final_demand["INVESTMENT"][i]) for i, c in enumerate(CODES)}
    gov = {c: scale * SEED_FD_WEIGHTS["GOVT"] * float(io.final_demand["GOVT"][i]) for i, c in enumerate(CODES)}
    exp = {c: scale * SEED_FD_WEIGHTS["EXPORTS"] * float(io.final_demand["EXPORTS"][i]) for i, c in enumerate(CODES)}
    fd, weights, coverage = aggregate_final_demand(pce, inv, gov, exp, identity_concordance())
    assert fd.shape == (4, 18)
    assert coverage == pytest.approx(1.0)
    for key in FD_KEYS:
        assert weights[key] == pytest.approx(SEED_FD_WEIGHTS[key])
        row = fd[list(FD_KEYS).index(key)]
        assert row.sum() == pytest.approx(scale * SEED_FD_WEIGHTS[key])
        assert np.allclose(row / row.sum(), io.final_demand[key])


def test_fd_component_weights_sum_to_one() -> None:
    w = fd_component_weights({"HOUSEHOLD": 62.0, "INVESTMENT": 19.0, "GOVT": 13.0, "EXPORTS": 6.0})
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["HOUSEHOLD"] == pytest.approx(0.62)


def test_unmapped_source_lowers_coverage_but_weights_use_mapped_dollars() -> None:
    pce = {"Food": 80.0, "UnmappedNipaLine": 20.0}
    inv = {"Machinery": 10.0}
    gov = {"Health": 10.0}
    exp = {"Autos": 10.0}
    conc = {"Food": "STAPLES", "Machinery": "CAPGOODS", "Health": "HEALTH", "Autos": "AUTOS"}
    _fd, weights, coverage = aggregate_final_demand(pce, inv, gov, exp, conc)
    assert coverage == pytest.approx(110.0 / 130.0)
    assert coverage < COVERAGE_FLOOR
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["HOUSEHOLD"] == pytest.approx(80.0 / 110.0)


def test_extract_from_attrs_and_named_columns() -> None:
    idx = ["Food and beverages purchased for off-premises consumption", "Motor vehicles and parts"]
    df = pd.DataFrame(
        {
            "Personal consumption expenditures": [70.0, 30.0],
            "Private fixed investment": [0.0, 20.0],
            "Government consumption expenditures and gross investment": [10.0, 0.0],
            "Exports of goods and services": [5.0, 15.0],
        },
        index=idx,
    )
    pce, inv, gov, exp = extract_nipa_final_demand(df)
    assert pce[idx[0]] == pytest.approx(70.0)
    assert inv[idx[1]] == pytest.approx(20.0)
    assert gov[idx[0]] == pytest.approx(10.0)
    assert exp[idx[1]] == pytest.approx(15.0)

    tiny = pd.DataFrame([[1.0]], index=["x"], columns=["z"])
    tiny.attrs["pce"] = {"Food": 1.0}
    tiny.attrs["investment"] = {"Machinery": 1.0}
    tiny.attrs["government"] = {"Health": 1.0}
    tiny.attrs["exports"] = {"Autos": 1.0}
    a, b, c, d = extract_nipa_final_demand(tiny)
    assert a == {"Food": 1.0}
    assert b == {"Machinery": 1.0}
    assert c == {"Health": 1.0}
    assert d == {"Autos": 1.0}


def _cli(argv: list[str]) -> int:
    try:
        return int(main(argv))
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code
        return int(code)


def test_cli_dry_run_and_help_exit_zero_without_network(monkeypatch) -> None:
    def _block(*_a, **_k):
        raise AssertionError("network must not be used")

    monkeypatch.setattr("socket.create_connection", _block)
    assert _cli(["--help"]) == 0
    assert _cli(["--dry-run"]) == 0
    assert _cli([]) == 2
    assert _cli(["--xlsx", "/no/such/use.xlsx"]) == 2
    assert _cli(["--fd-xlsx", "/no/such/fd.xlsx"]) == 2


def test_fd_xlsx_does_not_touch_seed_io(tmp_path: Path, monkeypatch) -> None:
    idx = ["Food", "Autos"]
    df = pd.DataFrame(
        {
            "Personal consumption expenditures": [60.0, 20.0],
            "Private fixed investment": [15.0, 5.0],
            "Government consumption expenditures and gross investment": [10.0, 5.0],
            "Exports of goods and services": [0.0, 5.0],
        },
        index=idx,
    )
    df.attrs["fd_concordance"] = {
        "Food": "STAPLES",
        "Autos": "AUTOS",
    }
    monkeypatch.setattr("fetch_bea_io._load_xlsx", lambda _p: df)
    before = SEED_IO_TABLE.read_text(encoding="utf-8") if SEED_IO_TABLE.exists() else None
    fd_xlsx = tmp_path / "fd.xlsx"
    fd_xlsx.write_bytes(b"placeholder")
    out = tmp_path / "fd_proposed.npz"
    rc = main(["--fd-xlsx", str(fd_xlsx), "--fd-out", str(out)])
    assert rc == 0
    assert out.exists()
    loaded = np.load(out)
    assert loaded["fd_matrix"].shape == (4, 18)
    if before is not None:
        assert SEED_IO_TABLE.read_text(encoding="utf-8") == before
    assert main(["--fd-xlsx", str(fd_xlsx), "--fd-out", str(SEED_IO_TABLE)]) == 2
    assert SEED_IO_TABLE.read_text(encoding="utf-8") == before


def test_load_io_still_sees_seed_weights() -> None:
    io = load_io(SEED_IO_TABLE)
    assert io.fd_weights["HOUSEHOLD"] == pytest.approx(0.62)
    assert io.source == "seed"
