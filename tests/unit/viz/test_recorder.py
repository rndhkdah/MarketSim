"""T10.16 run recorder — hash neutrality, monthly rows, round trip, golden reading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marketsim.real.economy import make_real_world
from marketsim.viz.recorder import (
    ECONOMY_SERIES,
    MATRIX_SERIES,
    SCALAR_SERIES,
    SCHEMA_VERSION,
    SERIES_UNITS,
    read_run,
    read_series,
    record_run,
    write_run,
)

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden" / "data"
TICKS = 63  # three closed months (21 days each)
MONTHS = 3


def test_recording_does_not_change_the_state_hash(config_dir: Path) -> None:
    """The recorder reads only — a recorded run must hash like an unrecorded one."""
    recorded = make_real_world(config_dir, seed=3, check_sfc=True)
    record_run(recorded, TICKS)
    plain = make_real_world(config_dir, seed=3, check_sfc=True)
    plain.step(TICKS)
    assert recorded.state_hash() == plain.state_hash()
    assert "recorder" not in recorded.to_state()["modules"]


def test_one_row_per_closed_month_with_units(config_dir: Path) -> None:
    world = make_real_world(config_dir, seed=0, check_sfc=True)
    rec = record_run(world, TICKS)
    assert len(rec) == MONTHS
    assert rec.months.tolist() == [1, 2, 3]
    assert rec.ticks.tolist() == [21, 42, 63]
    for name, _source, _unit in SCALAR_SERIES:
        assert name in rec.scalars and rec.scalars[name].shape == (MONTHS,)
        assert SERIES_UNITS[name]
    for name, _unit in ECONOMY_SERIES:
        assert name in rec.scalars and np.all(np.isfinite(rec.scalars[name]))
    for name, _unit in MATRIX_SERIES:
        assert rec.matrices[name].shape == (MONTHS, len(rec.codes))
    assert rec.meta["schema_version"] == SCHEMA_VERSION
    assert rec.meta["n_months"] == MONTHS


def test_series_match_the_step_month_record(config_dir: Path) -> None:
    """``gdp``/``cpi``/``x`` are the same numbers the goldens are built from."""
    world = make_real_world(config_dir, seed=0, check_sfc=True)
    rec = record_run(world, TICKS)
    other = make_real_world(config_dir, seed=0, check_sfc=True)
    eco = next(m for m in other.modules if getattr(m, "name", None) == "real_economy")
    for i in range(MONTHS):
        row = eco.step_month()
        assert rec.scalars["gdp"][i] == row["gdp"]
        assert rec.scalars["cpi"][i] == row["cpi"]
        assert rec.scalars["u"][i] == row["U"]
        assert rec.scalars["r"][i] == row["r"]
        assert np.array_equal(rec.matrices["x"][i], row["x"])


def test_write_then_read_round_trips(config_dir: Path, tmp_path: Path) -> None:
    world = make_real_world(config_dir, seed=1, check_sfc=True)
    rec = record_run(world, TICKS)
    npz_path, json_path = write_run(tmp_path / "run", rec)
    assert npz_path.is_file() and json_path.is_file()
    back = read_run(tmp_path / "run")
    assert back.codes == rec.codes
    assert back.months.tolist() == rec.months.tolist()
    assert back.series_names() == rec.series_names()
    for name in rec.series_names():
        assert np.array_equal(back.scalars[name], rec.scalars[name])
    for name in rec.matrix_names():
        assert np.array_equal(back.matrices[name], rec.matrices[name])
    assert back.meta["state_hash"] == rec.meta["state_hash"]


def test_reads_the_committed_goldens() -> None:
    """The reader is the comparison path, so it must load the shipped baselines unchanged."""
    baseline = read_series(GOLDEN_DIR / "aggregate_baseline.npz")
    assert len(baseline) == 120
    assert {"gdp", "cpi", "u", "r", "w"} <= set(baseline.series_names())
    assert baseline.matrices["x"].shape == (120, 18)
    assert len(baseline.codes) == 18

    r1 = read_series(GOLDEN_DIR / "r1_phase2.npz", prefix="ns_")
    assert len(r1) == 36
    assert {"gdp", "cpi", "u"} <= set(r1.series_names())


def test_recording_shares_series_with_the_baseline(config_dir: Path) -> None:
    world = make_real_world(config_dir, seed=0, check_sfc=True)
    rec = record_run(world, TICKS)
    baseline = read_series(GOLDEN_DIR / "aggregate_baseline.npz")
    assert rec.shared_series(baseline) == ("cpi", "gdp", "r", "u", "w")


def test_no_economy_records_nothing(config_dir: Path) -> None:
    from marketsim.world import World

    world = World.create(config_dir, seed=0, modules=[])
    rec = record_run(world, 21)
    assert len(rec) == 0
    assert rec.series_names() == ()
