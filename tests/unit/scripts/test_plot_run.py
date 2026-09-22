"""T10.17 plot_run CLI — importable, --help, override parsing, no network, writes HTML."""

from __future__ import annotations

from pathlib import Path

import pytest

import plot_run

TICKS = 42  # two closed months


def _cli(argv: list[str]) -> int:
    try:
        return int(plot_run.main(argv))
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)


def test_help_exits_zero_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _block(*_a: object, **_k: object) -> None:
        raise AssertionError("network must not be used")

    monkeypatch.setattr("socket.create_connection", _block)
    assert _cli(["--help"]) == 0


def test_parse_overrides() -> None:
    parsed = plot_run.parse_overrides(["dynamics.banks.mode=passthrough", "world.scale=2.5"])
    assert parsed == {"dynamics.banks.mode": "passthrough", "world.scale": 2.5}
    with pytest.raises(ValueError, match="key=value"):
        plot_run.parse_overrides(["nonsense"])


def test_out_must_be_html(tmp_path: Path) -> None:
    assert _cli(["--run", "missing.npz", "--out", str(tmp_path / "run.png")]) == 2


def test_records_and_renders(tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _block(*_a: object, **_k: object) -> None:
        raise AssertionError("network must not be used")

    monkeypatch.setattr("socket.create_connection", _block)
    out = tmp_path / "run.html"
    code = _cli(
        [
            "--config",
            str(config_dir),
            "--ticks",
            str(TICKS),
            "--record",
            str(tmp_path / "run"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    assert (tmp_path / "run.npz").is_file()
    assert (tmp_path / "run.json").is_file()


def test_renders_an_existing_recording_against_a_golden(tmp_path: Path, config_dir: Path) -> None:
    golden = Path(__file__).resolve().parents[2] / "golden" / "data" / "aggregate_baseline.npz"
    assert _cli(
        ["--config", str(config_dir), "--ticks", str(TICKS), "--record", str(tmp_path / "r"), "--out", str(tmp_path / "a.html")]
    ) == 0
    code = _cli(
        [
            "--run",
            str(tmp_path / "r.npz"),
            "--golden",
            str(golden),
            "--golden-label",
            "baseline",
            "--out",
            str(tmp_path / "b.html"),
        ]
    )
    assert code == 0
    page = (tmp_path / "b.html").read_text(encoding="utf-8")
    assert "baseline" in page
    assert "Run vs baseline" in page
