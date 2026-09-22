"""T10.17 dashboard — determinism, self-containment, geometry, escaping, deviation gates."""

from __future__ import annotations

import re

import numpy as np
import pytest

from marketsim.viz.dashboard import DEVIATION_TOLERANCE, render
from marketsim.viz.recorder import Recording

MONTHS = 24
CODES = ("ENERGY", "MATERIALS", "AGRIFOOD")


def _recording(*, scale: float = 1.0, codes: tuple[str, ...] = CODES) -> Recording:
    """Synthetic run: no engine needed, so these stay fast and focused on rendering."""
    months = np.arange(1, MONTHS + 1, dtype=np.int64)
    wave = np.cos(np.linspace(0.0, 3.0, MONTHS))
    return Recording(
        codes=codes,
        months=months,
        ticks=months * 21,
        scalars={
            "gdp": 94.0 * scale + wave,
            "cpi": 1.0 * scale + wave * 0.01,
            "core_cpi": 1.0 * scale + wave * 0.005,
            "u": np.full(MONTHS, 0.05) + wave * 0.001,
            "r": np.full(MONTHS, 0.02),
            "pi12": wave * 0.002,
            "debt_gdp": np.full(MONTHS, 0.6),
            "w": np.full(MONTHS, 1.0),
        },
        matrices={
            "x": np.outer(np.ones(MONTHS), np.arange(1.0, len(codes) + 1.0)) + wave[:, None],
            "p": np.ones((MONTHS, len(codes))),
        },
        meta={"seed": 0, "state_hash": "a" * 64, "module_names": ["real_economy"]},
    )


def test_render_is_deterministic() -> None:
    rec = _recording()
    assert render(rec) == render(rec)


def test_page_is_self_contained_and_offline() -> None:
    page = render(_recording(), golden=_recording(scale=1.001), golden_label="baseline")
    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    assert "http://" not in page and "https://" not in page
    assert "<script src" not in page and "<link" not in page


def test_every_multi_series_chart_has_a_legend_and_a_table_view() -> None:
    page = render(_recording(), golden=_recording(scale=1.01), golden_label="baseline")
    # cpi/core_cpi share a plot, and every comparison overlay carries two series.
    assert page.count('class="viz-legend"') >= 1
    assert '<details class="viz-table">' in page
    # Values are never hover-gated: the table view repeats them.
    assert "<table>" in page


def test_text_never_wears_a_series_colour() -> None:
    page = render(_recording())
    assert re.search(r"<text[^>]*fill=\"var\(--series-", page) is None


def test_gridlines_are_solid() -> None:
    page = render(_recording())
    assert re.search(r"stroke=\"var\(--grid\)\"[^>]*stroke-dasharray", page) is None


@pytest.mark.parametrize("compare", [None, _recording(scale=1.02)])
def test_all_coordinates_stay_inside_the_viewbox(compare: Recording | None) -> None:
    page = render(_recording(), golden=compare)
    charts = re.findall(r'(<svg viewBox="0 0 \d+ \d+".*?</svg>)', page, re.S)
    assert charts
    for chart in charts:
        head = re.match(r'<svg viewBox="0 0 (\d+) (\d+)"', chart)
        assert head is not None
        width, height = int(head.group(1)), int(head.group(2))
        for attr in ("x", "y", "x1", "x2", "y1", "y2", "cx", "cy"):
            for raw in re.findall(rf'\b{attr}="(-?[\d.]+)"', chart):
                limit = width if attr in ("x", "x1", "x2", "cx") else height
                assert -0.5 <= float(raw) <= limit + 0.5
        for points in re.findall(r'points="([^"]+)"', chart):
            for pair in points.split():
                px, py = (float(v) for v in pair.split(","))
                assert -0.5 <= px <= width + 0.5
                assert -0.5 <= py <= height + 0.5


def test_deviation_verdict_uses_the_gate_tolerance() -> None:
    """Assert on the table cell, not the substring — "overlapping" contains "over"."""
    golden = _recording()
    assert DEVIATION_TOLERANCE["gdp"] == 0.01
    within = render(_recording(scale=1.0001), golden=golden, golden_label="baseline")
    assert "<td>within</td>" in within
    assert "<td>over</td>" not in within
    breached = render(_recording(scale=1.10), golden=golden, golden_label="baseline")
    assert "<td>over</td>" in breached


def test_identical_run_and_golden_show_zero_deviation() -> None:
    rec = _recording()
    page = render(rec, golden=_recording(), golden_label="baseline")
    assert "<td>over</td>" not in page
    assert "<td>within</td>" in page
    # series with no defined tolerance report a dash rather than a verdict
    assert "<td>—</td>" in page


def test_labels_are_escaped() -> None:
    rec = _recording(codes=("<script>alert(1)</script>", "B", "C"))
    page = render(rec, title="<b>hi</b>")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "<b>hi</b>" not in page


def test_empty_recording_renders_without_charts() -> None:
    page = render(Recording())
    assert page.startswith("<!doctype html>")
    assert "0 months" in page
