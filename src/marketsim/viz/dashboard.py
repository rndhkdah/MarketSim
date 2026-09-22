"""Render a :class:`~marketsim.viz.recorder.Recording` as one self-contained HTML page (T10.17).

No plotting dependency: the marks are inline SVG built here, and the hover layer is a
small embedded script. Nothing is fetched at view time, so the page opens from disk and
satisfies the "never call the network" rule.

Output is a pure function of the recording — no timestamps, sorted iteration — so two
renders of the same run are byte-identical and a test can assert on the HTML.

Chart conventions follow the house data-viz rules: one y-axis per chart (never two
scales), a legend whenever two or more series share a plot, solid hairline gridlines,
2px lines with >=8px end markers ringed in the surface colour, text in ink tokens rather
than series colours, and a table view beside every section so no value is hover-gated.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.viz.recorder import SERIES_UNITS, Recording

# Presentation geometry (px). These are layout, not model parameters, so they live
# here rather than in config/*.yaml.
CHART_W = 380
CHART_H = 208
PAD_L = 56
PAD_R = 58  # room for the end-of-line direct label
PAD_T = 10
PAD_B = 30
SPARK_W = 104
SPARK_H = 28
Y_TICKS = 4
X_TICKS = 4
FLAT_PAD = 0.01  # half-window for a constant series, as a fraction of its level

# Gate P5-1 tolerances (05-phase5-firms.md §5.1): GDP 1 %, sector output 2 %, CPI 0.5 %.
DEVIATION_TOLERANCE: dict[str, float] = {"gdp": 0.01, "x": 0.02, "cpi": 0.005}

# Panels shown in the macro section: (series names on one plot, heading).
MACRO_PANELS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gdp",), "Real GDP (production)"),
    (("cpi", "core_cpi"), "Consumer prices"),
    (("u",), "Unemployment"),
    (("r",), "Policy rate"),
    (("debt_gdp",), "Government debt"),
    (("pi12",), "Inflation, 12-month"),
)

KPI_SERIES: tuple[tuple[str, str], ...] = (
    ("gdp", "Real GDP"),
    ("cpi", "Consumer prices"),
    ("u", "Unemployment"),
    ("r", "Policy rate"),
)

_STYLE = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --plane: #f9f9f7;
  --ink-1: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --ring: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--plane);
  color: var(--ink-1);
  padding: 24px 16px 48px;
  margin: 0;
  line-height: 1.45;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --plane: #0d0d0d;
    --ink-1: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --ring: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --plane: #0d0d0d;
  --ink-1: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --ring: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
}
.viz-wrap { max-width: 1200px; margin: 0 auto; }
.viz-head h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.viz-meta { color: var(--ink-2); font-size: 13px; margin: 0 0 20px; }
.viz-meta code { font-size: 12px; color: var(--muted); }
.viz-kpis { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
.viz-tile {
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px;
  padding: 12px 14px; min-width: 168px; flex: 1 1 168px;
}
.viz-tile .label { color: var(--ink-2); font-size: 12px; }
.viz-tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.viz-tile .delta { font-size: 12px; color: var(--ink-2); margin-top: 2px; }
.viz-section { margin: 0 0 28px; }
.viz-section > h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.viz-section > p.note { color: var(--ink-2); font-size: 13px; margin: 0 0 12px; }
.viz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
.viz-grid.dense { grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }
.viz-card {
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px; padding: 10px 12px 6px;
}
.viz-card h3 { font-size: 13px; font-weight: 600; margin: 0; }
.viz-card .unit { color: var(--muted); font-size: 11px; margin: 0 0 6px; }
.viz-legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 2px 0 4px; font-size: 12px; color: var(--ink-2); }
.viz-legend span { display: inline-flex; align-items: center; gap: 5px; }
.viz-legend i { display: inline-block; width: 14px; height: 2px; border-radius: 1px; }
.viz-chart { position: relative; }
.viz-chart svg { display: block; width: 100%; height: auto; }
.viz-tip {
  position: absolute; pointer-events: none; opacity: 0; transition: opacity .08s linear;
  background: var(--surface-1); border: 1px solid var(--ring); border-radius: 8px;
  padding: 6px 8px; font-size: 12px; color: var(--ink-2); box-shadow: 0 2px 10px rgba(0,0,0,.10);
  white-space: nowrap; z-index: 2;
}
.viz-tip b { color: var(--ink-1); font-weight: 600; font-variant-numeric: tabular-nums; }
.viz-tip .row { display: flex; align-items: center; gap: 6px; }
.viz-tip i { display: inline-block; width: 12px; height: 2px; border-radius: 1px; }
details.viz-table { margin-top: 10px; }
details.viz-table summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
details.viz-table table { border-collapse: collapse; margin-top: 8px; font-size: 12px; width: 100%; }
details.viz-table th, details.viz-table td {
  text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
}
details.viz-table th:first-child, details.viz-table td:first-child { text-align: left; }
details.viz-table th { color: var(--ink-2); font-weight: 600; }
.viz-news { font-size: 13px; color: var(--ink-2); margin: 0; padding-left: 18px; }
.viz-empty { color: var(--muted); font-size: 13px; }
"""

_SCRIPT = """
(function () {
  var data = JSON.parse(document.getElementById('viz-data').textContent);
  Object.keys(data).sort().forEach(function (id) {
    var spec = data[id];
    var wrap = document.getElementById(id);
    if (!wrap) { return; }
    var svg = wrap.querySelector('svg');
    var tip = wrap.querySelector('.viz-tip');
    var rule = svg.querySelector('.crosshair');
    function hide() { tip.style.opacity = 0; if (rule) { rule.setAttribute('opacity', 0); } }
    function show(evt) {
      var box = svg.getBoundingClientRect();
      var scale = spec.vw / box.width;
      var px = (evt.clientX - box.left) * scale;
      var best = 0, bestd = Infinity;
      for (var i = 0; i < spec.xs.length; i++) {
        var d = Math.abs(spec.xs[i] - px);
        if (d < bestd) { bestd = d; best = i; }
      }
      if (rule) {
        rule.setAttribute('x1', spec.xs[best]);
        rule.setAttribute('x2', spec.xs[best]);
        rule.setAttribute('opacity', 1);
      }
      while (tip.firstChild) { tip.removeChild(tip.firstChild); }
      var head = document.createElement('div');
      head.appendChild(document.createTextNode(spec.xlabel + ' ' + spec.months[best]));
      tip.appendChild(head);
      spec.series.forEach(function (s) {
        var row = document.createElement('div');
        row.className = 'row';
        var key = document.createElement('i');
        key.style.background = 'var(--series-' + s.slot + ')';
        var val = document.createElement('b');
        val.appendChild(document.createTextNode(s.text[best]));
        var name = document.createElement('span');
        name.appendChild(document.createTextNode(s.label));
        row.appendChild(key); row.appendChild(val); row.appendChild(name);
        tip.appendChild(row);
      });
      tip.style.opacity = 1;
      var left = (spec.xs[best] / spec.vw) * box.width + 12;
      if (left + tip.offsetWidth > box.width) { left = left - tip.offsetWidth - 24; }
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = '6px';
    }
    wrap.addEventListener('pointermove', show);
    wrap.addEventListener('pointerleave', hide);
    wrap.addEventListener('focus', function () { show({ clientX: svg.getBoundingClientRect().right }); });
    wrap.addEventListener('blur', hide);
  });
})();
"""


@dataclass(frozen=True)
class Series:
    """One plotted line. ``slot`` is a categorical palette slot (1-based, fixed order)."""

    label: str
    values: np.ndarray
    slot: int = 1
    dashed: bool = False


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _json_payload(specs: dict[str, Any]) -> str:
    """Serialise the hover data for a ``<script type="application/json">`` block.

    Series labels are untrusted (they come from whatever ``.npz`` is rendered), so the
    markup-significant characters are emitted as JSON escapes. ``</script>`` inside a
    label would otherwise close the element early and inject live markup; the escapes
    parse back to the identical string.
    """
    raw = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return raw.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _fmt(value: float) -> str:
    """Adaptive fixed-point. Units are never converted — the axis shows what was recorded."""
    if not math.isfinite(value):
        return "n/a"
    a = abs(value)
    if a >= 1000.0:
        return f"{value:,.0f}"
    if a >= 10.0:
        return f"{value:,.2f}"
    if a >= 1.0:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _nice_ticks(lo: float, hi: float, count: int = Y_TICKS) -> list[float]:
    """Round tick values spanning ``[lo, hi]`` on a 1/2/5 × 10^k ladder."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if raw <= mag * mult:
            step = mag * mult
            break
    else:  # pragma: no cover - the 10× rung always matches
        step = mag * 10.0
    first = math.ceil(lo / step) * step
    ticks: list[float] = []
    value = first
    while value <= hi + step * 1e-9 and len(ticks) <= count + 2:
        ticks.append(round(value, 12))
        value += step
    return ticks or [lo]


def _domain(series: list[Series], *, include_zero: bool) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s.values, dtype=float) for s in series]) if series else np.zeros(1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo <= abs(hi) * 1e-12:
        pad = max(abs(hi) * FLAT_PAD, 1e-9)
        return lo - pad, hi + pad
    margin = (hi - lo) * 0.08
    return lo - margin, hi + margin


def _line_chart(
    chart_id: str,
    heading: str,
    unit: str,
    months: np.ndarray,
    series: list[Series],
    *,
    include_zero: bool = False,
    band: float | None = None,
    compact: bool = False,
) -> tuple[str, dict[str, Any]]:
    """One line chart. Returns the HTML card and the hover spec for the embedded script."""
    height = CHART_H if not compact else int(CHART_H * 0.72)
    pad_l = PAD_L if not compact else 44
    pad_r = PAD_R if not compact else 14
    plot_w = CHART_W - pad_l - pad_r
    plot_h = height - PAD_T - PAD_B
    n = int(months.shape[0])
    if n == 0 or not series:
        return f'<div class="viz-card"><h3>{_esc(heading)}</h3><p class="viz-empty">no data</p></div>', {}

    lo, hi = _domain(series, include_zero=include_zero or band is not None)
    if band is not None:
        lo, hi = min(lo, -band * 1.3), max(hi, band * 1.3)

    def sx(i: int) -> float:
        return pad_l + (plot_w * (i / (n - 1)) if n > 1 else plot_w / 2.0)

    def sy(v: float) -> float:
        if not math.isfinite(v):
            return PAD_T + plot_h
        frac = (v - lo) / (hi - lo) if hi > lo else 0.5
        return PAD_T + plot_h - frac * plot_h

    parts: list[str] = [
        f'<div class="viz-card"><h3>{_esc(heading)}</h3><p class="unit">{_esc(unit)}</p>'
    ]
    if len(series) > 1:
        keys = "".join(
            f'<span><i style="background:var(--series-{s.slot})"></i>{_esc(s.label)}</span>' for s in series
        )
        parts.append(f'<div class="viz-legend">{keys}</div>')
    parts.append(f'<div class="viz-chart" id="{_esc(chart_id)}" tabindex="0">')
    svg: list[str] = [
        f'<svg viewBox="0 0 {CHART_W} {height}" role="img" '
        f'aria-label="{_esc(heading)} — {_esc(unit)}">'
    ]

    for tick in _nice_ticks(lo, hi):
        y = sy(tick)
        svg.append(
            f'<line x1="{pad_l}" y1="{y:.2f}" x2="{pad_l + plot_w}" y2="{y:.2f}" '
            f'stroke="var(--grid)" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{pad_l - 6}" y="{y + 3.5:.2f}" text-anchor="end" font-size="10" '
            f'fill="var(--muted)" style="font-variant-numeric:tabular-nums">{_esc(_fmt(tick))}</text>'
        )
    if band is not None:
        top, bottom = sy(band), sy(-band)
        svg.append(
            f'<rect x="{pad_l}" y="{top:.2f}" width="{plot_w}" height="{max(bottom - top, 0):.2f}" '
            f'fill="var(--axis)" opacity="0.14" />'
        )
    if lo < 0.0 < hi:
        zero = sy(0.0)
        svg.append(
            f'<line x1="{pad_l}" y1="{zero:.2f}" x2="{pad_l + plot_w}" y2="{zero:.2f}" '
            f'stroke="var(--axis)" stroke-width="1" />'
        )
    baseline = PAD_T + plot_h
    svg.append(
        f'<line x1="{pad_l}" y1="{baseline}" x2="{pad_l + plot_w}" y2="{baseline}" '
        f'stroke="var(--axis)" stroke-width="1" />'
    )
    step = max(1, (n - 1) // X_TICKS) if n > 1 else 1
    for i in range(0, n, step):
        svg.append(
            f'<text x="{sx(i):.2f}" y="{height - 10}" text-anchor="middle" font-size="10" '
            f'fill="var(--muted)" style="font-variant-numeric:tabular-nums">{int(months[i])}</text>'
        )

    ends: list[tuple[float, str, int]] = []
    for s in series:
        values = np.asarray(s.values, dtype=float)
        pts = " ".join(f"{sx(i):.2f},{sy(float(values[i])):.2f}" for i in range(n))
        dash = ' stroke-dasharray="6 3"' if s.dashed else ""
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="var(--series-{s.slot})" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash} />'
        )
        ex, ey = sx(n - 1), sy(float(values[-1]))
        svg.append(
            f'<circle cx="{ex:.2f}" cy="{ey:.2f}" r="4" fill="var(--series-{s.slot})" '
            f'stroke="var(--surface-1)" stroke-width="2" />'
        )
        ends.append((ey, _fmt(float(values[-1])), s.slot))

    # Direct end labels only when they do not collide; otherwise legend + tooltip carry it.
    if not compact and (len(ends) == 1 or abs(ends[0][0] - ends[-1][0]) >= 12.0):
        for ey, text, _slot in ends:
            svg.append(
                f'<text x="{pad_l + plot_w + 8}" y="{ey + 3.5:.2f}" font-size="10" fill="var(--ink-2)" '
                f'style="font-variant-numeric:tabular-nums">{_esc(text)}</text>'
            )
    svg.append(
        f'<line class="crosshair" x1="{pad_l}" y1="{PAD_T}" x2="{pad_l}" y2="{baseline}" '
        f'stroke="var(--axis)" stroke-width="1" opacity="0" />'
    )
    svg.append("</svg>")
    parts.append("".join(svg))
    parts.append('<div class="viz-tip"></div></div></div>')

    spec = {
        "vw": CHART_W,
        "xlabel": "month",
        "months": [int(m) for m in months.tolist()],
        "xs": [round(sx(i), 2) for i in range(n)],
        "series": [
            {
                "label": s.label,
                "slot": s.slot,
                "text": [_fmt(float(v)) for v in np.asarray(s.values, dtype=float).tolist()],
            }
            for s in series
        ],
    }
    return "".join(parts), spec


def _sparkline(values: np.ndarray) -> str:
    """A 12-point trend for a stat tile. Recessive: one hairline, no axis."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return ""
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo if hi > lo else max(abs(hi) * FLAT_PAD, 1e-9)
    n = arr.size
    pts = " ".join(
        f"{SPARK_W * i / (n - 1):.2f},{SPARK_H - (float(arr[i]) - lo) / span * SPARK_H:.2f}"
        for i in range(n)
    )
    return (
        f'<svg viewBox="0 0 {SPARK_W} {SPARK_H}" width="{SPARK_W}" height="{SPARK_H}" aria-hidden="true">'
        f'<polyline points="{pts}" fill="none" stroke="var(--series-1)" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" /></svg>'
    )


def _kpi_row(rec: Recording) -> str:
    tiles: list[str] = []
    for name, label in KPI_SERIES:
        values = rec.scalars.get(name)
        if values is None or values.size == 0:
            continue
        first, last = float(values[0]), float(values[-1])
        delta = last - first
        sign = "+" if delta >= 0 else "−"
        tiles.append(
            f'<div class="viz-tile"><div class="label">{_esc(label)}</div>'
            f'<div class="value">{_esc(_fmt(last))}</div>'
            f'<div class="delta">{sign}{_esc(_fmt(abs(delta)))} since month {int(rec.months[0])}</div>'
            f"{_sparkline(values)}</div>"
        )
    return f'<div class="viz-kpis">{"".join(tiles)}</div>' if tiles else ""


def _table(caption: str, headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="viz-table"><summary>{_esc(caption)}</summary>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>"
    )


def _macro_section(rec: Recording, specs: dict[str, Any]) -> str:
    cards: list[str] = []
    shown: list[str] = []
    for names, heading in MACRO_PANELS:
        present = [n for n in names if n in rec.scalars]
        if not present:
            continue
        series = [
            Series(label=n, values=rec.scalars[n], slot=i + 1) for i, n in enumerate(present)
        ]
        chart_id = f"macro-{present[0]}"
        html_card, spec = _line_chart(
            chart_id,
            heading,
            SERIES_UNITS.get(present[0], ""),
            rec.months,
            series,
            include_zero=present[0] in ("pi12", "govt_balance", "trade_balance"),
        )
        cards.append(html_card)
        shown.extend(present)
        if spec:
            specs[chart_id] = spec
    if not cards:
        return ""
    rows = [
        [str(int(m))] + [_fmt(float(rec.scalars[n][i])) for n in shown]
        for i, m in enumerate(rec.months.tolist())
    ]
    table = _table("Table view — macro series by month", ["month", *shown], rows)
    return (
        '<section class="viz-section"><h2>Macro</h2>'
        '<p class="note">One month per point; one measure per axis.</p>'
        f'<div class="viz-grid">{"".join(cards)}</div>{table}</section>'
    )


def _compare_section(rec: Recording, golden: Recording, label: str, specs: dict[str, Any]) -> str:
    shared = rec.shared_series(golden)
    n = min(len(rec), len(golden))
    if not shared or n == 0:
        return (
            '<section class="viz-section"><h2>Run vs '
            f"{_esc(label)}</h2><p class=\"viz-empty\">no overlapping series</p></section>"
        )
    months = rec.months[:n]
    cards: list[str] = []
    rows: list[list[str]] = []
    for name in shared:
        run_v = np.asarray(rec.scalars[name][:n], dtype=float)
        gold_v = np.asarray(golden.scalars[name][:n], dtype=float)
        overlay_id = f"cmp-{name}"
        card, spec = _line_chart(
            overlay_id,
            f"{name} — run vs {label}",
            SERIES_UNITS.get(name, ""),
            months,
            [
                Series(label="run", values=run_v, slot=1),
                Series(label=label, values=gold_v, slot=2, dashed=True),
            ],
        )
        cards.append(card)
        if spec:
            specs[overlay_id] = spec

        denom = np.where(np.abs(gold_v) > 1e-12, np.abs(gold_v), 1.0)
        rel = (run_v - gold_v) / denom
        dev_id = f"dev-{name}"
        tol = DEVIATION_TOLERANCE.get(name)
        dev_card, dev_spec = _line_chart(
            dev_id,
            f"{name} — relative deviation",
            "fraction of the baseline (0 = identical)",
            months,
            [Series(label="run − " + label, values=rel, slot=1)],
            include_zero=True,
            band=tol,
        )
        cards.append(dev_card)
        if dev_spec:
            specs[dev_id] = dev_spec
        worst = float(np.max(np.abs(rel))) if rel.size else 0.0
        rows.append(
            [
                name,
                _fmt(float(run_v[-1])),
                _fmt(float(gold_v[-1])),
                _fmt(worst),
                "—" if tol is None else _fmt(tol),
                "—" if tol is None else ("within" if worst <= tol else "over"),
            ]
        )
    table = _table(
        "Table view — deviation summary",
        ["series", "run (last)", f"{label} (last)", "max |rel dev|", "tolerance", "verdict"],
        rows,
    )
    return (
        f'<section class="viz-section"><h2>Run vs {_esc(label)}</h2>'
        f'<p class="note">Shaded band is the gate tolerance where one is defined; '
        f"{n} overlapping months.</p>"
        f'<div class="viz-grid">{"".join(cards)}</div>{table}</section>'
    )


def _sectors_section(rec: Recording, name: str, heading: str, specs: dict[str, Any]) -> str:
    matrix = rec.matrices.get(name)
    if matrix is None or matrix.size == 0 or matrix.ndim < 2:
        return ""
    flat = matrix.reshape(matrix.shape[0], -1)
    codes = rec.codes if len(rec.codes) == flat.shape[1] else tuple(f"s{i}" for i in range(flat.shape[1]))
    cards: list[str] = []
    rows: list[list[str]] = []
    for j, code in enumerate(codes):
        chart_id = f"sector-{name}-{code}"
        column = flat[:, j]
        card, spec = _line_chart(
            chart_id,
            str(code),
            "",
            rec.months,
            [Series(label=str(code), values=column, slot=1)],
            compact=True,
        )
        cards.append(card)
        if spec:
            specs[chart_id] = spec
        rows.append(
            [
                str(code),
                _fmt(float(column[0])),
                _fmt(float(column[-1])),
                _fmt(float(column.min())),
                _fmt(float(column.max())),
            ]
        )
    table = _table(
        f"Table view — {heading} per sector",
        ["sector", "first", "last", "min", "max"],
        rows,
    )
    return (
        f'<section class="viz-section"><h2>{_esc(heading)}</h2>'
        '<p class="note">Small multiples, one sector per panel, shared measure.</p>'
        f'<div class="viz-grid dense">{"".join(cards)}</div>{table}</section>'
    )


def _news_section(rec: Recording) -> str:
    if not rec.news:
        return ""
    items = "".join(
        f"<li>tick {row.tick} · {_esc(row.category)} · {_esc(row.headline)}"
        f"{' (rumour)' if row.is_rumour else ''}</li>"
        for row in rec.news
    )
    return (
        '<section class="viz-section"><h2>Events</h2>'
        f'<ul class="viz-news">{items}</ul></section>'
    )


def render(
    rec: Recording,
    *,
    golden: Recording | None = None,
    golden_label: str = "golden",
    title: str = "MarketSim run",
) -> str:
    """Render one self-contained HTML page. Pure: same recording in, same bytes out."""
    specs: dict[str, Any] = {}
    sections = [_macro_section(rec, specs)]
    if golden is not None:
        sections.append(_compare_section(rec, golden, golden_label, specs))
    sections.append(_sectors_section(rec, "x", "Sector output", specs))
    sections.append(_sectors_section(rec, "p", "Sector prices", specs))
    sections.append(_news_section(rec))

    meta = rec.meta
    bits = [f"{len(rec)} months"]
    if "seed" in meta:
        bits.append(f"seed {meta['seed']}")
    if meta.get("module_names"):
        bits.append("modules: " + ", ".join(str(m) for m in meta["module_names"]))
    state_hash = str(meta.get("state_hash", ""))
    hash_html = f' · <code>{_esc(state_hash[:16])}</code>' if state_hash else ""

    body = (
        '<div class="viz-root"><div class="viz-wrap">'
        f'<div class="viz-head"><h1>{_esc(title)}</h1>'
        f'<p class="viz-meta">{_esc(" · ".join(bits))}{hash_html}</p></div>'
        f"{_kpi_row(rec)}{''.join(s for s in sections if s)}"
        "</div></div>"
    )
    payload = _json_payload(specs)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head><body>"
        f"{body}"
        f'<script type="application/json" id="viz-data">{payload}</script>'
        f"<script>{_SCRIPT}</script>"
        "</body></html>\n"
    )


def write_dashboard(
    path: str | Path,
    rec: Recording,
    *,
    golden: Recording | None = None,
    golden_label: str = "golden",
    title: str = "MarketSim run",
) -> Path:
    """Render and write the page. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rec, golden=golden, golden_label=golden_label, title=title), encoding="utf-8")
    return out
