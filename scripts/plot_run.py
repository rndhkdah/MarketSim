"""Record a run and render it as one self-contained HTML page (T10.16 / T10.17).

Two modes:

    python scripts/plot_run.py --ticks 2520 --out run.html
    python scripts/plot_run.py --run out/run.npz --golden tests/golden/data/aggregate_baseline.npz \
        --out run.html

``--ticks`` is simulated days (21 = 1 month, 252 = 1 year). No network, no plotting
dependency; the page opens from disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marketsim.core.config import default_config_dir, load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy
from marketsim.viz.dashboard import write_dashboard
from marketsim.viz.recorder import Recording, read_run, read_series, record_run, write_run
from marketsim.world import World


def parse_overrides(pairs: list[str]) -> dict[str, object]:
    """``["dynamics.banks.mode=passthrough"]`` -> dotted-path overrides (T0.03).

    Values parse as JSON where possible (``0.9`` -> float, ``true`` -> bool), else stay
    strings. The committed goldens are recorded with ``dynamics.banks.mode=passthrough``.
    """
    out: dict[str, object] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--override expects key=value, got {pair!r}")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def build_recording(
    config_dir: Path,
    *,
    ticks: int,
    seed: int,
    check_sfc: bool = True,
    pi_star: float = 0.0,
    overrides: dict[str, object] | None = None,
) -> Recording:
    """Step a real-economy world for ``ticks`` days and return the recording.

    Mirrors ``make_real_world`` but threads ``overrides`` through both the economy and
    the ``World``; switch to ``World.create`` with no ``modules`` argument once T10.03
    makes the assembled world the default.
    """
    cfg = load_config(config_dir, overrides)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=check_sfc)
    world = World.create(config_dir, seed=seed, overrides=overrides, modules=[eco])
    return record_run(world, ticks)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="config directory (default: shipped bundle)")
    p.add_argument("--run", type=Path, default=None, help="existing recording (.npz) to render")
    p.add_argument("--record", type=Path, default=None, help="write the new recording here (.npz + .json)")
    p.add_argument("--ticks", type=int, default=2520, help="days to simulate when recording (252 = 1 year)")
    p.add_argument("--seed", type=int, default=0, help="World root seed (unitless)")
    p.add_argument("--no-sfc", action="store_true", help="skip the monthly SFC assertion (faster)")
    p.add_argument("--pi-star", type=float, default=0.0, help="inflation target, annual decimal")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (goldens use dynamics.banks.mode=passthrough)",
    )
    p.add_argument("--golden", type=Path, default=None, help="baseline .npz to overlay")
    p.add_argument("--golden-prefix", default="", help="series prefix inside the golden, e.g. ns_")
    p.add_argument("--golden-label", default="golden", help="legend label for the baseline")
    p.add_argument("--title", default="MarketSim run", help="page title")
    p.add_argument("--out", type=Path, default=Path("run.html"), help="HTML output path")
    args = p.parse_args(argv)

    if args.out.suffix.lower() not in (".html", ".htm"):
        p.error(f"--out must be an .html path, got {args.out}")

    if args.run is not None:
        rec = read_run(args.run)
        source = str(args.run)
    else:
        config_dir = args.config if args.config is not None else default_config_dir()
        try:
            overrides = parse_overrides(args.override)
        except ValueError as exc:
            p.error(str(exc))
        rec = build_recording(
            config_dir,
            ticks=args.ticks,
            seed=args.seed,
            check_sfc=not args.no_sfc,
            pi_star=args.pi_star,
            overrides=overrides or None,
        )
        source = f"{args.ticks} ticks, seed {args.seed}"
        if args.record is not None:
            npz_path, json_path = write_run(args.record, rec)
            print(f"wrote {npz_path}")
            print(f"wrote {json_path}")

    golden = read_series(args.golden, prefix=args.golden_prefix) if args.golden is not None else None
    out = write_dashboard(
        args.out, rec, golden=golden, golden_label=args.golden_label, title=args.title
    )
    print(f"recorded {len(rec)} months from {source}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
