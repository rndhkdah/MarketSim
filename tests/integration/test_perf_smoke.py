from __future__ import annotations

from bench import bench


def test_r1_throughput_5000_ticks_per_s(config_dir) -> None:
    rec = bench(config_dir, ticks=2520, warmup=42, seed=0)
    assert rec["ticks_per_s"] >= 5000.0
