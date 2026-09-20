"""Regional labour-force pools and migration (§3.2)."""

from __future__ import annotations

import numpy as np

from marketsim.regions.geometry import MigrationCfg


def migrate_labour_force(
    lf: np.ndarray,
    u: np.ndarray,
    cfg: MigrationCfg,
) -> np.ndarray:
    """Monthly migration: ``ΔLF_r = LF_r · rate · (Ū − U_r)·100 / 12``, then renormalise.

    ``lf`` and ``u`` are ``(R,)``. ``rate`` is share of LF per year per pp U-gap.
    National LF is conserved to numerical precision. Units: wage-units of LF.
    """
    lf_a = np.asarray(lf, dtype=float)
    total = float(lf_a.sum())
    if not cfg.enabled or total <= 0:
        return lf_a.copy()
    u_a = np.asarray(u, dtype=float)
    u_bar = float(np.average(u_a, weights=lf_a))
    # (Ū − U)·100 is the gap in percentage points (§3.2).
    delta = lf_a * float(cfg.rate_per_pp_per_year) * (u_bar - u_a) * 100.0 / 12.0
    moved = np.maximum(lf_a + delta, 0.0)
    return moved * (total / max(float(moved.sum()), 1e-15))
