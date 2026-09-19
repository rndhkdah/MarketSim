from __future__ import annotations

import marketsim.layer1.betas as betas_mod
from marketsim.layer1.betas import BetasParams, derive_betas, regime_demo


def test_importing_module_does_not_touch_io(monkeypatch, cfg, io) -> None:
    def boom(*_a, **_k):
        raise AssertionError("derive_betas must not run at import")

    monkeypatch.setattr(betas_mod, "derive_betas", boom)
    monkeypatch.setattr(betas_mod, "regime_demo", boom)
    # Re-importing the functions from the already-loaded module is fine; the
    # contract is that `import marketsim.layer1.betas` itself does no solve.
    import importlib

    importlib.reload(betas_mod)


def test_regime_demo_seed_7(cfg, io) -> None:
    betas = derive_betas(io, cfg, BetasParams())
    demo = regime_demo(betas, seed=7)
    assert abs(demo.demand_corr - (-0.74)) < 0.02, demo.demand_corr
    assert abs(demo.supply_corr - 0.90) < 0.02, demo.supply_corr
    assert demo.n_sign_flips == 13, demo.n_sign_flips


def test_banks_only_positive_rate_beta(cfg, io) -> None:
    betas = derive_betas(io, cfg)
    positive = [c for c, b in zip(betas.codes, betas.beta_rate, strict=True) if b > 0]
    assert positive == ["BANKS"], positive
