"""T6.24 — decaying-coupon bucket arithmetic, par/market switch, no reval income."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config, load_config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import GOVT_MIX, open_passthrough_books, opening_bond_books
from marketsim.pricing.bond_buckets import (
    BUCKET_ORDER,
    DURATION_REF_YIELD,
    BondBooks,
    BondsFile,
    bucket_duration_years,
    bucket_price,
    issuance_mix_tuple,
    month_face_flows,
    monthly_holding_return,
    post_coupon_and_redeem,
    revaluation,
)
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def test_shipped_bonds_yaml_is_par(cfg: Config) -> None:
    assert cfg.bonds is not None
    assert cfg.bonds.pricing == "par"
    assert set(cfg.bonds.buckets) == set(BUCKET_ORDER)
    assert issuance_mix_tuple(cfg.bonds) == GOVT_MIX
    assert cfg.bonds.buckets["GB_BILL"].decay == pytest.approx(4.0)
    assert cfg.bonds.buckets["GB_NOTE"].decay == pytest.approx(1.0 / 3.0)
    assert cfg.bonds.buckets["GB_BOND"].decay == pytest.approx(0.10)
    assert cfg.bonds.buckets["CORP_POOL"].decay == pytest.approx(0.20)
    assert cfg.bonds.duration_ref_yield == pytest.approx(DURATION_REF_YIELD)


@pytest.mark.parametrize("y", [0.0, 0.042, 0.10])
def test_formulas_at_three_yields(cfg: Config, y: float) -> None:
    assert cfg.bonds is not None
    kappa = DURATION_REF_YIELD
    for _name, spec in cfg.bonds.buckets.items():
        p = bucket_price(y, kappa, spec.decay)
        assert p > 0.0
        if y == kappa:
            assert p == pytest.approx(1.0)
        # Holding return at an unchanged yield equals the carry identity at P.
        ret = monthly_holding_return(p, p, kappa, spec.decay)
        coupon, redeem, face_next = month_face_flows(1.0, kappa, spec.decay)
        assert face_next == pytest.approx(1.0 - spec.decay / 12.0)
        assert coupon == pytest.approx(kappa / 12.0)
        assert redeem == pytest.approx(spec.decay / 12.0)
        # At P=1 and y=κ the monthly return is κ/12 (carry).
        if y == kappa:
            assert ret == pytest.approx(kappa / 12.0)


def test_gb_bond_duration_7_07_years_at_4_2pct(cfg: Config) -> None:
    assert cfg.bonds is not None
    dur = bucket_duration_years(DURATION_REF_YIELD, cfg.bonds.buckets["GB_BOND"].decay)
    assert dur == pytest.approx(7.07, abs=0.01)


def test_revaluation_never_appears_as_income(cfg: Config) -> None:
    assert cfg.bonds is not None
    led = Ledger.empty(debug_journal=True)
    led.register_entity("HH:0")
    led.register_entity("GOVT")
    face = 100.0
    led.post(
        Tx(
            0,
            "opening",
            (Entry("HH:0", "GB_BOND", face), Entry("GOVT", "GB_BOND", -face)),
        )
    )
    books = BondBooks.from_bonds(cfg.bonds, ss_yield=DURATION_REF_YIELD)
    p1 = bucket_price(0.05, DURATION_REF_YIELD, cfg.bonds.buckets["GB_BOND"].decay)
    mark = revaluation(face, books.prices["GB_BOND"], p1)
    books.mark("GB_BOND", p1)
    assert mark == pytest.approx(face * (p1 - 1.0))
    assert mark != 0.0
    assert "revaluation" not in led._tags
    coupon, redeem, _ = post_coupon_and_redeem(
        led,
        tick=1,
        holder="HH:0",
        issuer="GOVT",
        instrument="GB_BOND",
        face=face,
        kappa_annual=DURATION_REF_YIELD,
        delta_annual=cfg.bonds.buckets["GB_BOND"].decay,
    )
    assert coupon > 0.0 and redeem > 0.0
    assert set(led._tags) <= {"opening", "interest_bonds", "bond_redeem"}
    flows = led.close_period()
    assert all(tag in {"opening", "interest_bonds", "bond_redeem"} for (*_, tag) in flows)
    assert led.position("HH:0", "GB_BOND") + led.position("GOVT", "GB_BOND") == pytest.approx(0.0)


def test_par_opening_bitwise_phase2(cfg: Config, io) -> None:
    """Gate 9: ``bonds.pricing: par`` opening equals the Phase-2 GOVT_MIX posts."""
    assert cfg.bonds is not None
    assert cfg.bonds.pricing == "par"
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    led = open_passthrough_books(cfg, real, fin)
    books = opening_bond_books(cfg, fin)
    assert books.pricing == "par"
    for inst, share in GOVT_MIX:
        assert led.position("HH:0", inst) == pytest.approx(fin.B * share)
        assert led.position("GOVT", inst) == pytest.approx(-fin.B * share)
        assert books.unit_value(inst) == pytest.approx(1.0)
        assert books.market_value(led.position("HH:0", inst), inst) == pytest.approx(led.position("HH:0", inst))


def test_market_mode_ss_price_is_par(config_dir, io) -> None:
    """Switching to market keeps P=1 at the SS yield (gate 9 second half)."""
    cfg = load_config(config_dir, {"bonds.pricing": "market"})
    assert cfg.bonds is not None
    assert cfg.bonds.pricing == "market"
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    books = BondBooks.from_bonds(cfg.bonds, ss_yield=float(fin.r0))
    for name, spec in cfg.bonds.buckets.items():
        assert bucket_price(fin.r0, books.kappa[name], spec.decay) == pytest.approx(1.0)
        assert books.unit_value(name) == pytest.approx(1.0)
    led = open_passthrough_books(cfg, real, fin)
    for inst, share in GOVT_MIX:
        assert led.position("HH:0", inst) == pytest.approx(fin.B * share)


def test_bond_books_roundtrip(cfg: Config) -> None:
    assert cfg.bonds is not None
    books = BondBooks.from_bonds(cfg.bonds, ss_yield=0.02)
    books.mark("GB_NOTE", 0.99)
    other = BondBooks.from_state(books.to_state())
    assert other.to_state() == books.to_state()


def test_bonds_file_rejects_bad_mix() -> None:
    with pytest.raises(Exception, match="issuance_mix"):
        BondsFile.model_validate(
            {
                "buckets": {
                    "GB_BILL": {"issuer": "GOVT", "decay": 4.0},
                    "GB_NOTE": {"issuer": "GOVT", "decay": 1.0 / 3.0},
                    "GB_BOND": {"issuer": "GOVT", "decay": 0.1},
                    "CORP_POOL": {"issuer": "CORPPOOL", "decay": 0.2},
                },
                "issuance_mix": {"GB_BILL": 1.0, "GB_NOTE": 0.0, "GB_BOND": 0.0, "EXTRA": 0.0},
            }
        )
