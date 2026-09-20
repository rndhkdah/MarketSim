"""T6.30 — financials mark-to-market from holdings; §6.3 overlays off in market mode."""

from __future__ import annotations

import numpy as np
import pytest
from tests.validation.test_pricing_vs_betas import _provider, _rate_dln

from marketsim.core.config import Config, load_config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.pricing.bond_buckets import DURATION_REF_YIELD, bucket_duration_years
from marketsim.pricing.fundamentals import MTM_BOOK_WEIGHT, financials_dlnv
from marketsim.real.banks import bond_mtm_pnl

# Gate 15: +100 bp parallel yield rise (annual decimal).
PARALLEL_DY = 0.01


@pytest.mark.validation
def test_gate15_insurance_mtm_from_ledger_exceeds_banks(cfg: Config) -> None:
    """+100bp parallel: INSURANCE Δequity = holdings × D; BANKS lose less (shorter D)."""
    assert cfg.bonds is not None
    decay = {name: spec.decay for name, spec in cfg.bonds.buckets.items()}
    d_bill = bucket_duration_years(DURATION_REF_YIELD, decay["GB_BILL"])
    d_note = bucket_duration_years(DURATION_REF_YIELD, decay["GB_NOTE"])
    d_bond = bucket_duration_years(DURATION_REF_YIELD, decay["GB_BOND"])
    # §6.11 table: GB_BOND ~7.07y at 4.2%; bills much shorter than notes.
    assert d_bond == pytest.approx(7.07, abs=0.01)
    assert d_bill < d_note < d_bond

    # INSURANCE float is long GB_BOND; BANKSYS liquidity book is bills/notes (§6.11).
    ins_bond = 100.0
    bank_bill = 100.0
    bank_note = 100.0
    led = Ledger.empty()
    for name in ("INSURANCE", "BANKSYS", "GOVT"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("INSURANCE", "GB_BOND", ins_bond),
                Entry("BANKSYS", "GB_BILL", bank_bill),
                Entry("BANKSYS", "GB_NOTE", bank_note),
                Entry("GOVT", "GB_BOND", -ins_bond),
                Entry("GOVT", "GB_BILL", -bank_bill),
                Entry("GOVT", "GB_NOTE", -bank_note),
            ),
        )
    )
    assert_consistent(led, 0)

    ins_face = led.position("INSURANCE", "GB_BOND")
    bank_bill_face = led.position("BANKSYS", "GB_BILL")
    bank_note_face = led.position("BANKSYS", "GB_NOTE")
    ins_pnl = bond_mtm_pnl(ins_face, d_bond, PARALLEL_DY)
    bank_pnl = bond_mtm_pnl(bank_bill_face, d_bill, PARALLEL_DY) + bond_mtm_pnl(
        bank_note_face, d_note, PARALLEL_DY
    )
    nfa = net_financial_assets(led)
    assert nfa[led.entities.id("INSURANCE")] == pytest.approx(ins_face)
    assert nfa[led.entities.id("BANKSYS")] == pytest.approx(bank_bill_face + bank_note_face)
    assert ins_pnl == pytest.approx(-ins_face * d_bond * PARALLEL_DY)
    assert bank_pnl == pytest.approx(-(bank_bill_face * d_bill + bank_note_face * d_note) * PARALLEL_DY)
    assert ins_pnl < 0.0
    assert bank_pnl < 0.0
    assert ins_pnl < bank_pnl  # insurance loss larger (more negative)


@pytest.mark.validation
def test_market_mode_disables_section63_overrides(config_dir) -> None:
    """``bonds.pricing: market`` zeros float_rate_beta and bond_mtm_duration only."""
    cfg_par = load_config(config_dir)
    cfg_mkt = load_config(config_dir, {"bonds.pricing": "market"})
    assert cfg_par.bonds is not None and cfg_par.bonds.pricing == "par"
    assert cfg_mkt.bonds is not None and cfg_mkt.bonds.pricing == "market"

    codes = list(cfg_par.codes)
    kwargs = {"r": 0.05, "r0": 0.04, "y10_t": 0.055, "y10_0": 0.042}
    got_par = financials_dlnv(codes, cfg_par, **kwargs)
    got_mkt = financials_dlnv(codes, cfg_mkt, **kwargs)

    dr = kwargs["r"] - kwargs["r0"]
    d_curve = kwargs["y10_t"] - kwargs["r"] - (kwargs["y10_0"] - kwargs["r0"])
    d_y10 = kwargs["y10_t"] - kwargs["y10_0"]
    passthrough = cfg_par.dynamics is not None and cfg_par.dynamics.banks.mode == "passthrough"
    hand_zero = np.zeros(len(codes), dtype=float)
    overlay = np.zeros(len(codes), dtype=float)
    idx = {c: i for i, c in enumerate(codes)}
    for code, fin in cfg_par.sectors.financials.items():
        i = idx[code]
        nim = float(fin.nim_rate_beta) if passthrough else 0.0
        hand_zero[i] = nim * dr + float(fin.curve_beta) * d_curve
        overlay[i] = float(fin.float_rate_beta) * dr - MTM_BOOK_WEIGHT * float(fin.bond_mtm_duration) * d_y10

    assert np.allclose(got_mkt, hand_zero)
    assert np.allclose(got_par, hand_zero + overlay)
    assert not np.allclose(got_par, got_mkt)
    ins = codes.index("INSURANCE")
    assert overlay[ins] != 0.0
    assert got_mkt[ins] == pytest.approx(float(cfg_par.sectors.financials["INSURANCE"].curve_beta) * d_curve)


@pytest.mark.validation
def test_t605_banks_still_only_rate_positive(config_dir, io) -> None:
    """T6.05 gate-1 BANKS-only check: default ``bonds.pricing: par`` is unchanged."""
    cfg = load_config(config_dir)
    assert cfg.bonds is None or cfg.bonds.pricing == "par"
    prov, _real, fin = _provider(cfg, io)
    dln_rate = _rate_dln(prov, fin, cfg)
    codes = list(cfg.codes)
    banks = codes.index("BANKS")
    positive = [c for c, v in zip(codes, dln_rate, strict=True) if v > 0]
    assert positive == ["BANKS"], positive
    assert dln_rate[banks] > 0
