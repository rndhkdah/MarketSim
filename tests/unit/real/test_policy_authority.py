from __future__ import annotations

from marketsim.real.economy import make_real_world
from marketsim.real.policy.authority import SOURCE_RANK, PolicyAuthority, PolicyDecision, PolicyDesk


def test_autopilot_world_hash_identical(config_dir) -> None:
    a = make_real_world(config_dir, seed=4, pi_star=0.0, check_sfc=False)
    b = make_real_world(config_dir, seed=4, pi_star=0.0, check_sfc=False)
    a.step(21 * 3)
    b.step(21 * 3)
    assert a.state_hash() == b.state_hash()
    assert a.modules[0].policy.govt.control == "autopilot"


def test_out_of_range_clipped_and_reported() -> None:
    auth = PolicyAuthority.government(tax_bounds=(0.0, 0.6), tariff_max=0.5)
    dec = PolicyDecision(source="agent", tau_y=0.9, tariff=0.8)
    reports = auth.submit(dec, month=0, lag_m=0)
    names = {r.lever: r for r in reports}
    assert "tau_y" in names
    assert names["tau_y"].applied == 0.6
    assert names["tau_y"].requested == 0.9
    assert names["tariff"].applied == 0.5
    assert auth.effective["tau_y"] == 0.6


def test_legislative_lag_honoured() -> None:
    auth = PolicyAuthority.government()
    auth.submit(PolicyDecision(source="agent", tau_y=0.20), month=2, lag_m=1)
    assert auth.effective.get("tau_y") is None
    due = auth.activate_due(3)
    assert due
    assert auth.effective["tau_y"] == 0.20
    assert not auth.activate_due(3)


def test_partial_decision_leaves_others_on_autopilot() -> None:
    auth = PolicyAuthority.government()
    auth.submit(PolicyDecision(source="agent", tau_y=0.15), month=0, lag_m=0)
    auth.activate_due(0)
    merged = auth.merged()
    assert merged["tau_y"] == 0.15
    assert "tau_c" not in merged
    assert "vat" not in merged


def test_arbitration_agent_beats_scripted() -> None:
    auth = PolicyAuthority.government()
    auth.submit(PolicyDecision(source="scripted", tau_y=0.10), month=0, lag_m=0)
    auth.submit(PolicyDecision(source="agent", tau_y=0.12), month=0, lag_m=0)
    auth.activate_due(0)
    assert auth.merged()["tau_y"] == 0.12
    assert SOURCE_RANK["agent"] > SOURCE_RANK["scripted"]


def test_state_round_trip() -> None:
    desk = PolicyDesk.default()
    desk.govt.submit(PolicyDecision(source="scripted", vat=0.1), month=1, lag_m=1)
    state = desk.to_state()
    other = PolicyDesk.default()
    other.from_state(state)
    assert other.govt.pending[0]["decision"]["vat"] == 0.1
    assert other.govt.news[0]["authority"] == "GOVT"
