from __future__ import annotations

from marketsim.core.hashing import canonical_bytes, state_diff, state_hash
from marketsim.core.rng import RngHub


def test_streams_independent_of_creation_order() -> None:
    a = RngHub(7)
    b = RngHub(7)
    xa = a.stream("pricing.noise").standard_normal(8)
    ya = a.stream("events").standard_normal(8)
    # opposite creation order
    yb = b.stream("events").standard_normal(8)
    xb = b.stream("pricing.noise").standard_normal(8)
    assert (xa == xb).all()
    assert (ya == yb).all()


def test_hash_invariant_to_dict_insertion_order() -> None:
    left = {"b": 1.5, "a": [2, 3], "z": {"k": 0.0}}
    right = {"z": {"k": 0.0}, "a": [2, 3], "b": 1.5}
    assert state_hash(left) == state_hash(right)
    assert canonical_bytes(left) == canonical_bytes(right)


def test_save_load_continue_matches_uninterrupted() -> None:
    hub = RngHub(11)
    g = hub.stream("matching")
    g.standard_normal(5)
    mid = hub.to_state()
    rest = list(g.standard_normal(7))

    clone = RngHub.from_state(mid)
    cont = list(clone.stream("matching").standard_normal(7))
    assert rest == cont


def test_state_diff_respects_tol() -> None:
    assert state_diff({"x": 1.0}, {"x": 1.0004}, tol=1e-3) == []
    assert state_diff({"x": 1.0}, {"x": 1.1}, tol=1e-3)
