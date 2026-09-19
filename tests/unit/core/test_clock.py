from __future__ import annotations

from marketsim.core.calendar import Calendar
from marketsim.core.clock import Clock, EventQueue


def test_ties_break_by_priority_then_sequence() -> None:
    q = EventQueue()
    q.schedule(5, "late-high", priority=2)
    q.schedule(5, "first-low", priority=0)
    q.schedule(5, "second-low", priority=0)
    q.schedule(5, "mid", priority=1)
    assert q.pop_due(5) == ["first-low", "second-low", "mid", "late-high"]


def test_pop_due_deterministic_and_ignores_future() -> None:
    q = EventQueue()
    q.schedule(3, "c")
    q.schedule(1, "a")
    q.schedule(2, "b")
    q.schedule(9, "later")
    assert q.pop_due(2) == ["a", "b"]
    assert q.pop_due(3) == ["c"]
    assert q.pop_due(8) == []


def test_queue_state_roundtrip() -> None:
    q = EventQueue()
    q.schedule(1, {"k": 1}, priority=3)
    q.schedule(4, "x")
    restored = EventQueue.from_state(q.to_state())
    assert restored.pop_due(10) == q.pop_due(10)


def test_calendar_boundaries_three_years() -> None:
    cal = Calendar(days_per_month=21)
    month_ends = 0
    quarter_ends = 0
    for t in range(3 * 252):
        if cal.is_month_end(t):
            month_ends += 1
        if cal.is_quarter_end(t):
            quarter_ends += 1
        year, month, day = cal.ymd(t)
        assert 1 <= month <= 12
        assert 1 <= day <= 21
        assert year == 1 + t // 252
    assert month_ends == 36
    assert quarter_ends == 12


def test_clock_advance_drains_due() -> None:
    clock = Clock()
    clock.queue.schedule(0, "now")
    clock.queue.schedule(2, "later")
    t0, due0 = clock.advance()
    assert t0 == 0 and due0 == ["now"]
    t1, due1 = clock.advance()
    assert t1 == 1 and due1 == []
    t2, due2 = clock.advance()
    assert t2 == 2 and due2 == ["later"]
