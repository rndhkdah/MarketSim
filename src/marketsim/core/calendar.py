"""Simulated calendar: 21 days / month, 12 months, 252 days / year."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Calendar:
    """Maps a tick index to (year, month, day). Units: tick = 1 simulated day."""

    days_per_month: int = 21

    def ymd(self, tick: int) -> tuple[int, int, int]:
        if tick < 0:
            raise ValueError(f"tick must be >= 0, got {tick}")
        day_of_year = tick % (12 * self.days_per_month)
        year = 1 + tick // (12 * self.days_per_month)
        month = 1 + day_of_year // self.days_per_month
        day = 1 + day_of_year % self.days_per_month
        return year, month, day

    def is_month_end(self, tick: int) -> bool:
        return self.ymd(tick)[2] == self.days_per_month

    def is_quarter_end(self, tick: int) -> bool:
        _, month, day = self.ymd(tick)
        return day == self.days_per_month and month in (3, 6, 9, 12)

    def months_elapsed(self, tick: int) -> int:
        """Completed months after `tick` has been processed (0-based count of month-ends)."""
        return tick // self.days_per_month
