"""Firm rating → credit-limit multipliers, never the borrowing rate (T5.10 / D14)."""

from __future__ import annotations

from marketsim.firms.firm import RATINGS, FirmsFile

# ND/EBITDA bands (§5.5). Higher leverage → worse letter.
_ND_BANDS: tuple[tuple[float, str], ...] = (
    (1.0, "AAA"),
    (2.0, "AA"),
    (3.0, "A"),
    (4.0, "BBB"),
    (5.0, "BB"),
    (6.0, "B"),
    (float("inf"), "CCC"),
)


def letter_from_nd(nd_ebitda: float) -> str:
    """Map ND/EBITDA to a letter. ``nd_ebitda`` is a ratio."""
    for cut, letter in _ND_BANDS:
        if nd_ebitda <= cut:
            return letter
    return "CCC"


def score_rating(
    *,
    nd_ebitda: float,
    icr: float,
    size: float,
    vol: float,
) -> str:
    """Combine ND/EBITDA, interest-cover, size and earnings vol → AAA…CCC."""
    letter = letter_from_nd(nd_ebitda)
    idx = RATINGS.index(letter)
    if icr < 1.0:
        idx = min(idx + 2, len(RATINGS) - 1)
    elif icr < 2.0:
        idx = min(idx + 1, len(RATINGS) - 1)
    if size < 1.0:
        idx = min(idx + 1, len(RATINGS) - 1)
    if vol > 0.3:
        idx = min(idx + 1, len(RATINGS) - 1)
    return RATINGS[idx]


def limit_multiplier(letter: str, cfg: FirmsFile) -> float:
    """Credit-limit multiplier (dimensionless). Does not enter the rate."""
    return float(cfg.financing.rating_multipliers[letter])
