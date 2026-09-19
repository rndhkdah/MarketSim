"""Typed errors for the engine core."""


class MarketsimError(Exception):
    """Base error."""


class ConfigError(MarketsimError):
    """Invalid or missing configuration."""


class IOTableError(MarketsimError):
    """IO table failed a load-time check."""


class StateError(MarketsimError):
    """Serialisation / state-protocol failure."""


class LedgerError(MarketsimError):
    """Unbalanced posting or unknown ledger name."""


class SFCError(MarketsimError):
    """Stock-flow-consistency assertion failed."""

    def __init__(self, message: str, *, tag: str | None = None, entity: str | None = None, amount: float | None = None):
        super().__init__(message)
        self.tag = tag
        self.entity = entity
        self.amount = amount
