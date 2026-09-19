"""Typed errors for the engine core."""


class MarketsimError(Exception):
    """Base error."""


class ConfigError(MarketsimError):
    """Invalid or missing configuration."""


class IOTableError(MarketsimError):
    """IO table failed a load-time check."""


class StateError(MarketsimError):
    """Serialisation / state-protocol failure."""
