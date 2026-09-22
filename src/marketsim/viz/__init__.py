"""Recording and rendering helpers for simulation output (T10.16 / T10.17).

Nothing in this package is part of the engine: it reads a stepping ``World``
and writes files. No module here may be attached to a ``World`` (that would
enter ``World.to_state()`` and change every ``state_hash``), and no engine
module may import from here.
"""

from marketsim.viz.recorder import (
    MATRIX_SERIES,
    SCALAR_SERIES,
    SCHEMA_VERSION,
    Recording,
    RunRecorder,
    read_run,
    read_series,
    record_run,
    write_run,
)

__all__ = [
    "MATRIX_SERIES",
    "SCALAR_SERIES",
    "SCHEMA_VERSION",
    "Recording",
    "RunRecorder",
    "read_run",
    "read_series",
    "record_run",
    "write_run",
]
