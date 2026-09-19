"""Named PCG64 streams. No global RNG."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


def _spawn_key(name: str) -> int:
    return int.from_bytes(sha256(name.encode("utf-8")).digest()[:4], "little")


class RngHub:
    """Root seed → named `numpy.random.Generator(PCG64)` streams."""

    def __init__(self, root_seed: int) -> None:
        self.root_seed = int(root_seed)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        cached = self._streams.get(name)
        if cached is not None:
            return cached
        ss = np.random.SeedSequence(self.root_seed, spawn_key=(_spawn_key(name),))
        gen = np.random.Generator(np.random.PCG64(ss))
        self._streams[name] = gen
        return gen

    def to_state(self) -> dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "streams": {
                name: _json_safe(gen.bit_generator.state) for name, gen in sorted(self._streams.items())
            },
        }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RngHub:
        hub = cls(int(state["root_seed"]))
        for name, bit_state in state["streams"].items():
            gen = hub.stream(name)
            gen.bit_generator.state = bit_state
        return hub
