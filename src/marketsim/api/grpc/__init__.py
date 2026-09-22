"""Optional gRPC transport (T9.06). Same closed schemas as REST / ``LocalClient``.

Importing this package (or ``marketsim.api``) does not require ``grpcio``.
Constructing ``serve`` / ``build_server`` does — install ``marketsim[grpc]``.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def _grpcio_installed() -> bool:
    """``find_spec`` does not import grpcio; a missing parent package is False."""
    try:
        return importlib.util.find_spec("grpc") is not None
    except ModuleNotFoundError:
        return False


GRPC_AVAILABLE: bool = _grpcio_installed()

__all__ = [
    "AUTHORIZATION_KEY",
    "GRPC_AVAILABLE",
    "SERVICE_NAME",
    "MarketSimServicer",
    "authorization_metadata",
    "bearer_token",
    "build_server",
    "decode_model",
    "decode_payload",
    "encode_model",
    "encode_payload",
    "serve",
]

_LAZY: dict[str, tuple[str, str]] = {
    "AUTHORIZATION_KEY": ("marketsim.api.grpc.servicer", "AUTHORIZATION_KEY"),
    "SERVICE_NAME": ("marketsim.api.grpc.servicer", "SERVICE_NAME"),
    "MarketSimServicer": ("marketsim.api.grpc.servicer", "MarketSimServicer"),
    "authorization_metadata": ("marketsim.api.grpc.servicer", "authorization_metadata"),
    "bearer_token": ("marketsim.api.grpc.servicer", "bearer_token"),
    "decode_model": ("marketsim.api.grpc.codec", "decode_model"),
    "decode_payload": ("marketsim.api.grpc.codec", "decode_payload"),
    "encode_model": ("marketsim.api.grpc.codec", "encode_model"),
    "encode_payload": ("marketsim.api.grpc.codec", "encode_payload"),
    "build_server": ("marketsim.api.grpc.server", "build_server"),
    "serve": ("marketsim.api.grpc.server", "serve"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _LAZY[name]
    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
