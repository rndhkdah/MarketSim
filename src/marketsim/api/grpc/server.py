"""Optional gRPC server (T9.06). Constructed only when ``grpcio`` is installed.

``serve`` binds a TCP port and blocks — not used in CI. Tests that need a
server construct ``build_server`` and must not call ``add_insecure_port``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from marketsim.api.grpc.codec import decode_payload, encode_model, encode_payload
from marketsim.api.grpc.servicer import SERVICE_NAME, MarketSimServicer
from marketsim.api.sessions import WorldManager
from marketsim.sdk.local import LOCAL_METHODS

# Conventional gRPC listen address (IANA 50051). Not a yaml key — T9.06 cannot edit config.py.
_DEFAULT_BIND = "[::]:50051"
# One worker: the engine is deterministic; concurrency is the client's problem.
_DEFAULT_WORKERS = 1


def _grpcio_installed() -> bool:
    try:
        return importlib.util.find_spec("grpc") is not None
    except ModuleNotFoundError:
        return False


GRPC_AVAILABLE: bool = _grpcio_installed()


def _import_grpc() -> Any:
    try:
        import grpc
    except ImportError as exc:
        raise ImportError(
            "gRPC transport requires the optional extra 'marketsim[grpc]' (grpcio>=1.67, protobuf>=5)."
        ) from exc
    return grpc


def _rpc_name(method: str) -> str:
    return "".join(part.capitalize() for part in method.split("_"))


def _serialize_reply(result: Any) -> bytes:
    if result is None:
        return encode_payload({})
    if isinstance(result, BaseModel):
        return encode_model(result)
    if isinstance(result, str):
        return encode_payload({"state_hash": result})
    if isinstance(result, Mapping):
        return encode_payload(result)
    raise TypeError(f"unsupported gRPC reply type {type(result).__name__}")


def _unary(servicer: MarketSimServicer, name: str) -> Any:
    def handler(request: dict[str, Any], context: Any) -> Any:
        return getattr(servicer, name)(request, context)

    return handler


def build_server(manager: WorldManager, *, max_workers: int = _DEFAULT_WORKERS) -> Any:
    """Construct an unstarted ``grpc.Server``. Does not bind or listen.

    ``max_workers`` is a thread count. Requires ``marketsim[grpc]``.
    """
    grpc = _import_grpc()
    from concurrent.futures import ThreadPoolExecutor

    servicer = MarketSimServicer(manager)
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
    handlers = {
        _rpc_name(name): grpc.unary_unary_rpc_method_handler(
            _unary(servicer, name),
            request_deserializer=decode_payload,
            response_serializer=_serialize_reply,
        )
        for name in sorted(LOCAL_METHODS)
    }
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),))
    return server


def serve(manager: WorldManager, bind: str = _DEFAULT_BIND) -> None:
    """Bind ``bind`` (host:port) and serve until terminated. Not started in CI."""
    server = build_server(manager)
    server.add_insecure_port(bind)
    server.start()
    server.wait_for_termination()
