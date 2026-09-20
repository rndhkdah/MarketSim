"""Pydantic model ↔ protobuf-like bytes (T9.06).

When ``protobuf`` is installed the JSON envelope is wrapped in
``google.protobuf.BytesValue`` (field 1, length-delimited). Otherwise the
envelope is raw UTF-8 JSON — the fallback used in default tests. Decode always
runs ``model_validate``, so ``Observation`` still rejects hidden keys (ξ, …).
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from marketsim.api.schemas import API_SCHEMA_VERSION

TModel = TypeVar("TModel", bound=BaseModel)

# Envelope keys (unitless). ``payload`` field units follow the pydantic model.
_SCHEMA_KEY = "schema_version"
_PAYLOAD_KEY = "payload"
_TYPE_KEY = "type"
# JSON objects start with 0x7b; protobuf BytesValue starts with field-1 tag 0x0a.
_JSON_OBJECT = b"{"


def protobuf_available() -> bool:
    """True if the ``protobuf`` package can be imported (does not import it)."""
    try:
        return importlib.util.find_spec("google.protobuf") is not None
    except ModuleNotFoundError:
        return False


def encode_payload(payload: Mapping[str, Any], *, type_name: str | None = None) -> bytes:
    """Encode a JSON object. Returns protobuf-wrapped or raw UTF-8 JSON bytes."""
    envelope: dict[str, Any] = {
        _SCHEMA_KEY: API_SCHEMA_VERSION,
        _PAYLOAD_KEY: dict(payload),
    }
    if type_name:
        envelope[_TYPE_KEY] = type_name
    raw = json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str).encode()
    wrapped = _wrap_bytes_value(raw)
    return raw if wrapped is None else wrapped


def encode_model(model: BaseModel) -> bytes:
    """Encode a closed API model. Field units match the model's field docs."""
    return encode_payload(model.model_dump(mode="json"), type_name=type(model).__name__)


def decode_payload(data: bytes) -> dict[str, Any]:
    """Unwrap wire bytes to a JSON object (the inner payload if enveloped)."""
    inner = _unwrap_bytes_value(data)
    parsed = json.loads(inner)
    if not isinstance(parsed, dict):
        raise ValueError("wire payload must be a JSON object")
    if _SCHEMA_KEY in parsed and _PAYLOAD_KEY in parsed:
        body = parsed[_PAYLOAD_KEY]
        if not isinstance(body, dict):
            raise ValueError("envelope payload must be a JSON object")
        return body
    return parsed


def decode_model(data: bytes, cls: type[TModel]) -> TModel:
    """Decode wire bytes into ``cls``. Extra keys (e.g. ``xi``) are rejected."""
    return cls.model_validate(decode_payload(data))


def _wrap_bytes_value(raw: bytes) -> bytes | None:
    if not protobuf_available():
        return None
    from google.protobuf.wrappers_pb2 import BytesValue

    return BytesValue(value=raw).SerializeToString()


def _unwrap_bytes_value(data: bytes) -> bytes:
    if data.startswith(_JSON_OBJECT) or not protobuf_available():
        return data
    from google.protobuf.wrappers_pb2 import BytesValue

    msg = BytesValue()
    try:
        msg.ParseFromString(data)
    except Exception:
        return data
    return bytes(msg.value) if msg.value else data
