"""Canonical serialisation, state hashing, and tolerant diffs."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic encoding. Dicts are sorted by key; floats are little-endian float64."""
    if obj is None:
        return b"N"
    if isinstance(obj, bool):
        return b"T" if obj else b"F"
    if isinstance(obj, (np.bool_,)):
        return b"T" if bool(obj) else b"F"
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return b"I" + str(int(obj)).encode("ascii")
    if isinstance(obj, (float, np.floating)):
        return b"D" + struct.pack("<d", float(obj))
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        return b"S" + str(len(raw)).encode("ascii") + b":" + raw
    if isinstance(obj, bytes):
        return b"B" + str(len(obj)).encode("ascii") + b":" + obj
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        le_dtype = arr.dtype.newbyteorder("<")
        payload = np.ascontiguousarray(arr.astype(le_dtype, copy=False)).tobytes()
        header = f"{le_dtype.str}:{arr.shape}".encode("ascii")
        return b"A" + header + b"|" + payload
    if isinstance(obj, dict):
        parts = [b"M"]
        for key in sorted(obj.keys(), key=lambda k: str(k)):
            parts.append(canonical_bytes(key))
            parts.append(canonical_bytes(obj[key]))
        return b"".join(parts)
    if isinstance(obj, (list, tuple)):
        tag = b"L" if isinstance(obj, list) else b"U"
        return tag + b"".join(canonical_bytes(x) for x in obj)
    raise TypeError(f"cannot canonicalise {type(obj)!r}")


def state_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def state_diff(a: Any, b: Any, tol: float = 0.0, path: str = "$") -> list[str]:
    """Return human-readable paths where `a` and `b` differ by more than `tol`."""
    diffs: list[str] = []
    _walk(a, b, tol, path, diffs)
    return diffs


def _walk(a: Any, b: Any, tol: float, path: str, diffs: list[str]) -> None:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.shape != bb.shape:
            diffs.append(f"{path}: shape {aa.shape} vs {bb.shape}")
            return
        if np.issubdtype(aa.dtype, np.floating) or np.issubdtype(bb.dtype, np.floating):
            if not np.allclose(aa.astype(float), bb.astype(float), atol=tol, rtol=0.0, equal_nan=True):
                diffs.append(f"{path}: ndarray mismatch")
            return
        if not np.array_equal(aa, bb):
            diffs.append(f"{path}: ndarray mismatch")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in sorted(keys, key=str):
            if k not in a:
                diffs.append(f"{path}.{k}: missing on left")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing on right")
            else:
                _walk(a[k], b[k], tol, f"{path}.{k}", diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, tol, f"{path}[{i}]", diffs)
        return
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        if abs(float(a) - float(b)) > tol:
            diffs.append(f"{path}: {a} vs {b}")
        return
    if a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")
