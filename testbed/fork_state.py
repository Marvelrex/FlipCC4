"""Match-state checkpoint / fork for a running FlipThem-CC4 match (paired-branch design).

A checkpoint is the live object graph that carries one match's run state at one instant, serialised once with
pickle: the CybORG environment (hosts, sessions, in-flight actions, Green agents, every RNG that
lives inside it), the wrapper's clock / value sums / completion billing, both team controllers
(ledger, pending actions, wake state, campaign memory, telemetry, request logs), the shared
RequestBudget and the adapters' accrued cost, the control trace and the passive execution
recorder. Python's and NumPy's global RNG states are stored beside it.

It is NOT reconstructed from logs: a branch is `pickle.loads` of these exact bytes, so two branches
share no INSTANCE state with each other or with the prefix. Module/class definitions and the
process-global RNG are process-wide, not per-branch: the global RNG is reset from the checkpoint
before each branch, and branches are run sequentially (concurrent or interleaved branches in one
process are not supported).

External handles (the model clients) are the only things NOT serialised. They are written as
persistent ids and supplied fresh per branch at restore time, which also guarantees the checkpoint
never contains an API key: a `RealClient` reached by the pickler without a handle mapping aborts
the checkpoint.

Verification (`state_fingerprint`) walks the whole graph canonically and hashes it twice:
``content`` (sets compared as sets) and ``ordered`` (sets in iteration order, which matters when
code draws from ``list(some_set)`` with the RNG). Restored branches must match each other on both.

Supported range (tested on CPython 3.10 with the repository's CybORG/flipthem object graph):
atoms, enums, ipaddress values, types/functions, dict (insertion order kept), list/tuple, set/
frozenset of IMMUTABLE VALUE elements only, numpy arrays (object dtype recursed element-wise) and
scalars, numpy/stdlib RNGs, bound methods, datetime family, bytearray, deque, functools.partial,
mappingproxy of static contents only, and plain objects via __dict__/__slots__ (a string
``__slots__`` is one slot); a C object without either is hashed through its __reduce_ex__.
Generators, coroutines, weakrefs, file objects, locks and sockets, sets holding identity-bearing
objects and mappingproxies over mutable contents are REFUSED (CheckpointError) rather than guessed.
A content-mode match proves equal content and aliasing; it does not by itself prove that the
restored object behaves identically to the live one (pickle rebuilds set layouts), which is why
branches are compared with each other in both modes, never with the live object.
"""
from __future__ import annotations

import collections
import datetime
import enum
import functools
import hashlib
import inspect
import io
import ipaddress
import pickle
import random
import socket
import threading
import types
import weakref

import numpy as np

CHECKPOINT_PROTOCOL = 5
CHECKPOINT_SCHEMA = "flipthem-fork-checkpoint-v1"


class CheckpointError(Exception):
    pass


def _forbidden_types():
    try:
        from flipthem.llm.real_client import RealClient
        return (RealClient,)
    except Exception:   # pragma: no cover - import guard only
        return ()


class _Pickler(pickle.Pickler):
    def __init__(self, stream, handles):
        super().__init__(stream, protocol=CHECKPOINT_PROTOCOL)
        self._handles = {id(obj): token for token, obj in handles.items()}
        self._keep = list(handles.values())   # keep ids stable while pickling
        self._forbidden = _forbidden_types()

    def persistent_id(self, obj):
        token = self._handles.get(id(obj))
        if token is not None:
            return token
        if self._forbidden and isinstance(obj, self._forbidden):
            raise CheckpointError("a real model client was reached without a handle mapping; "
                                  "refusing to serialise it (it holds credentials)")
        return None

    def reducer_override(self, obj):
        # CybORG keeps inspect.signature(...).parameters (a read-only mappingproxy of static
        # metadata) in agent action tables; rebuild it as an equal read-only proxy. Pickle cannot
        # preserve the alias to a proxy's underlying dict, so only immutable static contents are
        # accepted; anything else fails closed.
        if type(obj) is types.MappingProxyType:
            if not _static_mapping(obj):
                raise CheckpointError("mappingproxy with mutable contents is not supported")
            return _mappingproxy, (dict(obj),)
        return NotImplemented


def _is_value(v) -> bool:
    """Immutable values whose identity carries no state (safe as set elements / proxy contents)."""
    if isinstance(v, (_ATOMS, _VALUE_OBJECTS, enum.Enum, type, types.FunctionType,
                      types.BuiltinFunctionType)):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(_is_value(x) for x in v)
    if isinstance(v, inspect.Parameter):
        return _is_value(v.default) or v.default is inspect.Parameter.empty
    return v is inspect.Parameter.empty


def _static_mapping(mapping) -> bool:
    return all(_is_value(k) and _is_value(v) for k, v in mapping.items())


def _mappingproxy(mapping):
    return types.MappingProxyType(mapping)


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, handles):
        super().__init__(stream)
        self._handles = handles

    def persistent_load(self, pid):
        if pid not in self._handles:
            raise CheckpointError(f"checkpoint needs external handle {pid!r}, none supplied")
        return self._handles[pid]


def global_rng_state() -> dict:
    return {"python_random": random.getstate(), "numpy_global": np.random.get_state()}


def set_global_rng_state(state: dict) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_global"])


def snapshot(bundle: dict, handles: dict) -> bytes:
    """Serialise `bundle` (one dict, so shared references inside it stay shared) with every object
    in `handles` ({token: obj}) replaced by its token."""
    buf = io.BytesIO()
    payload = {"schema": CHECKPOINT_SCHEMA, "bundle": bundle, "global_rng": global_rng_state()}
    _Pickler(buf, handles).dump(payload)
    return buf.getvalue()


def restore(blob: bytes, handles: dict, *, apply_global_rng: bool = True) -> dict:
    """A fresh, fully independent copy of the bundle, with tokens bound to `handles`."""
    payload = _Unpickler(io.BytesIO(blob), handles).load()
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("unknown checkpoint schema")
    if apply_global_rng:
        set_global_rng_state(payload["global_rng"])
    return payload["bundle"]


def restore_with_rng(blob: bytes, handles: dict) -> tuple:
    """(bundle, stored global RNG state) without touching the process's global RNG."""
    payload = _Unpickler(io.BytesIO(blob), handles).load()
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("unknown checkpoint schema")
    return payload["bundle"], payload["global_rng"]


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------------------------------------
# Canonical state fingerprint
# ------------------------------------------------------------------------------------------------

_ATOMS = (type(None), bool, int, float, complex, str, bytes)
# Value objects whose __dict__ may hold lazily computed caches (functools.cached_property such as
# IPv4Network.broadcast_address) that pickle legitimately drops: hashed by value.
_VALUE_OBJECTS = (ipaddress._BaseAddress, ipaddress._BaseNetwork, ipaddress._BaseV4, ipaddress._BaseV6)
# Live resources and lazy iterators carry state that neither pickling nor this walk can reproduce.
_UNSUPPORTED = (types.GeneratorType, types.CoroutineType, types.AsyncGeneratorType, weakref.ref,
                io.IOBase, type(threading.Lock()), type(threading.RLock()), socket.socket)


class _Canon:
    """Deterministic structural hash of an object graph. Shared references and cycles are encoded
    by first-visit index, so aliasing (e.g. two components sharing ONE budget/ledger) is part of the
    fingerprint: a copy that silently split a shared object would not match."""

    def __init__(self, ordered: bool, handles: dict):
        self.ordered = ordered
        self.seen = {}
        self.keep = []
        self.handles = {id(o): t for t, o in handles.items()}

    def h(self, obj) -> bytes:
        m = hashlib.sha256()
        self._feed(obj, m)
        return m.digest()

    def _feed(self, obj, m):
        token = self.handles.get(id(obj))
        if token is not None:
            m.update(b"H" + token.encode())
            return
        if isinstance(obj, _ATOMS):
            m.update(b"A" + type(obj).__name__.encode() + b":" + repr(obj).encode())
            return
        if isinstance(obj, _VALUE_OBJECTS):
            m.update(b"V" + type(obj).__name__.encode() + b":" + str(obj).encode())
            return
        if isinstance(obj, enum.Enum):
            m.update(b"E" + type(obj).__qualname__.encode() + b"." + obj.name.encode())
            return
        if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType, types.ModuleType)):
            m.update(b"T" + getattr(obj, "__module__", "").encode() + b"."
                     + getattr(obj, "__qualname__", getattr(obj, "__name__", "")).encode())
            return
        oid = id(obj)
        if oid in self.seen:
            m.update(b"R" + str(self.seen[oid]).encode())
            return
        self.seen[oid] = len(self.seen)
        self.keep.append(obj)
        if isinstance(obj, np.random.Generator):
            m.update(b"G")
            self._feed(obj.bit_generator.state, m)
        elif isinstance(obj, np.random.RandomState):
            m.update(b"S")
            self._feed(obj.get_state(legacy=False), m)
        elif isinstance(obj, random.Random):
            m.update(b"P")
            self._feed(obj.getstate(), m)
        elif isinstance(obj, np.ndarray):
            m.update(b"N" + str(obj.dtype).encode() + str(obj.shape).encode())
            if obj.dtype == object:
                for v in obj.ravel(order="C"):        # recurse: object state and aliasing count
                    self._feed(v, m)
            else:
                m.update(np.ascontiguousarray(obj).tobytes())
        elif isinstance(obj, np.generic):
            m.update(b"n" + repr(obj.item()).encode())
        elif isinstance(obj, types.MethodType):
            m.update(b"M" + obj.__func__.__qualname__.encode())
            self._feed(obj.__self__, m)
        elif isinstance(obj, (dict, types.MappingProxyType)):
            if isinstance(obj, types.MappingProxyType) and not _static_mapping(obj):
                raise CheckpointError("cannot verify a mappingproxy with mutable contents")
            m.update(b"D" + type(obj).__name__.encode())
            # Insertion order is real state (RNG draws over list(dict)); keep it in both modes.
            for k, v in obj.items():
                self._feed(k, m)
                self._feed(v, m)
            m.update(b"d")
        elif isinstance(obj, (list, tuple)):
            m.update(b"L" + type(obj).__name__.encode())
            for v in obj:
                self._feed(v, m)
            m.update(b"l")
        elif isinstance(obj, (set, frozenset)):
            # Only immutable value elements: an identity-bearing element could alias state outside
            # the set, which the order-free content mode cannot encode. Fail closed.
            if not all(_is_value(v) for v in obj):
                raise CheckpointError(f"unsupported {type(obj).__name__} element type for verification")
            m.update(b"Z" + type(obj).__name__.encode())
            if self.ordered:
                for v in obj:
                    self._feed(v, m)
            else:
                for d in sorted(_Canon(False, {}).h(v) for v in obj):
                    m.update(d)
            m.update(b"z")
        elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time, datetime.timedelta)):
            m.update(b"t" + type(obj).__name__.encode() + b":" + repr(obj).encode())
        elif isinstance(obj, bytearray):
            m.update(b"b" + bytes(obj))
        elif isinstance(obj, collections.deque):
            m.update(b"Q" + repr(obj.maxlen).encode())
            for v in obj:
                self._feed(v, m)
            m.update(b"q")
        elif isinstance(obj, _UNSUPPORTED):
            raise CheckpointError(f"unsupported type in match state: {type(obj).__qualname__}")
        elif isinstance(obj, functools.partial):
            m.update(b"F")
            self._feed((obj.func, obj.args, obj.keywords), m)
        else:
            m.update(b"O" + type(obj).__module__.encode() + b"." + type(obj).__qualname__.encode())
            state = getattr(obj, "__dict__", None)
            slots = []
            for cls in type(obj).__mro__:
                declared = getattr(cls, "__slots__", ()) or ()
                if isinstance(declared, str):          # __slots__ = "name" declares ONE slot
                    declared = (declared,)
                slots += [x for x in declared if isinstance(x, str) and x not in ("__dict__", "__weakref__")]
            if state is not None:
                self._feed(state, m)
            for slot in slots:
                if hasattr(obj, slot):
                    m.update(b"s" + slot.encode())
                    self._feed(getattr(obj, slot), m)
            if state is None and not slots:
                # A C-implemented object whose state lives elsewhere: never guess "equal".
                try:
                    reduced = obj.__reduce_ex__(CHECKPOINT_PROTOCOL)
                except Exception as exc:
                    raise CheckpointError(f"cannot fingerprint {type(obj).__qualname__}") from exc
                self._feed(reduced[1:] if isinstance(reduced, tuple) else reduced, m)
            m.update(b"o")


def state_fingerprint(bundle: dict, handles: dict | None = None,
                      global_rng: dict | None = None) -> dict:
    """Section digests (hex) of a bundle in both modes, plus the global RNG states."""
    handles = handles or {}
    out = {}
    for mode, ordered in (("content", False), ("ordered", True)):
        sections = {}
        canon = _Canon(ordered, handles)   # one walker: cross-section aliasing is encoded too
        for name in sorted(bundle):
            sections[name] = canon.h(bundle[name]).hex()
        sections["global_rng"] = _Canon(ordered, {}).h(global_rng or global_rng_state()).hex()
        sections["__all__"] = hashlib.sha256(
            "".join(f"{k}={v};" for k, v in sorted(sections.items())).encode()).hexdigest()
        out[mode] = sections
    return out


def compare_fingerprints(a: dict, b: dict) -> dict:
    """{mode: [differing section names]} -- empty lists mean identical."""
    return {mode: sorted(k for k in set(a[mode]) | set(b[mode])
                         if a[mode].get(k) != b[mode].get(k)) for mode in ("content", "ordered")}
