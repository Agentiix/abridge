"""Sink Protocol — the contract every RL-framework adapter implements.

A sink is "the thing that receives `Rollout`s and does framework-specific
stuff with them." abridge ships **no** framework-specific sinks — each
RL framework gets its own adapter package that depends on abridge. This
keeps the abridge dependency surface free of RL-framework machinery
(data buffers, rollout datasets, …) which tends to drag in CUDA,
Megatron, Ray, etc.

The Protocol is intentionally minimal — one async method. A sink that
needs setup/teardown can implement `__aenter__` / `__aexit__` and
`abridge.run(...)` will use it; otherwise it's just `push(rollout)`.

Built-in for local inspection and tests only: `JsonlSink`, which writes
each rollout as one JSON line. Useful for `abridge tap --writer jsonl
--out rollouts.jsonl` to verify the trace pipeline before wiring an
adapter in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from abridge.models import Rollout


@runtime_checkable
class Sink(Protocol):
    """Receives correlated rollouts. Implementations live in adapter packages.

    The framework calls `push` once per closed rollout, in the order the
    correlator yields them. If the sink raises, `abridge.run` propagates
    the exception — the caller decides whether to restart the pipeline
    or abort. Sinks that need to batch / debounce should do so inside
    `push` and override their own flush semantics; abridge does not
    impose a batching layer (every framework wants a different one).
    """

    async def push(self, rollout: Rollout) -> None: ...


class JsonlSink:
    """Append one JSON object per rollout to a file. For smoke tests and
    local inspection — not a production training pipeline.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    async def __aenter__(self) -> "JsonlSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    async def push(self, rollout: Rollout) -> None:
        if self._fh is None:
            # Allow ad-hoc use without `async with`. Open on first push.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(rollout.model_dump_json())
        self._fh.write("\n")
        self._fh.flush()
