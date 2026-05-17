"""End-to-end pipeline: tap → correlate → sink.

`abridge.run(url, sink)` is the one-call entrypoint that ties the three
layers together. Most users only ever need this and the `Sink` Protocol
from their adapter package.
"""

from __future__ import annotations

import contextlib

from abridge.correlator import correlate
from abridge.sink import Sink
from abridge.tap import tap


async def run(
    runtime_url: str,
    sink: Sink,
    *,
    kind: str | None = None,
    call_id: str | None = None,
    include_open: bool = False,
) -> None:
    """Stream rollouts from `runtime_url` into `sink` until the runtime
    closes the connection.

    `include_open` controls whether the final flush at stream end pushes
    rollouts that never received a terminal `reward` / `rollout_end`.
    Default off — most RL trainers can't use partial rollouts and would
    only get confused by them.

    Supports both bare-Protocol sinks (just `push`) and context-manager
    sinks (with `__aenter__` / `__aexit__`). The branching is mechanical:
    if the sink has `__aenter__`, enter it; otherwise call `push`
    directly. No `Sink` subclass tax.
    """
    async with _maybe_enter(sink) as bound_sink:
        async with tap(runtime_url, kind=kind, call_id=call_id) as events:
            async for rollout in correlate(events):
                if rollout.status == "open" and not include_open:
                    continue
                await bound_sink.push(rollout)


@contextlib.asynccontextmanager
async def _maybe_enter(sink: Sink):
    aenter = getattr(sink, "__aenter__", None)
    aexit = getattr(sink, "__aexit__", None)
    if aenter is None or aexit is None:
        yield sink
        return
    bound = await aenter()
    try:
        yield bound
    finally:
        await aexit(None, None, None)
