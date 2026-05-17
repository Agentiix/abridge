"""Trace tap — subscribe to an Agentix runtime's trace stream.

`tap(url)` is an async context manager that connects to the runtime's
Socket.IO `trace` channel (via `agentix.runtime.client.RuntimeClient`)
and yields raw `TraceEvent`s as they fire inside the sandbox. No
correlation, no filtering, no buffering beyond what the underlying
client already does.

Two reasons this is a thin wrapper around `RuntimeClient.traces()`
rather than re-implementing the Socket.IO subscription:

  1. `RuntimeClient` is the canonical client for everything the runtime
     server exposes. Touching its private Socket.IO state from outside
     would couple abridge to the wire layout — the wrapper means a wire
     change in Agentix is invisible to us.
  2. The same connection multiplexes traces alongside any other RPC, so
     callers who already hold a `RuntimeClient` can plug it in directly
     via `tap_client(client)` rather than opening a second socket.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentix.runtime.client.client import RuntimeClient
    from agentix.runtime.models import TraceEvent


@contextlib.asynccontextmanager
async def tap(
    runtime_url: str,
    *,
    kind: str | None = None,
    call_id: str | None = None,
) -> AsyncIterator[AsyncIterator["TraceEvent"]]:
    """Connect to an Agentix runtime and yield its trace stream.

    Usage:

    ```python
    async with abridge.tap("http://localhost:8000") as events:
        async for ev in events:
            ...
    ```

    `kind` and `call_id` are forwarded to `RuntimeClient.traces()` as
    client-side filters — the server still broadcasts everything; the
    filter just drops events before they reach the iterator.
    """
    from agentix.runtime.client.client import RuntimeClient

    async with RuntimeClient(runtime_url) as client:
        yield client.traces(kind=kind, call_id=call_id)


async def tap_client(
    client: "RuntimeClient",
    *,
    kind: str | None = None,
    call_id: str | None = None,
) -> AsyncIterator["TraceEvent"]:
    """Tap an already-open `RuntimeClient`. Useful when the caller already
    holds the client for other RPC and wants to share the Socket.IO
    connection rather than opening a second one.

    Returns the iterator directly — no context manager, because the
    caller owns the client's lifetime.
    """
    async for ev in client.traces(kind=kind, call_id=call_id):
        yield ev
