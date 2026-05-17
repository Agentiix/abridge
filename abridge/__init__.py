"""abridge — Agentix's host-side bridge from agent rollouts to RL training.

abridge is the official Agentix extension that turns the runtime's
trace stream (LLM requests, tool calls, rewards, arbitrary
`trace.emit(...)` events) into per-rollout records and hands them to
whatever data buffer an RL training loop expects. Framework-specific
adapters live in their own packages and depend on `abridge` for the
rollout type + sink Protocol; this package deliberately ships no
framework-specific wiring.

abridge is host-side, not sandbox-side — it plugs into Agentix via
plain Python (`agentix.trace.subscribe` and `RuntimeClient.traces()`),
not via the `agentix.namespace` / `agentix.deployment` entry-point
axes. See `CLAUDE.md` for the rationale.

Three layers, each independently usable:

  - `tap`         — connect to an Agentix runtime, stream raw `TraceEvent`s
  - `correlate`   — group events by `call_id` into a `Rollout`
  - `Sink`        — Protocol your sink implements; `abridge.run(...)` wires
                    a tap → correlator → sink pipeline end-to-end

A minimal smoke run:

```python
import asyncio, abridge

async def main():
    async with abridge.tap("http://localhost:8000") as events:
        async for rollout in abridge.correlate(events):
            print(rollout.call_id, len(rollout.steps))

asyncio.run(main())
```

For RL training: implement `Sink`, then `await abridge.run(url, sink)`.
"""

from __future__ import annotations

from abridge.correlator import correlate
from abridge.models import LLMTurn, Rollout, Step, ToolCall
from abridge.runner import run
from abridge.sink import JsonlSink, Sink
from abridge.tap import tap

__all__ = [
    "JsonlSink",
    "LLMTurn",
    "Rollout",
    "Sink",
    "Step",
    "ToolCall",
    "correlate",
    "run",
    "tap",
]
