"""Correlator — assemble TraceEvents into Rollouts grouped by call_id.

The Agentix trace stream is event-at-a-time and interleaved across
concurrent rollouts. RL frameworks want one record per rollout, with
LLM requests joined to their responses and (optionally) tool calls
joined to their results. The correlator does that join in-memory.

`correlate(events)` is an async generator: feed it raw `TraceEvent`s,
get back `Rollout`s as each one closes. A rollout closes on:

  - a `rollout_end` sentinel event (the convention is `trace.emit(
    "rollout_end", {"reward": ...})` from the namespace), OR
  - a `reward` event (RL frameworks treat reward as terminal), OR
  - the upstream iterator ends (correlator flushes everything still
    pending).

If a rollout has neither sentinel and the trace stream stays open, it
sits in the open-rollouts table until the stream ends. The framework
that emits the traces is in control of when to close — abridge does
not impose a timeout, because rollouts can legitimately be long-lived.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING, Any

from abridge.models import LLMTurn, Rollout, Step, ToolCall

if TYPE_CHECKING:
    from agentix.runtime.models import TraceEvent

# Kinds the in-tree Agentix LLM proxy emits. Defined here (not imported)
# so abridge runs against a runtime that has the proxy disabled too —
# only one event of each pair is needed for the LLM-turn join to work.
_LLM_REQUEST = "llm_request"
_LLM_RESPONSE = "llm_response"

# Optional namespace-emitted kinds. Namespaces that don't emit them
# just get empty tool_calls; everything else still works.
_TOOL_CALL = "tool_call"
_TOOL_RESULT = "tool_result"

# Terminal sentinels.
_REWARD = "reward"
_ROLLOUT_END = "rollout_end"


async def correlate(events: AsyncIterable["TraceEvent"]) -> AsyncIterator[Rollout]:
    """Group `events` by `call_id` and yield a `Rollout` as each closes.

    Events with `call_id is None` are dropped — without a correlation
    key they can't be assigned to any rollout. (The Agentix dispatcher
    pins `call_id` into a contextvar for every dispatched call, so this
    only loses traces emitted from background tasks the framework
    doesn't know about.)
    """
    open_rollouts: dict[str, Rollout] = {}

    async for ev in events:
        cid = ev.call_id
        if cid is None:
            continue

        r = open_rollouts.get(cid)
        if r is None:
            r = Rollout(call_id=cid)
            open_rollouts[cid] = r

        r.steps.append(Step(
            kind=ev.kind, payload=ev.payload, timestamp=ev.timestamp, source=ev.source,
        ))

        _apply(r, ev.kind, ev.payload, ev.timestamp)

        if ev.kind in (_REWARD, _ROLLOUT_END):
            r.status = "closed"
            del open_rollouts[cid]
            yield r

    # Stream ended — flush whatever's still open as-is. Sinks decide
    # whether to keep `status="open"` records or drop them.
    for r in open_rollouts.values():
        yield r


def _apply(r: Rollout, kind: str, payload: dict[str, Any], ts: float) -> None:
    """Mutate `r` with the interpretation of one event. Anything we don't
    recognise stays in `r.steps` only — silently ignored at the
    interpretation level, never lost from the raw stream.
    """
    if kind == _LLM_REQUEST:
        r.llm_turns.append(LLMTurn(
            provider=str(payload.get("provider", "")),
            path=str(payload.get("path", "")),
            request_body=payload.get("body"),
            started_at=ts,
        ))
        return

    if kind == _LLM_RESPONSE:
        # Match to the most recent open (request-only) turn. Two
        # interleaved requests on the same call_id is unusual — the
        # dispatcher serialises calls per call_id — so FIFO is fine.
        for turn in reversed(r.llm_turns):
            if turn.pending and turn.provider == str(payload.get("provider", "")):
                turn.response_body = payload.get("body")
                turn.status = payload.get("status")
                turn.ended_at = ts
                turn.pending = False
                return
        # Orphan response (no matching request). Keep it as raw step;
        # the caller can still find it in `r.steps`.
        return

    if kind == _TOOL_CALL:
        r.tool_calls.append(ToolCall(
            name=payload.get("name"),
            arguments=payload.get("arguments"),
            id=payload.get("id"),
            started_at=ts,
        ))
        return

    if kind == _TOOL_RESULT:
        tid = payload.get("id")
        for tc in reversed(r.tool_calls):
            if tc.pending and (tid is None or tc.id == tid):
                tc.result = payload.get("result")
                tc.ended_at = ts
                tc.pending = False
                return
        return

    if kind == _REWARD:
        v = payload.get("value", payload.get("reward"))
        if isinstance(v, (int, float)):
            r.reward = float(v)
        return

    if kind == _ROLLOUT_END:
        if r.reward is None:
            v = payload.get("reward")
            if isinstance(v, (int, float)):
                r.reward = float(v)
        if isinstance(payload.get("metadata"), dict):
            r.metadata.update(payload["metadata"])
        return
