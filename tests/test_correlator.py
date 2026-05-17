"""Correlator unit tests — no runtime, no Socket.IO. We synthesise
TraceEvents and assert the assembled Rollouts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from abridge.correlator import correlate


class _Ev:
    """Lightweight stand-in for `agentix.runtime.models.TraceEvent` — only
    the attributes the correlator reads. Keeps the test file free of an
    `agentix` import (the package is optional at unit-test time).
    """

    def __init__(self, kind, payload, call_id, ts=0.0, source="agentix.test"):
        self.kind = kind
        self.payload = payload
        self.call_id = call_id
        self.timestamp = ts
        self.source = source


async def _stream(events):
    for e in events:
        yield e


async def _collect(it: AsyncIterator) -> list:
    out = []
    async for x in it:
        out.append(x)
    return out


@pytest.mark.asyncio
async def test_llm_request_paired_with_response():
    events = [
        _Ev("llm_request", {"provider": "anthropic", "path": "/v1/messages", "body": {"a": 1}}, "c1", 1.0),
        _Ev("llm_response", {"provider": "anthropic", "status": 200, "body": {"b": 2}}, "c1", 1.5),
        _Ev("reward", {"value": 1.0}, "c1", 2.0),
    ]
    rollouts = await _collect(correlate(_stream(events)))
    assert len(rollouts) == 1
    r = rollouts[0]
    assert r.status == "closed"
    assert r.reward == 1.0
    assert len(r.llm_turns) == 1
    t = r.llm_turns[0]
    assert not t.pending
    assert t.request_body == {"a": 1}
    assert t.response_body == {"b": 2}
    assert t.status == 200


@pytest.mark.asyncio
async def test_tool_call_paired_by_id():
    events = [
        _Ev("tool_call", {"id": "x", "name": "bash", "arguments": {"cmd": "ls"}}, "c1", 1.0),
        _Ev("tool_call", {"id": "y", "name": "bash", "arguments": {"cmd": "pwd"}}, "c1", 1.1),
        _Ev("tool_result", {"id": "y", "result": "/home"}, "c1", 1.2),
        _Ev("tool_result", {"id": "x", "result": "a\nb"}, "c1", 1.3),
        _Ev("rollout_end", {"reward": 0.5}, "c1", 2.0),
    ]
    rollouts = await _collect(correlate(_stream(events)))
    assert len(rollouts) == 1
    by_id = {tc.id: tc for tc in rollouts[0].tool_calls}
    assert by_id["x"].result == "a\nb"
    assert by_id["y"].result == "/home"
    assert rollouts[0].reward == 0.5


@pytest.mark.asyncio
async def test_interleaved_call_ids_kept_separate():
    events = [
        _Ev("llm_request", {"provider": "openai", "body": {}}, "c1", 1.0),
        _Ev("llm_request", {"provider": "openai", "body": {}}, "c2", 1.1),
        _Ev("llm_response", {"provider": "openai", "body": {}}, "c1", 1.2),
        _Ev("llm_response", {"provider": "openai", "body": {}}, "c2", 1.3),
        _Ev("reward", {"value": 1.0}, "c1", 2.0),
        _Ev("reward", {"value": 0.0}, "c2", 2.1),
    ]
    rollouts = await _collect(correlate(_stream(events)))
    assert len(rollouts) == 2
    by_cid = {r.call_id: r for r in rollouts}
    assert by_cid["c1"].reward == 1.0
    assert by_cid["c2"].reward == 0.0
    assert all(r.status == "closed" for r in rollouts)


@pytest.mark.asyncio
async def test_open_rollout_flushed_at_stream_end():
    events = [
        _Ev("llm_request", {"provider": "anthropic", "body": {}}, "c1", 1.0),
        # no response, no reward — correlator should still emit at flush
    ]
    rollouts = await _collect(correlate(_stream(events)))
    assert len(rollouts) == 1
    assert rollouts[0].status == "open"
    assert rollouts[0].llm_turns[0].pending is True


@pytest.mark.asyncio
async def test_call_id_none_is_dropped():
    events = [
        _Ev("llm_request", {"provider": "anthropic", "body": {}}, None, 1.0),
        _Ev("reward", {"value": 0.0}, "c1", 2.0),
    ]
    rollouts = await _collect(correlate(_stream(events)))
    assert len(rollouts) == 1
    assert rollouts[0].call_id == "c1"
    assert rollouts[0].llm_turns == []
