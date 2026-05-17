"""JsonlSink round-trip — push a rollout, read the file back."""

from __future__ import annotations

import json

import pytest

from abridge.models import LLMTurn, Rollout
from abridge.sink import JsonlSink, Sink


def test_jsonl_sink_is_a_sink():
    # Sink is a runtime_checkable Protocol — JsonlSink should structurally match.
    assert isinstance(JsonlSink("/tmp/whatever"), Sink)


@pytest.mark.asyncio
async def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    async with JsonlSink(path) as sink:
        await sink.push(Rollout(
            call_id="c1",
            status="closed",
            reward=0.7,
            llm_turns=[LLMTurn(
                provider="anthropic", path="/v1/messages",
                request_body={"msg": "hi"}, response_body={"msg": "hello"},
                status=200, started_at=1.0, ended_at=1.5, pending=False,
            )],
        ))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["call_id"] == "c1"
    assert obj["reward"] == 0.7
    assert obj["llm_turns"][0]["status"] == 200
