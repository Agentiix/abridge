"""Rollout data model — the shape every sink consumes.

A `Rollout` is the result of correlating every `TraceEvent` that shares
a single `call_id`. It carries:

  - the ordered LLM turns (`request` + matched `response`)
  - the ordered tool calls (`tool_call` + matched `tool_result`)
  - terminal `reward` events
  - the raw underlying steps, untouched, so sinks can opt out of the
    correlator's interpretation when their framework needs something
    finer-grained (per-token logprobs, custom kinds, …)

This module is RL-framework-agnostic. Translating a `Rollout` into the
training-sample shape of any specific framework is the adapter's job —
abridge ships none.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """One raw trace event, preserved verbatim from the Agentix wire.

    Sinks that want full fidelity (every `trace.emit` the namespace ever
    fired) iterate `rollout.steps` instead of the interpreted lists.
    """

    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float
    source: str | None = None


class LLMTurn(BaseModel):
    """One LLM round-trip — a request paired with its response.

    The provider's own message / completion shape is preserved under
    `request_body` / `response_body`; abridge does not normalize across
    Anthropic vs OpenAI vs other on-the-wire formats, because every RL
    framework expects a different normalization and doing it here would
    just push the inverse mapping into every adapter. The fields we DO
    name (`provider`, `path`) are stable across both.
    """

    provider: str                          # "anthropic", "openai", …
    path: str                              # "/v1/messages", "/v1/chat/completions", …
    request_body: Any                      # provider-shaped messages / params
    response_body: Any = None              # set once the matching response arrives
    status: int | None = None              # HTTP status from the response
    started_at: float                      # request emit timestamp
    ended_at: float | None = None          # response emit timestamp
    pending: bool = True                   # True until a response trace lands


class ToolCall(BaseModel):
    """One tool invocation — a `tool_call` event paired with its `tool_result`.

    Namespaces emit `tool_call` / `tool_result` voluntarily; rollouts
    without these events just have an empty `tool_calls` list. The
    correlator matches by `id` if present in the payload, otherwise
    pairs in arrival order.
    """

    name: str | None = None
    arguments: Any = None
    result: Any = None
    id: str | None = None
    started_at: float
    ended_at: float | None = None
    pending: bool = True


class Rollout(BaseModel):
    """All trace events for one Agentix `call_id`, in arrival order, plus
    the correlator's interpretation of them.

    `status` reflects whether the rollout's owning remote call has
    finished — `open` while events are still arriving, `closed` once
    the correlator's flush condition fires (terminal reward, sentinel
    event, or `close()` on the correlator). Sinks should only push
    `closed` rollouts unless they explicitly handle partial state.
    """

    call_id: str
    status: Literal["open", "closed"] = "open"
    steps: list[Step] = Field(default_factory=list)
    llm_turns: list[LLMTurn] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
