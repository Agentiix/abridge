# Project conventions — abridge

## Scope

abridge is an **Anthropic-shaped HTTP proxy for OpenAI-compatible
providers**, bridged across the agentix sandbox boundary.

The agent (claude CLI, anthropic SDK, ...) inside the sandbox sees a
normal `/v1/messages` endpoint. The actual provider call (OpenAI,
OpenRouter, model-eval, ...) lives on the **host** — the sandbox-side
service forwards translated requests over the `/abridge` Socket.IO
namespace.

## Architecture (one diagram)

```text
sandbox                                 host
─────────────────────────────────       ──────────────────────────────
claude CLI                               openai.AsyncOpenAI
    │                                          ▲
    │ POST /v1/messages                        │ chat.completions.create
    ▼                                          │
abridge service          /abridge          AnthropicGateway
(FastAPI + uvicorn)  ◄───  SIO   ───►    (AsyncClientNamespace)
    │   anthropic_to_openai_body              │   openai_to_anthropic_response
    └──────────────────► ns.request ──────────┘
```

Two halves, three files:

- `abridge/_service.py` — sandbox side. FastAPI app + uvicorn launcher
  + `agentix.Namespace` subclass. `start_anthropic_service(...)` boots
  one server and returns its URL; `stop_anthropic_service(handle)`
  tears down. The service handles `/v1/messages` and
  `/v1/messages/count_tokens`.
- `abridge/_gateway.py` — host side. `AnthropicGateway` extends
  `agentix.AsyncClientNamespace`, owns an `AsyncOpenAI` client, and
  answers `openai_complete` requests from the sandbox.
- `abridge/_translate.py` — pure-function Anthropic ↔ OpenAI
  converters. No HTTP, no I/O.

## Two Ideas, Not Three

abridge does exactly two things:

1. **Translate** Anthropic `/v1/messages` ↔ OpenAI `/chat/completions`
   (including SSE streaming format on Anthropic's side).
2. **Bridge** the sandbox → host call so credentials live host-side.

It owns nothing else. No correlator, no rollout schema, no RL training
sinks, no JSONL writer. That earlier design is gone.

## Streaming

When the agent asks for `stream=true`, abridge **buffers** the upstream
OpenAI non-streaming response and replays it as Anthropic SSE on the
agent side. We don't pipe chunk-by-chunk OpenAI streaming through SIO
yet — adding that is a backlog item.

## Relationship to Agentix

abridge depends on `agentix` only:

- `agentix.Namespace` for the sandbox-side namespace
- `agentix.AsyncClientNamespace` for the host-side namespace
- `agentix.register_namespace` to attach the sandbox namespace
- `agentix.RuntimeClient.register_namespace` to attach the host
  namespace before connecting

No special agentix entry-point axis is involved. The user registers
the host gateway with their `RuntimeClient` themselves, and calls
`c.remote(abridge.start_anthropic_service, ...)` to spin up the
sandbox service.

## Reserved Namespace

abridge owns the `/abridge` Socket.IO namespace. Events on that path:

| Event                    | Direction       | Payload shape                                  |
|--------------------------|-----------------|------------------------------------------------|
| `openai_complete`        | sandbox → host  | `{request_id, data: <openai-chat-body>}`       |
| `openai_complete:result` | host → sandbox  | `{request_id, value: <openai-chat-response>}`  |
| `openai_complete:error`  | host → sandbox  | `{request_id, error: {type, message}}`         |

Round-trip correlation by `request_id` is handled by
`agentix.Namespace.request(...)`. abridge plugin code doesn't touch
the request_id directly.

## Project Management — uv

Same convention as `agentix`:

- `uv sync` to install / refresh the venv
- `uv add <pkg>` for runtime deps, `uv add --dev <pkg>` for dev deps
- Never `pip install` directly
- `uv build` for packaging (when we eventually publish)
- We pull `agentix` from github during development via
  `[tool.uv.sources]` rather than PyPI

## Typing — No Bypass

`pyright abridge` must stay at zero errors. Fix the root cause, do
not `# type: ignore`. See agentix's CLAUDE.md for the full convention.

## No Backward Compatibility Shims

This repo is in active design (same posture as agentix). Breaking
changes are fine. Rename by deleting; don't accept both names.

## Distribution

abridge is **not** on PyPI during active development. Consumers pull
it via `[tool.uv.sources]` git URLs:

```toml
[project]
dependencies = ["abridge"]

[tool.uv.sources]
abridge = { git = "https://github.com/Agentiix/abridge.git" }
```
