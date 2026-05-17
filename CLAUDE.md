# Project conventions — abridge

## Scope

abridge is the **host-side bridge** between Agentix's trace stream and
any RL training framework's data buffer. It owns:

  - the `Rollout` data model
  - the trace tap (Socket.IO client to the Agentix runtime)
  - the correlator (TraceEvents → Rollouts)
  - the `Sink` Protocol

It does **not** own framework-specific sinks. Those live in their own
adapter packages that depend on abridge. Two reasons:

  1. Each RL framework drags in heavy native deps (CUDA, Megatron,
     Ray). abridge's whole point is to be the small layer everything
     depends on; piling adapters in here would invert that.
  2. Adapter authors iterate on the translation (rollout → framework
     sample) at a different cadence than the rollout schema itself.

If you find yourself adding an `import <some-rl-framework>` to a file
in this repo, stop. That code goes in `<framework>-abridge`.

## Relationship to Agentix

abridge is an Agentix extension. It is **not** an entry-point plugin
(`agentix.namespace` and `agentix.deployment` are the only two
entry-point axes; that line is held — see the parent project's
`feedback-plugins-only-cross-sandbox` memory). abridge attaches via
the host-side plain-Python hooks Agentix already exposes:

  - `agentix.runtime.client.RuntimeClient.traces()` for out-of-process
    trace consumption (the production path)
  - `agentix.trace.subscribe(fn)` for in-process consumption when the
    bridge runs in the runtime's own process (tests, embedded mode)

Adding a third entry-point axis just to install abridge would violate
the host-side rule for no payoff — `pip install abridge` plus the
caller importing it is already as low-friction as a plugin install,
and stays out of the framework's stable plugin contract.

## Composition over inheritance

Inherited from the parent Agentix project. `Sink` is a Protocol (not an
ABC); adapters implement it structurally without inheriting. The
correlator is a free function over a tagged-union of event kinds — no
class hierarchy.

The reverse — using inheritance — is allowed only when the
relationship is genuinely is-a. `JsonlSink` uses no base class; it just
structurally satisfies `Sink`.

## No backward compatibility

Same policy as the parent Agentix project: breaking changes are fine.
This package is in active design — rename, restructure, delete. Update
adapters in lockstep.

## File map

```
abridge/
  __init__.py      # public API: tap, correlate, run, Sink, JsonlSink, Rollout, ...
  models.py        # Rollout, Step, LLMTurn, ToolCall  (Pydantic v2)
  tap.py           # async context manager around RuntimeClient.traces()
  correlator.py    # TraceEvents → Rollouts grouped by call_id
  sink.py          # Sink Protocol + JsonlSink (smoke-test only)
  runner.py        # tap → correlate → sink one-call entrypoint
  cli.py           # `abridge tap <url> --writer jsonl --out ...`
tests/             # pytest, async; no Socket.IO — uses fake TraceEvent objects
```

## Test policy

- Correlator tests are pure-Python, no `agentix` import needed —
  they use a duck-typed `_Ev` class with the four attributes the
  correlator reads.
- The wire-level tap is *not* unit-tested here; it's a one-line
  wrapper around `RuntimeClient.traces()` and the integration test
  would just re-test the framework. End-to-end testing happens in
  Agentix's own test suite plus the adapter packages.

## Wire-level events the correlator recognises

| kind            | payload                                            | source             |
|-----------------|----------------------------------------------------|--------------------|
| `llm_request`   | `{provider, method, path, body}`                   | `llm_proxy` (built-in)|
| `llm_response`  | `{provider, status, body}`                         | `llm_proxy` (built-in)|
| `tool_call`     | `{id?, name, arguments}`                           | namespace-emitted  |
| `tool_result`   | `{id?, result}`                                    | namespace-emitted  |
| `reward`        | `{value}` or `{reward}`                            | namespace-emitted  |
| `rollout_end`   | `{reward?, metadata?}`                             | namespace-emitted  |

Everything else lands in `Rollout.steps` only — never lost, just not
interpreted into the typed fields. Adding a new typed field is one
branch in `correlator._apply` plus a Pydantic field on `Rollout`.
