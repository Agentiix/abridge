"""`abridge` console script — `abridge tap <url> --writer jsonl --out <path>`.

The CLI is intentionally just the JSONL writer — production sinks ship
their own console scripts in their adapter packages because their flags
(buffer URL, model name, group keys, …) belong to the framework, not
abridge.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from abridge.runner import run
from abridge.sink import JsonlSink


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tap_p = sub.add_parser("tap", help="stream rollouts from a runtime into a writer")
    tap_p.add_argument("url", help="runtime base URL, e.g. http://localhost:8000")
    tap_p.add_argument("--writer", choices=["jsonl"], default="jsonl")
    tap_p.add_argument("--out", required=True, help="output path (jsonl writer)")
    tap_p.add_argument("--kind", default=None, help="filter by trace kind")
    tap_p.add_argument("--call-id", default=None, help="filter by call_id")
    tap_p.add_argument("--include-open", action="store_true",
                       help="emit rollouts that never received a terminal reward/rollout_end")

    args = parser.parse_args(argv)

    if args.cmd == "tap":
        sink = JsonlSink(args.out)
        try:
            asyncio.run(run(
                args.url, sink,
                kind=args.kind, call_id=args.call_id, include_open=args.include_open,
            ))
        except KeyboardInterrupt:
            return 130
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
