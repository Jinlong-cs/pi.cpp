"""LIBERO closed-loop websocket client example."""

from __future__ import annotations
import argparse
from pi_cpp.commands.client import run_client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    return run_client(host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
