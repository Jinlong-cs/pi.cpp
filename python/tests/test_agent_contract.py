from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PRODUCT_COMMANDS = ("eval", "infer", "latency", "graph", "calibrate", "build", "server", "client")


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(ROOT / "python"), "PATH": str(ROOT / ".venv/bin")}
    return subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pi_cpp.cli", *args], capture_output=True, text=True, env=env, cwd=ROOT, check=False)


def test_cli_help_is_parseable():
    for command in PRODUCT_COMMANDS:
        result = _cli(command, "--help")
        assert result.returncode == 0, (command, result.stderr)
        assert "usage: picpp" in result.stdout, command


def test_cli_surface_is_product_only():
    result = _cli("--help")
    assert result.returncode == 0
    for command in PRODUCT_COMMANDS:
        assert command in result.stdout, command
    for removed in ("package", "verify"):
        assert removed not in result.stdout, removed


def test_cli_usage_error_exits_nonzero():
    result = _cli("build")
    assert result.returncode != 0
    assert "required" in (result.stderr + result.stdout)
