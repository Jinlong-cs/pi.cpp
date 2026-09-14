"""Deployment manifest v1: load, validate, and represent model packages.

The manifest is the single source of truth a package ships: model identity,
engine identity + rebuild recipe, loop shape, runtime options, gates, and
the acceptance lock. Fail-fast validation rejects a structurally broken
package with a precise message instead of degrading later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "picpp.deployment-manifest.v1"

_HEX64 = 64


@dataclass(frozen=True)
class EngineEntry:
    path: str
    size_bytes: int
    sha256: str
    recipe: dict[str, Any]


@dataclass(frozen=True)
class LoopSpec:
    kind: str
    step_stage: str
    steps: int
    graph_capture: str


@dataclass(frozen=True)
class LockRecord:
    sha256: str
    accepted_at: str
    accepted_d10_p50_ms: float | None
    parity: dict[str, float] | None
    evidence: dict[str, str]


@dataclass(frozen=True)
class DeploymentManifest:
    model: dict[str, str]
    export_manifest: dict[str, str]
    engines: dict[str, EngineEntry]
    loop: LoopSpec
    runtime: dict[str, Any]
    gates: dict[str, Any] | None = None
    lock: LockRecord | None = None


def _fail(message: str) -> None:
    raise ValueError(message)


def _require_str(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        _fail(f"{where}.{key}: expected a non-empty string, got {value!r}")
    return value


def _require_sha256(mapping: dict[str, Any], key: str, where: str) -> str:
    value = _require_str(mapping, key, where)
    if len(value) != _HEX64 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{where}.{key}: expected a 64-char lowercase sha256 hex, got {value!r}")
    return value


def _validate_engines(raw: dict[str, Any]) -> dict[str, EngineEntry]:
    raw_engines = raw.get("engines")
    if not isinstance(raw_engines, dict) or not raw_engines:
        _fail("engines: expected a non-empty mapping of stage name -> engine entry")
    engines: dict[str, EngineEntry] = {}
    for name, entry in raw_engines.items():
        if not isinstance(entry, dict):
            _fail(f"engines.{name}: expected a mapping")
        engines[name] = EngineEntry(
            path=_require_str(entry, "path", f"engines.{name}"),
            size_bytes=_require_int(entry, "size_bytes", f"engines.{name}"),
            sha256=_require_sha256(entry, "sha256", f"engines.{name}"),
            recipe=_require_dict(entry, "recipe", f"engines.{name}"),
        )
        if engines[name].size_bytes <= 0:
            _fail(f"engines.{name}.size_bytes: expected a positive integer")
        if not engines[name].recipe:
            _fail(f"engines.{name}.recipe: expected a non-empty rebuild recipe")
    return engines


def _require_int(mapping: dict[str, Any], key: str, where: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(f"{where}.{key}: expected an integer, got {value!r}")
    return value


def _require_dict(mapping: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict) or not value:
        _fail(f"{where}.{key}: expected a non-empty mapping")
    return value


def validate(raw: dict[str, Any]) -> None:
    """Raise ValueError on the first structural violation."""
    if not isinstance(raw, dict):
        _fail("manifest: expected a JSON object")
    if raw.get("schema") != SCHEMA:
        _fail(f"schema: expected {SCHEMA!r}, got {raw.get('schema')!r}")

    model = _require_dict(raw, "model", "manifest")
    for key in ("family", "variant", "display_name", "runtime"):
        _require_str(model, key, "model")

    export = _require_dict(raw, "export_manifest", "manifest")
    _require_str(export, "path", "export_manifest")
    _require_sha256(export, "sha256", "export_manifest")

    engines = _validate_engines(raw)

    if "authorization" in raw:
        authorization = _require_dict(raw, "authorization", "manifest")
        if "required" in authorization and not isinstance(authorization["required"], bool):
            _fail(f"authorization.required: expected a bool, got {authorization['required']!r}")

    loop = _require_dict(raw, "loop", "manifest")
    if _require_str(loop, "kind", "loop") != "host_denoise":
        _fail(f"loop.kind: expected 'host_denoise', got {loop['kind']!r}")
    step_stage = _require_str(loop, "step_stage", "loop")
    if step_stage not in engines:
        _fail(f"loop.step_stage: {step_stage!r} is not an engine stage ({sorted(engines)})")
    if _require_int(loop, "steps", "loop") <= 0:
        _fail("loop.steps: expected a positive integer")
    graph_capture = _require_str(loop, "graph_capture", "loop")
    if graph_capture not in ("off", "auto"):
        _fail(f"loop.graph_capture: expected 'off' or 'auto', got {graph_capture!r}")

    _require_dict(raw, "runtime", "manifest")
    _require_str(raw["runtime"], "adapter", "runtime")

    gates = raw.get("gates")
    if gates is not None and not isinstance(gates, dict):
        _fail("gates: expected a mapping when present")

    lock = raw.get("lock")
    if lock is not None:
        if not isinstance(lock, dict):
            _fail("lock: expected a mapping when present")
        _require_sha256(lock, "sha256", "lock")
        _require_str(lock, "accepted_at", "lock")


def load(path: Path) -> DeploymentManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate(raw)
    return DeploymentManifest(
        model=raw["model"],
        export_manifest=raw["export_manifest"],
        engines={name: EngineEntry(**entry) for name, entry in raw["engines"].items()},
        loop=LoopSpec(**raw["loop"]),
        runtime=raw["runtime"],
        gates=raw.get("gates"),
        lock=LockRecord(**raw["lock"]) if raw.get("lock") else None,
    )
