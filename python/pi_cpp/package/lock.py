"""Package integrity: per-engine sha256, adapter/export hashing, package lock.

The lock = one canonical package hash (sha256 over the manifest without
its lock section + per-engine hashes + adapter.py + export manifest) plus
the per-engine table inside the manifest itself, so a single broken engine
is pinpointed while the package hash guarantees whole-package consistency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pi_cpp.package.manifest import SCHEMA, DeploymentManifest

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_hashes(package_dir: Path, manifest: DeploymentManifest) -> dict[str, str]:
    return {name: sha256_file(package_dir / entry.path) for name, entry in manifest.engines.items()}


def canonical_manifest(manifest: DeploymentManifest) -> dict[str, Any]:
    """The manifest minus the lock section, in a deterministic dict form."""
    return {
        "schema": SCHEMA,
        "model": manifest.model,
        "export_manifest": manifest.export_manifest,
        "engines": {
            name: {
                "path": entry.path,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
                "recipe": entry.recipe,
            }
            for name, entry in sorted(manifest.engines.items())
        },
        "loop": {
            "kind": manifest.loop.kind,
            "step_stage": manifest.loop.step_stage,
            "steps": manifest.loop.steps,
            "graph_capture": manifest.loop.graph_capture,
        },
        "runtime": manifest.runtime,
        "gates": manifest.gates,
    }


def compute_package_hash(package_dir: Path, manifest: DeploymentManifest, hashes: dict[str, str]) -> str:
    adapter = package_dir / "adapter.py"
    export = package_dir / manifest.export_manifest["path"]
    payload = {
        "manifest": canonical_manifest(manifest),
        "engines": hashes,
        "adapter": sha256_file(adapter),
        "export_manifest": sha256_file(export),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_integrity(package_dir: Path, manifest: DeploymentManifest) -> tuple[bool, list[str]]:
    """File-level verification: existence, size, sha256, and the lock."""
    problems: list[str] = []
    for name, entry in manifest.engines.items():
        engine = package_dir / entry.path
        if not engine.is_file():
            problems.append(f"engines.{name}: {entry.path} missing")
            continue
        if engine.stat().st_size != entry.size_bytes:
            problems.append(f"engines.{name}: size {engine.stat().st_size} != manifest {entry.size_bytes}")
        if sha256_file(engine) != entry.sha256:
            problems.append(f"engines.{name}: sha256 mismatch")
    adapter = package_dir / "adapter.py"
    if not adapter.is_file():
        problems.append("adapter.py missing")
    export = package_dir / manifest.export_manifest["path"]
    if not export.is_file():
        problems.append(f"export_manifest: {manifest.export_manifest['path']} missing")
    elif sha256_file(export) != manifest.export_manifest["sha256"]:
        problems.append("export_manifest: sha256 mismatch")
    if manifest.lock is not None:
        current = compute_package_hash(package_dir, manifest, engine_hashes(package_dir, manifest))
        if current != manifest.lock.sha256:
            problems.append(f"lock: package hash {current[:12]} != locked {manifest.lock.sha256[:12]}")
    return not problems, problems
