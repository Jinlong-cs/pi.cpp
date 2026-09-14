"""Model package primitives for picpp package: shared helpers."""

from __future__ import annotations

from pi_cpp.package.gates import FAMILY_GATES, gates_for
from pi_cpp.package.lock import (
    compute_package_hash,
    engine_hashes,
    sha256_file,
    verify_integrity,
)
from pi_cpp.package.manifest import (
    SCHEMA,
    DeploymentManifest,
    EngineEntry,
    LockRecord,
    LoopSpec,
    load,
    validate,
)

__all__ = [
    "FAMILY_GATES",
    "SCHEMA",
    "DeploymentManifest",
    "EngineEntry",
    "LockRecord",
    "LoopSpec",
    "compute_package_hash",
    "engine_hashes",
    "gates_for",
    "load",
    "sha256_file",
    "validate",
    "verify_integrity",
]
