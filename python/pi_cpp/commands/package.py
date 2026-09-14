"""picpp package: install / list / info / verify / lock / export model packages."""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import tyro

from pi_cpp.package.gates import gates_for
from pi_cpp.package.lock import compute_package_hash, engine_hashes, verify_integrity
from pi_cpp.package.manifest import load

MANIFEST_NAME = "deployment_manifest.json"


@dataclass(kw_only=True)
class PackageConfig:
    action: Annotated[
        Literal["install", "list", "info", "verify", "lock", "export"], tyro.conf.Positional
    ]
    package: Path | None = None
    root: Path | None = None
    out: Path | None = None
    force: bool = False


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_package(path: Path | None) -> tuple[Path, Path]:
    if path is None:
        raise ValueError("--package is required for this action")
    path = path.resolve()
    manifest_path = path if path.name == MANIFEST_NAME else path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"{manifest_path} not found")
    return path if path.is_dir() else path.parent, manifest_path


def _root(root: Path | None) -> Path:
    return (root or Path.home() / ".picpp" / "packages").resolve()


def run_install(package: Path, root: Path, force: bool) -> int:
    package_dir, manifest_path = _resolve_package(package)
    manifest = load(manifest_path)
    ok, problems = verify_integrity(package_dir, manifest)
    destination = root / manifest.model["display_name"]
    if not ok and not force:
        _print({"installed": False, "verify": problems})
        return 1
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(package_dir, destination)
    _print({"installed": True, "display_name": manifest.model["display_name"], "root": str(root), "verify": problems})
    return 0


def run_list(root: Path) -> int:
    _print(
        {
            "packages": [
                {"display_name": path.name, "manifest": str(path / MANIFEST_NAME)}
                for path in sorted(root.iterdir())
                if (path / MANIFEST_NAME).is_file()
            ]
        }
    )
    return 0


def run_info(package: Path) -> int:
    _, manifest_path = _resolve_package(package)
    manifest = load(manifest_path)
    _print(
        {
            "model": manifest.model,
            "engines": {
                name: {"path": entry.path, "size_bytes": entry.size_bytes, "recipe": entry.recipe}
                for name, entry in manifest.engines.items()
            },
            "loop": {"kind": manifest.loop.kind, "step_stage": manifest.loop.step_stage, "steps": manifest.loop.steps,
                     "graph_capture": manifest.loop.graph_capture},
            "runtime": manifest.runtime,
            "gates": gates_for(manifest.model["family"], manifest.gates),
            "locked": manifest.lock is not None,
        }
    )
    return 0


def run_verify(package: Path) -> int:
    package_dir, manifest_path = _resolve_package(package)
    manifest = load(manifest_path)
    ok, problems = verify_integrity(package_dir, manifest)
    _print({"ok": ok, "problems": problems, "locked": manifest.lock is not None})
    return 0 if ok else 1


def run_lock(package: Path, d10_p50_ms: float | None, parity: dict | None) -> int:
    package_dir, manifest_path = _resolve_package(package)
    manifest = load(manifest_path)
    ok, problems = verify_integrity(package_dir, manifest)
    if not ok:
        _print({"locked": False, "verify": problems})
        return 1
    hashes = engine_hashes(package_dir, manifest)
    package_digest = compute_package_hash(package_dir, manifest, hashes)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["lock"] = {
        "sha256": package_digest,
        "accepted_at": datetime.now(tz=timezone.utc).date().isoformat(),
        "accepted_d10_p50_ms": d10_p50_ms,
        "parity": parity,
        "evidence": {},
    }
    manifest_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    _print({"locked": True, "sha256": package_digest})
    return 0


def run_export(package: Path, out: Path) -> int:
    package_dir, manifest_path = _resolve_package(package)
    manifest = load(manifest_path)
    archive = (out or Path.cwd() / f"{manifest.model['display_name']}.tar.gz").resolve()
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(package_dir, arcname=manifest.model["display_name"])
    _print({"exported": True, "archive": str(archive)})
    return 0


def run_package(config: PackageConfig) -> int:
    action = config.action
    if action == "install":
        return run_install(config.package, _root(config.root), config.force)
    if action == "list":
        return run_list(_root(config.root))
    if action == "info":
        return run_info(config.package)
    if action == "verify":
        return run_verify(config.package)
    if action == "lock":
        return run_lock(config.package, None, None)
    if action == "export":
        return run_export(config.package, config.out)
    raise ValueError(f"unknown package action: {action}")
