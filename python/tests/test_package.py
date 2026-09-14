from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pi_cpp.commands.package import run_install, run_list, run_lock
from pi_cpp.package.gates import gates_for
from pi_cpp.package.lock import compute_package_hash, engine_hashes, verify_integrity
from pi_cpp.package.manifest import SCHEMA, load, validate

PREFIX_BYTES = b"prefix-engine-bytes"
SUFFIX_BYTES = b"suffix-engine-bytes"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_manifest_dict() -> dict:
    return {
        "schema": SCHEMA,
        "model": {"family": "pi06", "variant": "rtc", "display_name": "test_pi06_rtc", "runtime": "pi06_offline"},
        "export_manifest": {"path": "export_manifest.json", "sha256": _digest(b'{"contract": {}}\n')},
        "engines": {
            "prefix_embed": {
                "path": "engines/prefix.engine",
                "size_bytes": len(PREFIX_BYTES),
                "sha256": _digest(PREFIX_BYTES),
                "recipe": {"tool": "trtexec", "profile": "fp16", "workspace_gb": 4},
            },
            "suffix_step": {
                "path": "engines/suffix.engine",
                "size_bytes": len(SUFFIX_BYTES),
                "sha256": _digest(SUFFIX_BYTES),
                "recipe": {"tool": "python", "profile": "hybrid", "keep_patterns": ["/mlp/"]},
            },
        },
        "loop": {"kind": "host_denoise", "step_stage": "suffix_step", "steps": 10, "graph_capture": "auto"},
        "runtime": {"adapter": "pi_cpp.pi06_rtc", "rtc": {"enabled": True, "max_delay": 4}},
    }


def make_package(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "pkg"
    (package / "engines").mkdir(parents=True)
    (package / "engines" / "prefix.engine").write_bytes(PREFIX_BYTES)
    (package / "engines" / "suffix.engine").write_bytes(SUFFIX_BYTES)
    (package / "adapter.py").write_text("# bundled adapter\n")
    (package / "export_manifest.json").write_text('{"contract": {}}\n')
    manifest_path = package / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(make_manifest_dict(), indent=2) + "\n")
    return package, manifest_path


def test_load_roundtrip(tmp_path):
    _package, manifest_path = make_package(tmp_path)
    manifest = load(manifest_path)
    assert manifest.model["family"] == "pi06"
    assert set(manifest.engines) == {"prefix_embed", "suffix_step"}
    assert manifest.loop.steps == 10
    assert manifest.lock is None


def test_validate_rejects_wrong_schema(tmp_path):
    raw = make_manifest_dict()
    raw["schema"] = "picpp.deployment-manifest.v0"
    with pytest.raises(ValueError, match="schema"):
        validate(raw)


def test_validate_rejects_bad_sha256(tmp_path):
    raw = make_manifest_dict()
    raw["engines"]["prefix_embed"]["sha256"] = "abcd"
    with pytest.raises(ValueError, match="sha256"):
        validate(raw)


def test_validate_rejects_unknown_step_stage(tmp_path):
    raw = make_manifest_dict()
    raw["loop"]["step_stage"] = "prefix_lm"
    with pytest.raises(ValueError, match="step_stage"):
        validate(raw)


def test_gates_pi06_default_and_override():
    assert gates_for("pi06", None)["parity"]["raw"] == 0.1
    merged = gates_for("pi06", {"parity": {"raw": 0.05}})
    assert merged["parity"]["raw"] == 0.05
    assert merged["parity"]["rel_l2"] == 0.02


def test_gates_unknown_family_requires_explicit():
    with pytest.raises(ValueError, match="declare gates"):
        gates_for("unknown_family", None)
    assert gates_for("unknown_family", {"parity": {"raw": 0.1}}) == {"parity": {"raw": 0.1}}


def test_package_hash_stable_and_sensitive(tmp_path):
    package, manifest_path = make_package(tmp_path)
    manifest = load(manifest_path)
    hashes = engine_hashes(package, manifest)
    assert compute_package_hash(package, manifest, hashes) == compute_package_hash(package, manifest, hashes)
    (package / "engines" / "prefix.engine").write_bytes(b"tampered")
    assert compute_package_hash(package, manifest, engine_hashes(package, manifest)) != compute_package_hash(
        package, manifest, hashes
    )


def test_verify_ok_and_tamper(tmp_path):
    package, manifest_path = make_package(tmp_path)
    ok, problems = verify_integrity(package, load(manifest_path))
    assert ok and not problems
    (package / "engines" / "suffix.engine").write_bytes(b"tampered-suffix")
    ok, problems = verify_integrity(package, load(manifest_path))
    assert not ok
    assert any("sha256 mismatch" in p for p in problems)


def test_lock_roundtrip(tmp_path):
    package, manifest_path = make_package(tmp_path)
    assert run_lock(package, None, None) == 0
    manifest = load(manifest_path)
    assert manifest.lock is not None
    assert verify_integrity(package, manifest)[0]
    (package / "adapter.py").write_text("# changed adapter\n")
    ok, problems = verify_integrity(package, load(manifest_path))
    assert not ok
    assert any("lock" in p for p in problems)


def test_install_and_list(tmp_path):
    package, _ = make_package(tmp_path)
    root = tmp_path / "root"
    assert run_install(package, root, force=False) == 0
    installed = root / "test_pi06_rtc"
    assert (installed / "engines" / "prefix.engine").is_file()
    assert run_list(root) == 0
    (package / "engines" / "prefix.engine").write_bytes(b"broken")
    assert run_install(package, tmp_path / "root2", force=False) == 1
    assert run_install(package, tmp_path / "root3", force=True) == 0
