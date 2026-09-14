from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest
from pi_cpp.commands.package import run_export, run_install, run_verify

ROOT = Path(__file__).resolve().parents[2]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {"PYTHONPATH": str(ROOT / "python"), "PATH": str(ROOT / ".venv/bin")}
    return subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pi_cpp.cli", *args], capture_output=True, text=True, env=env, cwd=ROOT, check=False)


def test_export_install_roundtrip(tmp_path):
    package, _ = make_package(tmp_path)
    archive = tmp_path / "bundle.tar.gz"
    assert run_export(package, archive) == 0
    assert archive.is_file()
    root = tmp_path / "root"
    root.mkdir()
    assert run_install(archive, root, force=False) == 0
    installed = root / "pi06-test"
    assert (installed / "deployment_manifest.json").is_file()
    assert (installed / "engines" / "prefix.engine").read_bytes() == PREFIX_BYTES
    assert run_verify(installed) == 0


def test_install_rejects_bad_archive(tmp_path):
    archive = tmp_path / "empty.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="exactly one package"):
        run_install(archive, root, force=False)


def test_cli_help_is_parseable():
    for command in ("eval", "infer", "latency", "package", "graph", "calibrate", "build", "verify", "server", "client"):
        result = _cli(command, "--help")
        assert result.returncode == 0, (command, result.stderr)
        assert "usage: picpp" in result.stdout, command


def test_package_actions_emit_json_and_exit_codes(tmp_path):
    package, _ = make_package(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    result = _cli("package", "list", "--root", str(root))
    assert result.returncode == 0
    assert json.loads(result.stdout)["packages"] == []
    result = _cli("package", "install", "--package", str(package), "--root", str(root))
    assert result.returncode == 0
    assert json.loads(result.stdout)["installed"] is True
    result = _cli("package", "verify", "--package", str(root / "pi06-test"))
    assert result.returncode == 0
    assert json.loads(result.stdout)["ok"] is True
    broken = root / "pi06-test" / "engines" / "prefix.engine"
    broken.write_bytes(b"tampered")
    result = _cli("package", "verify", "--package", str(root / "pi06-test"))
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False


def test_cli_verify_usage_error_exits_nonzero():
    result = _cli("verify", "parity")
    assert result.returncode != 0
    assert "required" in (result.stderr + result.stdout)


# The minimal package fixtures (mirrors test_package.py).
PREFIX_BYTES = b"prefix-engine-bytes"
SUFFIX_BYTES = b"suffix-engine-bytes"


def make_package(tmp_path: Path) -> tuple[Path, Path]:
    import hashlib

    package = tmp_path / "pkg"
    (package / "engines").mkdir(parents=True)
    (package / "engines" / "prefix.engine").write_bytes(PREFIX_BYTES)
    (package / "engines" / "suffix.engine").write_bytes(SUFFIX_BYTES)
    (package / "adapter.py").write_text("# adapter stub\n")
    (package / "export_manifest.json").write_text(json.dumps({"model_family": "pi06_heterogeneous_3view"}))
    manifest = {
        "schema": "picpp.deployment-manifest.v1",
        "model": {"family": "pi06", "variant": "test", "display_name": "pi06-test", "runtime": "tensorrt"},
        "export_manifest": {
            "path": "export_manifest.json",
            "sha256": hashlib.sha256((package / "export_manifest.json").read_bytes()).hexdigest(),
        },
        "engines": {
            "prefix_embed": {
                "path": "engines/prefix.engine",
                "size_bytes": len(PREFIX_BYTES),
                "sha256": hashlib.sha256(PREFIX_BYTES).hexdigest(),
                "recipe": {"onnx": "prefix.onnx"},
            },
            "suffix_step": {
                "path": "engines/suffix.engine",
                "size_bytes": len(SUFFIX_BYTES),
                "sha256": hashlib.sha256(SUFFIX_BYTES).hexdigest(),
                "recipe": {"onnx": "suffix.onnx"},
            },
        },
        "loop": {"kind": "host_denoise", "step_stage": "suffix_step", "steps": 10, "graph_capture": "auto"},
        "runtime": {"adapter": "adapter.py"},
    }
    manifest_path = package / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return package, manifest_path
