from __future__ import annotations

import json

import pytest
from pi_cpp.authorization import authorization_denied
from pi_cpp.package.manifest import validate

VALID = {
    "schema": "picpp.deployment-manifest.v1",
    "model": {"family": "pi06", "variant": "rtc", "display_name": "pi06 rtc", "runtime": "tensorrt"},
    "export_manifest": {"path": "export_manifest.json", "sha256": "a" * 64},
    "engines": {
        "suffix_step": {
            "path": "engines/suffix_step.engine",
            "size_bytes": 1000,
            "sha256": "b" * 64,
            "recipe": {"onnx": "suffix.onnx"},
        }
    },
    "loop": {"kind": "host_denoise", "step_stage": "suffix_step", "steps": 10, "graph_capture": "auto"},
    "runtime": {"adapter": "adapter.py"},
}


def _manifest_with(authorization: dict | None) -> dict:
    manifest = json.loads(json.dumps(VALID))
    if authorization is not None:
        manifest["authorization"] = authorization
    return manifest


def test_authorization_denied_without_flag(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(json.dumps(_manifest_with({"required": True})))
    assert authorization_denied(model_dir, authorized=False)
    assert not authorization_denied(model_dir, authorized=True)


def test_authorization_absent_never_denied(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(json.dumps(_manifest_with({"required": False})))
    assert not authorization_denied(model_dir, authorized=False)
    model_dir = tmp_path / "nomanifest"
    model_dir.mkdir()
    assert not authorization_denied(model_dir, authorized=False)


def test_validate_accepts_authorization():
    validate(_manifest_with({"required": True}))
    validate(_manifest_with({"required": False}))


def test_validate_rejects_bad_authorization():
    with pytest.raises(ValueError, match="authorization.required"):
        validate(_manifest_with({"required": "yes"}))
