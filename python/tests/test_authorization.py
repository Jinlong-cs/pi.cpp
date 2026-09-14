from __future__ import annotations

import json

from pi_cpp.authorization import authorization_denied

AUTHORIZATION_REQUIRED = {"authorization": {"required": True}}


def test_authorization_denied_without_flag(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(json.dumps(AUTHORIZATION_REQUIRED))
    assert authorization_denied(model_dir, authorized=False)
    assert not authorization_denied(model_dir, authorized=True)


def test_authorization_absent_never_denied(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(json.dumps({"authorization": {"required": False}}))
    assert not authorization_denied(model_dir, authorized=False)
    model_dir = tmp_path / "nomanifest"
    model_dir.mkdir()
    assert not authorization_denied(model_dir, authorized=False)
