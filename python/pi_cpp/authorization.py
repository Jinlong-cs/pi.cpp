"""The arm-authorization boundary.

A package whose deployment manifest declares ``authorization.required``
must be started with the explicit ``--authorize`` flag; without it the
closed loop refuses to serve and the command exits 2 (unauthorized).
Packages without the section have no requirement (the historical
deployments stay untouched).
"""

from __future__ import annotations

import json
from pathlib import Path

DEPLOYMENT_MANIFEST_NAME = "deployment_manifest.json"


def authorization_denied(model_dir: str | Path, authorized: bool) -> bool:
    deployment = Path(model_dir) / DEPLOYMENT_MANIFEST_NAME
    if not deployment.is_file():
        return False
    manifest = json.loads(deployment.read_text(encoding="utf-8"))
    required = bool(manifest.get("authorization", {}).get("required", False))
    return required and not authorized


__all__ = ["authorization_denied"]
