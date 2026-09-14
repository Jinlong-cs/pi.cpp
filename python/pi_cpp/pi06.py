"""Unified PI0.6 family entry point: variant detection + one builder.

The pi06 family ships three ABI variants — airbot (LeRobot 14-dim
contract), heterogeneous (continuous-state 16-dim flow loop), and rtc
(heterogeneous + the training-time-RTC delay/action_prefix ABI). The
variant is declared by the manifests, never by the caller:

- the deployment manifest (``picpp.deployment-manifest.v1``) wins when
  present (``model.variant`` / ``runtime.rtc.enabled``),
- otherwise the export manifest: schema ``pi_cpp.pi06_airbot.v1`` ->
  airbot; schema ``pi_cpp.pi06_heterogeneous.v1`` + ``rtc.training_time_rtc``
  -> rtc; the same schema without the rtc section -> heterogeneous.

Fail-closed: an rtc section whose ``training_time_rtc`` is not True is
rejected, and a contradictory deployment manifest raises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pi_cpp.pi06_airbot import Pi06AirbotRunnerWrapper
from pi_cpp.pi06_heterogeneous import Pi06HeterogeneousRunnerWrapper
from pi_cpp.pi06_rtc import Pi06RtcRunnerWrapper

DEPLOYMENT_MANIFEST_NAME = "deployment_manifest.json"
EXPORT_MANIFEST_NAME = "export_manifest.json"

_VARIANTS = ("airbot", "heterogeneous", "rtc")


def detect_pi06_variant(manifest: dict[str, Any]) -> str:
    """Return the ABI variant a manifest declares."""
    schema = manifest.get("schema")
    if schema == "pi_cpp.pi06_airbot.v1":
        return "airbot"
    if schema == "pi_cpp.pi06_heterogeneous.v1":
        rtc = manifest.get("rtc")
        if not rtc:
            return "heterogeneous"
        if not rtc.get("training_time_rtc", False):
            raise ValueError("manifest declares an rtc section but rtc.training_time_rtc != True")
        return "rtc"
    raise ValueError(f"unrecognized pi06 manifest schema: {schema!r}")


def _variant_from_deployment_manifest(manifest: dict[str, Any]) -> str:
    variant = manifest.get("model", {}).get("variant")
    rtc_enabled = manifest.get("runtime", {}).get("rtc", {}).get("enabled", False)
    if variant is not None and variant not in _VARIANTS:
        raise ValueError(f"deployment manifest model.variant {variant!r} not in {_VARIANTS}")
    if rtc_enabled and variant not in (None, "rtc"):
        raise ValueError("deployment manifest runtime.rtc.enabled contradicts model.variant")
    if rtc_enabled or variant == "rtc":
        return "rtc"
    if variant is not None:
        return variant
    raise ValueError("deployment manifest declares neither model.variant nor runtime.rtc.enabled")


def detect_variant(model_dir: Path) -> str:
    """Detect the ABI variant from the manifests in model_dir."""
    deployment = model_dir / DEPLOYMENT_MANIFEST_NAME
    if deployment.is_file():
        manifest = json.loads(deployment.read_text(encoding="utf-8"))
        return _variant_from_deployment_manifest(manifest)
    export = model_dir / EXPORT_MANIFEST_NAME
    if not export.is_file():
        raise ValueError(f"neither {DEPLOYMENT_MANIFEST_NAME} nor {EXPORT_MANIFEST_NAME} found in {model_dir}")
    return detect_pi06_variant(json.loads(export.read_text(encoding="utf-8")))


def build_pi06_runner(model_dir: str | Path, *, variant: str | None = None):
    """Build the wrapper for the variant the manifests declare.

    ``variant`` overrides detection when set; the wrapper still fails
    closed if the manifests contradict it (e.g. the rtc wrapper refuses a
    manifest without rtc.training_time_rtc=True).
    """
    resolved = variant or detect_variant(Path(model_dir))
    if resolved == "airbot":
        return Pi06AirbotRunnerWrapper(model_dir)
    if resolved == "heterogeneous":
        return Pi06HeterogeneousRunnerWrapper(model_dir)
    if resolved == "rtc":
        return Pi06RtcRunnerWrapper(model_dir)
    raise ValueError(f"unknown pi06 variant: {resolved!r}")


__all__ = [
    "build_pi06_runner",
    "detect_pi06_variant",
    "detect_variant",
]
