#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort

import ort_cuda_parity as base


ROOT = Path("/work")
APPROVED_MANIFEST = ROOT / "results" / "placement_manifest.json"
APPROVED_ALLOWLIST = ROOT / "results" / "cpu_allowlist.json"
RESULT = ROOT / "results" / "mixed_ep_stage_kv_d10_parity.json"
PROFILE_DIR = ROOT / "results" / "mixed_profiles"

APPROVED_MANIFEST_SHA256 = (
    "0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2"
)
APPROVED_ALLOWLIST_SHA256 = (
    "474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341"
)
APPROVED_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DISABLED_OPTIMIZERS = {"SimplifiedLayerNormFusion"}
MEMCPY_OPS = {"MemcpyFromHost", "MemcpyToHost"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def approved_cpu_rows() -> dict[str, set[tuple[str, str]]]:
    observed = {
        "placement_manifest": file_sha256(APPROVED_MANIFEST),
        "cpu_allowlist": file_sha256(APPROVED_ALLOWLIST),
    }
    expected = {
        "placement_manifest": APPROVED_MANIFEST_SHA256,
        "cpu_allowlist": APPROVED_ALLOWLIST_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"approved placement hash drift: {observed} != {expected}")
    payload = json.loads(APPROVED_ALLOWLIST.read_text())
    if not (
        payload.get("allowlist_pass")
        and payload.get("count_match")
        and payload.get("cpu_control_only")
    ):
        raise RuntimeError("approved CPU allowlist is not marked passing")
    return {
        stage: {(row["name"], row["op_type"]) for row in value["cpu_rows"]}
        for stage, value in payload["stages"].items()
    }


EXPECTED_CPU_ROWS = approved_cpu_rows()


class ApprovedMixedStage:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.model_path = base.PACKAGE / f"{stage}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(PROFILE_DIR / f"walloss_mixed_{stage}")
        started = time.monotonic()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=APPROVED_PROVIDERS,
            disabled_optimizers=DISABLED_OPTIMIZERS,
        )
        self.session_create_seconds = time.monotonic() - started
        self.input_meta = {item.name: item for item in self.session.get_inputs()}
        self.output_meta = {item.name: item for item in self.session.get_outputs()}

    run = base.StrictCudaStage.run

    def close(self) -> dict[str, Any]:
        session_providers = self.session.get_providers()
        profile_path = self.session.end_profiling()
        events = json.loads(Path(profile_path).read_text())
        provider_rows: set[tuple[str, str, str]] = set()
        memcpy_rows: set[tuple[str, str]] = set()
        for event in events:
            args = event.get("args", {})
            provider = args.get("provider")
            op_type = args.get("op_name")
            if not provider or not op_type:
                continue
            name = event.get("name", "")
            if name.endswith("_kernel_time"):
                name = name[: -len("_kernel_time")]
            if op_type in MEMCPY_OPS:
                memcpy_rows.add((name, op_type))
            else:
                provider_rows.add((name, op_type, provider))

        cpu_rows = {(name, op_type) for name, op_type, provider in provider_rows if provider == "CPUExecutionProvider"}
        non_cpu_rows = {
            (name, op_type, provider)
            for name, op_type, provider in provider_rows
            if provider not in APPROVED_PROVIDERS
        }
        expected_cpu = EXPECTED_CPU_ROWS[self.stage]
        cpu_exact = cpu_rows == expected_cpu
        providers_exact = session_providers == APPROVED_PROVIDERS and not non_cpu_rows
        cuda_present = any(provider == "CUDAExecutionProvider" for _, _, provider in provider_rows)
        placement_exact = bool(cpu_exact and providers_exact and cuda_present)
        result = {
            "session_providers": session_providers,
            "session_create_seconds": self.session_create_seconds,
            "profile": profile_path,
            "node_providers": sorted({provider for _, _, provider in provider_rows}),
            "approved_manifest_sha256": APPROVED_MANIFEST_SHA256,
            "approved_allowlist_sha256": APPROVED_ALLOWLIST_SHA256,
            "expected_cpu_node_count": len(expected_cpu),
            "profiled_cpu_node_count": len(cpu_rows),
            "cpu_allowlist_exact": cpu_exact,
            "missing_cpu_rows": sorted(expected_cpu - cpu_rows),
            "unexpected_cpu_rows": sorted(cpu_rows - expected_cpu),
            "unexpected_provider_rows": sorted(non_cpu_rows),
            "profiled_memcpy_rows": sorted(memcpy_rows),
            "approved_mixed_placement_passed": placement_exact,
            # Revision 5 main uses this field as its provider gate. Here it
            # means exact approved mixed placement, not CUDA-only execution.
            "strict_cuda_only": placement_exact,
        }
        del self.session
        gc.collect()
        return result


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    base.RESULT = RESULT
    base.PROFILE_DIR = PROFILE_DIR
    base.StrictCudaStage = ApprovedMixedStage
    exit_code = 0
    try:
        base.main()
    except SystemExit as error:
        exit_code = int(error.code or 0)

    report = json.loads(RESULT.read_text())
    placement_pass = all(
        stage.get("approved_mixed_placement_passed", False)
        for stage in report["stages"].values()
    )
    mechanical_pass = bool(
        report["decision"]["mechanical_single_fixture_parity_passed"]
        and placement_pass
    )
    report["schema_version"] = "walloss.revision6.mixed-ep-parity.v1"
    report["provider_contract"] = {
        "provider_order": APPROVED_PROVIDERS,
        "disabled_optimizers": sorted(DISABLED_OPTIMIZERS),
        "placement_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "cpu_allowlist_sha256": APPROVED_ALLOWLIST_SHA256,
        "exact_profile_placement_passed": placement_pass,
    }
    report["decision"] = {
        "single_fixture_mixed_ep_mechanical_parity_passed": mechanical_pass,
        "5090_mixed_ep_diagnostic_gate_passed": mechanical_pass,
        "export_parity_gate_passed": False,
        "status": "passed_claim_bounded" if mechanical_pass else "failed",
        "reason": (
            "The approved mixed placement and single real fixture passed all registered stage, KV, teacher-forced and recursive-D10 checks."
            if mechanical_pass
            else "Provider placement or one or more registered numerical checks failed."
        ),
    }
    report["claim_limit"] = (
        "single-fixture mixed-EP mechanical parity only; no held-out fixture or PyTorch x2..x9 references; "
        "not export-parity, AGX TensorRT buildability, latency, server/client, closed-loop, or deployment evidence"
    )
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    if not mechanical_pass:
        raise SystemExit(2 if exit_code == 0 else exit_code)


if __name__ == "__main__":
    main()
