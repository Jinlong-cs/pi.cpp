#!/usr/bin/env python3
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ort_cuda_parity import (
    FIXTURE,
    FLOW_SCALE,
    FLOW_STEPS,
    PROFILE_DIR,
    ROOT,
    StrictCudaStage,
    apply_threshold,
    metric,
    sha256_bytes,
    tensor_to_numpy,
)


RESULT = ROOT / "results" / "ort_cuda_postfix_teacher_d10_parity.json"


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    stage = StrictCudaStage("postfix_step")

    teacher_inputs = {
        name: tensor_to_numpy(value) for name, value in fixture["postfix_inputs"].items()
    }
    kv_names = sorted(name for name in teacher_inputs if name.startswith("prefix_kv."))
    kv_before = {name: sha256_bytes(teacher_inputs[name]) for name in kv_names}
    teacher_outputs, teacher_seconds = stage.run(teacher_inputs)
    kv_after = {name: sha256_bytes(teacher_inputs[name]) for name in kv_names}
    teacher_metric = apply_threshold(
        metric(tensor_to_numpy(fixture["postfix_output"]), teacher_outputs["x_next"]),
        "final_normalized_action",
    )

    times = np.linspace(0, 1, FLOW_STEPS + 1, dtype=np.float32) * FLOW_SCALE
    x_i = tensor_to_numpy(fixture["prefill_outputs"]["x1"])
    controls = {
        name: tensor_to_numpy(fixture["prefix_outputs"][name])
        for name in (
            "dof_mask",
            "v_padding",
            "postfix_position_ids",
            "postfix_attention_mask",
        )
    }
    kv = {name: tensor_to_numpy(fixture["prefill_outputs"][name]) for name in kv_names}
    recursive_kv_before = {name: sha256_bytes(kv[name]) for name in kv_names}
    steps = []
    for index in range(1, FLOW_STEPS):
        feeds = {
            "x_i": x_i,
            "t_i": np.ascontiguousarray(times[index:index + 1]),
            "dt_i": np.ascontiguousarray(times[index + 1:index + 2] - times[index:index + 1]),
            **controls,
            **kv,
        }
        output, seconds = stage.run(feeds)
        x_i = output["x_next"]
        steps.append(
            {
                "step": index + 1,
                "t": float(times[index + 1]),
                "execution_seconds": seconds,
                "finite": bool(np.isfinite(x_i).all()),
                "pytorch_reference_available": index == FLOW_STEPS - 1,
            }
        )
    recursive_kv_after = {name: sha256_bytes(kv[name]) for name in kv_names}
    final_metric = apply_threshold(
        metric(tensor_to_numpy(fixture["postfix_output"]), x_i),
        "final_normalized_action",
    )
    steps[-1]["metrics"] = final_metric
    provider = stage.close()
    passed = bool(
        provider["strict_cuda_only"]
        and len(kv_names) == 72
        and kv_before == kv_after
        and recursive_kv_before == recursive_kv_after
        and teacher_metric["passed"]
        and final_metric["passed"]
        and all(item["finite"] for item in steps)
    )
    report = {
        "schema_version": "walloss.revision5.ort-cuda-postfix-parity.v1",
        "generated_at_epoch": time.time(),
        "scope": "strict CUDA postfix_step only; prefix_embed and prefill remain provider-blocked",
        "fixture_coverage": {
            "fixture_count": 1,
            "held_out_fixture": False,
            "x1_and_kv_source": "real PyTorch export fixture",
            "x2_to_x9_pytorch_reference_available": False,
            "x10_pytorch_reference_available": True,
        },
        "provider": provider,
        "kv_tensor_count": len(kv_names),
        "teacher_forced_final_step": {
            "execution_seconds": teacher_seconds,
            "metrics": teacher_metric,
            "kv_hash_unchanged": kv_before == kv_after,
        },
        "recursive_postfix_x2_to_x10": {
            "steps": steps,
            "final_x10_metrics": final_metric,
            "kv_hash_unchanged": recursive_kv_before == recursive_kv_after,
        },
        "passed": passed,
        "claim_limit": (
            "postfix mechanical parity only; not full three-stage parity, held-out export parity, "
            "AGX TensorRT parity, latency, server/client, or closed-loop evidence"
        ),
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "provider": provider, "teacher": teacher_metric, "x10": final_metric}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
