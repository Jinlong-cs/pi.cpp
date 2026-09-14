#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort
import torch


ROOT = Path("/work")
PACKAGE = ROOT / "inbound_v18" / "onnx_real_legacy_v18"
FIXTURE = ROOT / "inbound_v18" / "pytorch" / "stage_tensors_explicitcache.pt"
RESULT = ROOT / "results" / "ort_cuda_stage_kv_d10_parity.json"
PROFILE_DIR = ROOT / "results" / "profiles"
LOGICAL_STAGES = ("prefix_embed", "prefill", "postfix_step")
FLOW_STEPS = 10
FLOW_SCALE = np.float32(0.999)

THRESHOLDS = {
    "embedding_kv": {"max_abs": 0.10, "rms": 0.02, "relative_l2": 0.02, "cosine": 0.999},
    "trajectory": {"max_abs": 0.15, "rms": 0.03, "relative_l2": 0.03, "cosine": 0.998},
    "final_normalized_action": {"max_abs": 0.20, "rms": 0.05, "relative_l2": 0.05, "cosine": 0.995},
}


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return value.numpy()


def as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = as_float32(reference).reshape(-1).astype(np.float64)
    got = as_float32(candidate).reshape(-1).astype(np.float64)
    if ref.shape != got.shape:
        return {"shape_match": False, "passed": False}
    finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
    if not finite:
        return {"shape_match": True, "finite": False, "passed": False}
    delta = got - ref
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    relative_l2 = None if ref_norm == 0.0 else float(np.linalg.norm(delta) / ref_norm)
    cosine = None if ref_norm == 0.0 or got_norm == 0.0 else float(np.dot(ref, got) / (ref_norm * got_norm))
    return {
        "shape_match": True,
        "finite": finite,
        "max_abs": max_abs,
        "rms": rms,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def apply_threshold(metrics: dict[str, Any], threshold_name: str) -> dict[str, Any]:
    limits = THRESHOLDS[threshold_name]
    values_defined = metrics.get("relative_l2") is not None and metrics.get("cosine") is not None
    passed = bool(
        metrics.get("shape_match")
        and metrics.get("finite")
        and values_defined
        and metrics["max_abs"] <= limits["max_abs"]
        and metrics["rms"] <= limits["rms"]
        and metrics["relative_l2"] <= limits["relative_l2"]
        and metrics["cosine"] >= limits["cosine"]
    )
    return {**metrics, "threshold": threshold_name, "limits": limits, "passed": passed}


def exact_result(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    same_shape = reference.shape == candidate.shape
    same_dtype = reference.dtype == candidate.dtype
    exact = bool(same_shape and same_dtype and np.array_equal(reference, candidate))
    return {
        "shape": list(candidate.shape),
        "dtype": str(candidate.dtype),
        "shape_match": same_shape,
        "dtype_match": same_dtype,
        "exact": exact,
        "passed": exact,
    }


def numpy_dtype(type_name: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(bool)": np.dtype(np.bool_),
        "tensor(bfloat16)": np.dtype(ml_dtypes.bfloat16),
    }
    if type_name not in mapping:
        raise TypeError(f"unsupported ORT tensor type: {type_name}")
    return mapping[type_name]


def fixed_shape(meta: ort.NodeArg) -> tuple[int, ...]:
    if any(not isinstance(dim, int) or dim <= 0 for dim in meta.shape):
        raise ValueError(f"non-fixed shape for {meta.name}: {meta.shape}")
    return tuple(int(dim) for dim in meta.shape)


class StrictCudaStage:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.model_path = PACKAGE / f"{stage}.onnx"
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(PROFILE_DIR / f"walloss_{stage}")
        started = time.monotonic()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider"],
            disabled_optimizers={"SimplifiedLayerNormFusion"},
        )
        self.session_create_seconds = time.monotonic() - started
        self.input_meta = {item.name: item for item in self.session.get_inputs()}
        self.output_meta = {item.name: item for item in self.session.get_outputs()}
        self.profile_path: str | None = None
        self.node_providers: list[str] = []

    def run(self, feeds: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        if set(feeds) != set(self.input_meta):
            raise ValueError(
                f"{self.stage} input mismatch missing={sorted(set(self.input_meta)-set(feeds))} "
                f"extra={sorted(set(feeds)-set(self.input_meta))}"
            )
        binding = self.session.io_binding()
        keepalive: list[Any] = []
        for name, meta in self.input_meta.items():
            value = np.ascontiguousarray(feeds[name])
            expected_dtype = numpy_dtype(meta.type)
            if value.shape != fixed_shape(meta) or value.dtype != expected_dtype:
                raise TypeError(
                    f"{self.stage}:{name} expected {fixed_shape(meta)}/{expected_dtype}, "
                    f"got {value.shape}/{value.dtype}"
                )
            if meta.type == "tensor(bfloat16)":
                raw = value.view(np.uint16)
                ort_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(raw, TensorProto.BFLOAT16)
                keepalive.extend((value, raw, ort_value))
                binding.bind_ortvalue_input(name, ort_value)
            else:
                binding.bind_cpu_input(name, value)
                keepalive.append(value)

        outputs: dict[str, np.ndarray] = {}
        for name, meta in self.output_meta.items():
            dtype = numpy_dtype(meta.type)
            shape = fixed_shape(meta)
            if meta.type == "tensor(bfloat16)":
                raw = np.empty(shape, dtype=np.uint16)
                value = raw.view(ml_dtypes.bfloat16)
                ort_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(raw, TensorProto.BFLOAT16)
                keepalive.extend((raw, value, ort_value))
            else:
                value = np.empty(shape, dtype=dtype)
                ort_value = ort.OrtValue.ortvalue_from_numpy(value)
                keepalive.extend((value, ort_value))
            outputs[name] = value
            binding.bind_ortvalue_output(name, ort_value)

        started = time.monotonic()
        self.session.run_with_iobinding(binding)
        seconds = time.monotonic() - started
        return outputs, seconds

    def close(self) -> dict[str, Any]:
        providers = self.session.get_providers()
        profile_path = self.session.end_profiling()
        self.profile_path = profile_path
        events = json.loads(Path(profile_path).read_text())
        self.node_providers = sorted(
            {event.get("args", {}).get("provider") for event in events if event.get("args", {}).get("provider")}
        )
        result = {
            "session_providers": providers,
            "session_create_seconds": self.session_create_seconds,
            "profile": profile_path,
            "node_providers": self.node_providers,
            "strict_cuda_only": self.node_providers == ["CUDAExecutionProvider"],
        }
        del self.session
        gc.collect()
        return result


def check_models() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for stage in LOGICAL_STAGES:
        path = PACKAGE / f"{stage}.onnx"
        started = time.monotonic()
        onnx.checker.check_model(str(path), full_check=False)
        checks[stage] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "seconds": time.monotonic() - started,
            "passed": True,
        }
    return checks


def stage_compare(reference: dict[str, torch.Tensor], candidate: dict[str, np.ndarray], floating: set[str], threshold: str) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError("stage output names do not match fixture")
    results: dict[str, Any] = {}
    for name in sorted(reference):
        ref = tensor_to_numpy(reference[name])
        got = candidate[name]
        results[name] = apply_threshold(metric(ref, got), threshold) if name in floating else exact_result(ref, got)
    return results


def all_pass(rows: dict[str, dict[str, Any]]) -> bool:
    return all(bool(row.get("passed")) for row in rows.values())


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    report: dict[str, Any] = {
        "schema_version": "walloss.revision5.ort-cuda-parity.v1",
        "generated_at_epoch": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "ort_version": ort.__version__,
        "onnx_version": onnx.__version__,
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "available_providers": ort.get_available_providers(),
        "thresholds": THRESHOLDS,
        "fixture_coverage": {
            "fixture_count": 1,
            "real_export_fixture": True,
            "held_out_fixture": False,
            "per_step_pytorch_x2_to_x9": False,
            "teacher_forced_final_step": True,
            "recursive_final_x10": True,
            "claim_limit": "mechanical export parity only; held-out export-parity remains pending",
        },
        "onnx_checker": check_models(),
        "stages": {},
    }

    prefix = StrictCudaStage("prefix_embed")
    prefix_inputs = {name: tensor_to_numpy(value) for name, value in fixture["prefix_inputs"].items()}
    prefix_outputs, prefix_seconds = prefix.run(prefix_inputs)
    prefix_metrics = stage_compare(
        fixture["prefix_outputs"], prefix_outputs, {"prefill_inputs_embeds"}, "embedding_kv"
    )
    report["stages"]["prefix_embed"] = {
        "execution_seconds": prefix_seconds,
        "metrics": prefix_metrics,
        "passed": all_pass(prefix_metrics),
        **prefix.close(),
    }

    prefill = StrictCudaStage("prefill")
    prefill_inputs = {name: prefix_outputs[name] for name in prefill.input_meta}
    prefill_outputs, prefill_seconds = prefill.run(prefill_inputs)
    prefill_metrics: dict[str, Any] = {}
    for name, reference_tensor in fixture["prefill_outputs"].items():
        ref = tensor_to_numpy(reference_tensor)
        threshold = "trajectory" if name == "x1" else "embedding_kv"
        prefill_metrics[name] = apply_threshold(metric(ref, prefill_outputs[name]), threshold)
        if name.startswith("prefix_kv."):
            prefill_metrics[name]["nonzero"] = int(np.count_nonzero(as_float32(prefill_outputs[name])))
            prefill_metrics[name]["passed"] = bool(prefill_metrics[name]["passed"] and prefill_metrics[name]["nonzero"] > 0)
    kv_names = sorted(name for name in prefill_outputs if name.startswith("prefix_kv."))
    report["stages"]["prefill"] = {
        "execution_seconds": prefill_seconds,
        "metrics": prefill_metrics,
        "kv_tensor_count": len(kv_names),
        "kv_layer_count": len({name.split(".")[1] for name in kv_names}),
        "kv_all_nonzero": all(np.count_nonzero(as_float32(prefill_outputs[name])) > 0 for name in kv_names),
        "passed": all_pass(prefill_metrics) and len(kv_names) == 72,
        **prefill.close(),
    }

    postfix = StrictCudaStage("postfix_step")
    teacher_inputs = {name: tensor_to_numpy(value) for name, value in fixture["postfix_inputs"].items()}
    kv_hash_before = {name: sha256_bytes(teacher_inputs[name]) for name in kv_names}
    teacher_outputs, teacher_seconds = postfix.run(teacher_inputs)
    kv_hash_after = {name: sha256_bytes(teacher_inputs[name]) for name in kv_names}
    teacher_metric = apply_threshold(
        metric(tensor_to_numpy(fixture["postfix_output"]), teacher_outputs["x_next"]),
        "final_normalized_action",
    )

    times = np.linspace(0, 1, FLOW_STEPS + 1, dtype=np.float32) * FLOW_SCALE
    x_i = prefill_outputs["x1"]
    recursive_steps: list[dict[str, Any]] = [
        {"step": 1, "t": float(times[1]), "reference_available": True,
         "metrics": apply_threshold(metric(tensor_to_numpy(fixture["prefill_outputs"]["x1"]), x_i), "trajectory")}
    ]
    recursive_seconds = 0.0
    for index in range(1, FLOW_STEPS):
        feeds = {
            "x_i": x_i,
            "t_i": np.ascontiguousarray(times[index:index + 1]),
            "dt_i": np.ascontiguousarray(times[index + 1:index + 2] - times[index:index + 1]),
            "dof_mask": prefix_outputs["dof_mask"],
            "v_padding": prefix_outputs["v_padding"],
            "postfix_position_ids": prefix_outputs["postfix_position_ids"],
            "postfix_attention_mask": prefix_outputs["postfix_attention_mask"],
            **{name: prefill_outputs[name] for name in kv_names},
        }
        step_outputs, seconds = postfix.run(feeds)
        recursive_seconds += seconds
        x_i = step_outputs["x_next"]
        recursive_steps.append({
            "step": index + 1,
            "t": float(times[index + 1]),
            "execution_seconds": seconds,
            "finite": bool(np.isfinite(x_i).all()),
            "reference_available": index == FLOW_STEPS - 1,
        })
    final_metric = apply_threshold(
        metric(tensor_to_numpy(fixture["postfix_output"]), x_i), "final_normalized_action"
    )
    recursive_steps[-1]["metrics"] = final_metric
    postfix_provider = postfix.close()
    report["stages"]["postfix_step"] = {
        "teacher_forced": {
            "execution_seconds": teacher_seconds,
            "metrics": teacher_metric,
            "kv_input_hash_unchanged": kv_hash_before == kv_hash_after,
            "passed": teacher_metric["passed"] and kv_hash_before == kv_hash_after,
        },
        "recursive_d10": {
            "steps": recursive_steps,
            "postfix_execution_seconds": recursive_seconds,
            "final_normalized_action_metrics": final_metric,
            "intermediate_pytorch_reference_available": False,
            "passed": final_metric["passed"] and all(row.get("finite", True) for row in recursive_steps),
        },
        **postfix_provider,
    }

    strict_cuda = all(stage.get("strict_cuda_only", False) for stage in report["stages"].values())
    mechanical_pass = bool(
        strict_cuda
        and report["stages"]["prefix_embed"]["passed"]
        and report["stages"]["prefill"]["passed"]
        and report["stages"]["postfix_step"]["teacher_forced"]["passed"]
        and report["stages"]["postfix_step"]["recursive_d10"]["passed"]
    )
    report["decision"] = {
        "strict_cuda_provider_passed": strict_cuda,
        "mechanical_single_fixture_parity_passed": mechanical_pass,
        "5090_ort_cuda_parity_gate_passed": False,
        "export_parity_gate_passed": False,
        "status": "fixture_coverage_blocked" if mechanical_pass else "numerical_or_provider_failed",
        "reason": (
            "The single real export fixture passes mechanical parity, but no registered held-out fixture "
            "and no PyTorch x2..x9 references exist; blocking gates cannot pass."
            if mechanical_pass else
            "One or more provider, stage, KV, teacher-forced, or recursive-D10 checks failed."
        ),
    }
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    if not mechanical_pass:
        raise SystemExit(2)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
