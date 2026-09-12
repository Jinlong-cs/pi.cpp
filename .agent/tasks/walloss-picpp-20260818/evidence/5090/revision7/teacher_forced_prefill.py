#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto


THRESHOLDS = {
    "embedding_kv": {"max_abs": 0.10, "rms": 0.02, "relative_l2": 0.02, "cosine": 0.999},
    "trajectory": {"max_abs": 0.15, "rms": 0.03, "relative_l2": 0.03, "cosine": 0.998},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_numpy(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return value.numpy()


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


def metric(reference: np.ndarray, candidate: np.ndarray, threshold_name: str) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1).astype(np.float64)
    got = np.asarray(candidate, dtype=np.float32).reshape(-1).astype(np.float64)
    if ref.shape != got.shape:
        return {"shape_match": False, "passed": False}
    finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
    if not finite:
        return {"shape_match": True, "finite": False, "passed": False}
    delta = got - ref
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    row = {
        "shape_match": True,
        "finite": True,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "relative_l2": None if ref_norm == 0.0 else float(np.linalg.norm(delta) / ref_norm),
        "cosine": None if ref_norm == 0.0 or got_norm == 0.0 else float(np.dot(ref, got) / (ref_norm * got_norm)),
        "threshold": threshold_name,
        "limits": THRESHOLDS[threshold_name],
    }
    limits = THRESHOLDS[threshold_name]
    row["passed"] = bool(
        row["relative_l2"] is not None
        and row["cosine"] is not None
        and row["max_abs"] <= limits["max_abs"]
        and row["rms"] <= limits["rms"]
        and row["relative_l2"] <= limits["relative_l2"]
        and row["cosine"] >= limits["cosine"]
    )
    return row


def make_session(model: Path, profile_prefix: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(
        str(model),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        disabled_optimizers={"SimplifiedLayerNormFusion"},
    )


def bind_and_run(session: ort.InferenceSession, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], float]:
    input_meta = {item.name: item for item in session.get_inputs()}
    output_meta = {item.name: item for item in session.get_outputs()}
    if set(inputs) != set(input_meta):
        raise ValueError(f"prefill inputs mismatch missing={sorted(set(input_meta)-set(inputs))} extra={sorted(set(inputs)-set(input_meta))}")
    binding = session.io_binding()
    keepalive: list[Any] = []
    for name, meta in input_meta.items():
        value = np.ascontiguousarray(tensor_numpy(inputs[name]))
        if value.shape != fixed_shape(meta) or value.dtype != numpy_dtype(meta.type):
            raise TypeError(f"{name}: expected {fixed_shape(meta)}/{numpy_dtype(meta.type)}, got {value.shape}/{value.dtype}")
        if meta.type == "tensor(bfloat16)":
            raw = value.view(np.uint16)
            ort_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(raw, TensorProto.BFLOAT16)
            keepalive.extend((value, raw, ort_value))
            binding.bind_ortvalue_input(name, ort_value)
        else:
            keepalive.append(value)
            binding.bind_cpu_input(name, value)

    outputs: dict[str, np.ndarray] = {}
    for name, meta in output_meta.items():
        shape = fixed_shape(meta)
        dtype = numpy_dtype(meta.type)
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
    session.run_with_iobinding(binding)
    return outputs, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if "prefill_inputs" not in fixture or "prefill_outputs" not in fixture:
        raise KeyError("fixture must contain prefill_inputs and prefill_outputs")
    session = make_session(args.model, args.profile_prefix)
    outputs, execution_seconds = bind_and_run(session, fixture["prefill_inputs"])
    profile = session.end_profiling()
    reference = fixture["prefill_outputs"]
    if set(reference) != set(outputs):
        raise ValueError("prefill output names do not match fixture")
    metrics: dict[str, Any] = {}
    for name in sorted(reference):
        threshold = "trajectory" if name == "x1" else "embedding_kv"
        metrics[name] = metric(tensor_numpy(reference[name]), outputs[name], threshold)
        if name.startswith("prefix_kv."):
            metrics[name]["nonzero"] = int(np.count_nonzero(np.asarray(outputs[name], dtype=np.float32)))
            metrics[name]["passed"] = bool(metrics[name]["passed"] and metrics[name]["nonzero"] > 0)
    kv_names = [name for name in metrics if name.startswith("prefix_kv.")]
    report = {
        "schema_version": "walloss.revision7.teacher-forced-prefill.v1",
        "host": platform.node(),
        "ort_version": ort.__version__,
        "model": {"path": str(args.model), "sha256": sha256_file(args.model)},
        "fixture": {"path": str(args.fixture), "sha256": sha256_file(args.fixture)},
        "input_source": "saved_pytorch_prefill_inputs",
        "consumes_ort_prefix_output": False,
        "execution_seconds": execution_seconds,
        "profile": profile,
        "providers": session.get_providers(),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "kv_tensor_count": len(kv_names),
        "kv_layer_count": len({name.split(".")[1] for name in kv_names}),
        "passed": bool(len(kv_names) == 72 and all(row["passed"] for row in metrics.values())),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "kv_tensor_count": len(kv_names)}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
