#!/usr/bin/env python3
"""Revision 7 same-host PyTorch/ORT prefix-prefill diagnostic harness.

This script is intentionally task-local.  It recovers the complete unpatched
PyTorch reference from the legacy prefix inputs, compares that reference with
the export-patched PyTorch path and the frozen V18 ONNX stages, and emits the
PT/ORT 2x2 prefix-prefill matrix.  It is a correctness diagnostic, not a
latency benchmark or an AGX deployment command.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
from onnx import TensorProto

import agx_export_runner as export_runner


APPROVED_PLACEMENT_MANIFEST_SHA256 = "0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2"
APPROVED_CPU_ALLOWLIST_SHA256 = "474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DISABLED_OPTIMIZERS = {"SimplifiedLayerNormFusion"}
MEMCPY_OPS = {"MemcpyFromHost", "MemcpyToHost"}
STAGES = ("prefix_embed", "prefill")

THRESHOLDS = {
    "external_x0_same_pytorch": {"max_abs": 1e-6, "rms": 1e-7, "relative_l2": 1e-6, "cosine": 0.999999},
    "embedding_kv": {"max_abs": 0.10, "rms": 0.02, "relative_l2": 0.02, "cosine": 0.999},
    "trajectory": {"max_abs": 0.15, "rms": 0.03, "relative_l2": 0.03, "cosine": 0.998},
    "final_normalized_action": {"max_abs": 0.20, "rms": 0.05, "relative_l2": 0.05, "cosine": 0.995},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        row = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        rows.append(row)
        aggregate.update(f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode())
    if not rows:
        raise RuntimeError(f"artifact tree is empty: {root}")
    return {"root": str(root), "file_count": len(rows), "aggregate_sha256": aggregate.hexdigest(), "files": rows}


def sha256_tensor(value: torch.Tensor | np.ndarray) -> str:
    array = value_to_numpy(value)
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def dtype_name(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def value_to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
        return tensor.numpy()
    return np.ascontiguousarray(value)


def numpy_to_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(value)
    if array.dtype == np.dtype(ml_dtypes.bfloat16):
        raw = array.view(np.uint16).copy()
        return torch.from_numpy(raw).view(torch.bfloat16)
    return torch.from_numpy(array.copy())


def cpu_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    return numpy_to_tensor(value)


def cpu_dict(values: Mapping[str, torch.Tensor | np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: cpu_tensor(value) for name, value in values.items()}


def device_dict(values: Mapping[str, torch.Tensor | np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: cpu_tensor(value).to(device=device) for name, value in values.items()}


def tensor_summary(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    array = value_to_numpy(value)
    floating = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(array.shape),
        "dtype": dtype_name(value),
        "finite": bool(np.isfinite(floating).all()),
        "nonzero": int(np.count_nonzero(floating)),
        "max_abs": float(np.max(np.abs(floating))) if floating.size else 0.0,
        "sha256": hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest(),
    }


def compare_tensor(
    reference: torch.Tensor | np.ndarray,
    candidate: torch.Tensor | np.ndarray,
    *,
    threshold: str | None = None,
    exact_required: bool = False,
    nonzero_required: bool = False,
) -> dict[str, Any]:
    ref_raw = value_to_numpy(reference)
    got_raw = value_to_numpy(candidate)
    shape_match = ref_raw.shape == got_raw.shape
    dtype_match = dtype_name(reference) == dtype_name(candidate)
    row: dict[str, Any] = {
        "reference": tensor_summary(reference),
        "candidate": tensor_summary(candidate),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "exact": bool(shape_match and dtype_match and np.array_equal(ref_raw, got_raw)),
        "threshold": threshold,
        "limits": None if threshold is None else THRESHOLDS[threshold],
        "nonzero_required": nonzero_required,
    }
    if not shape_match:
        row.update({"finite": False, "max_abs": None, "rms": None, "relative_l2": None, "cosine": None, "passed": False})
        return row
    ref = np.asarray(ref_raw, dtype=np.float32).reshape(-1).astype(np.float64)
    got = np.asarray(got_raw, dtype=np.float32).reshape(-1).astype(np.float64)
    finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
    difference = got - ref
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    relative_l2 = None if ref_norm == 0.0 else float(np.linalg.norm(difference) / ref_norm)
    cosine = None if ref_norm == 0.0 or got_norm == 0.0 else float(np.dot(ref, got) / (ref_norm * got_norm))
    row.update(
        {
            "finite": finite,
            "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
            "rms": float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0,
            "relative_l2": relative_l2,
            "cosine": cosine,
        }
    )
    nonzero_pass = not nonzero_required or row["candidate"]["nonzero"] > 0
    if exact_required:
        passed = bool(row["exact"] and finite and nonzero_pass)
    else:
        if threshold is None:
            raise ValueError("non-exact comparison requires a registered threshold")
        limits = THRESHOLDS[threshold]
        passed = bool(
            shape_match
            and dtype_match
            and finite
            and relative_l2 is not None
            and cosine is not None
            and row["max_abs"] <= limits["max_abs"]
            and row["rms"] <= limits["rms"]
            and relative_l2 <= limits["relative_l2"]
            and cosine >= limits["cosine"]
            and nonzero_pass
        )
    row["passed"] = passed
    return row


def compare_prefix(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError(f"prefix output names differ: missing={sorted(set(reference) - set(candidate))}, extra={sorted(set(candidate) - set(reference))}")
    return {
        name: compare_tensor(
            reference[name],
            candidate[name],
            threshold="embedding_kv" if name == "prefill_inputs_embeds" else None,
            exact_required=name != "prefill_inputs_embeds",
        )
        for name in sorted(reference)
    }


def compare_prefill(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError(f"prefill output names differ: missing={sorted(set(reference) - set(candidate))}, extra={sorted(set(candidate) - set(reference))}")
    return {
        name: compare_tensor(
            reference[name],
            candidate[name],
            threshold="trajectory" if name == "x1" else "embedding_kv",
            nonzero_required=name.startswith("prefix_kv."),
        )
        for name in sorted(reference)
    }


def rows_pass(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(rows) and all(bool(row["passed"]) for row in rows.values())


def section_inventory(values: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor_summary(value) for name, value in sorted(values.items())}


def embedding_region_metrics(
    reference: torch.Tensor | np.ndarray,
    candidate: torch.Tensor | np.ndarray,
    input_ids: torch.Tensor,
    prefix_length: int,
    image_token_id: int,
) -> dict[str, Any]:
    ref = cpu_tensor(reference)
    got = cpu_tensor(candidate)
    if ref.ndim != 3 or got.shape != ref.shape or input_ids.shape != ref.shape[:2]:
        raise ValueError(
            f"embedding region ABI differs: ref={tuple(ref.shape)}, got={tuple(got.shape)}, input_ids={tuple(input_ids.shape)}"
        )
    positions = torch.arange(ref.shape[1]).unsqueeze(0)
    masks = {
        "full": torch.ones_like(input_ids, dtype=torch.bool),
        "prefix_all": positions < prefix_length,
        "action_suffix": positions >= prefix_length,
        "image_tokens": input_ids == image_token_id,
        "prefix_non_image": (positions < prefix_length) & (input_ids != image_token_id),
    }
    rows: dict[str, Any] = {}
    for name, mask in masks.items():
        token_count = int(torch.count_nonzero(mask))
        if token_count == 0:
            raise RuntimeError(f"embedding region {name} has no tokens")
        row = compare_tensor(ref[mask], got[mask], threshold="embedding_kv")
        row["token_count"] = token_count
        rows[name] = row
    return {"regions": rows, "passed": rows_pass(rows)}


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
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in meta.shape):
        raise ValueError(f"non-fixed shape for {meta.name}: {meta.shape}")
    return tuple(int(dimension) for dimension in meta.shape)


def load_approved_placement(placement_path: Path, allowlist_path: Path) -> tuple[dict[str, set[tuple[str, str]]], dict[str, Any]]:
    observed_hashes = {"placement_manifest": sha256_file(placement_path), "cpu_allowlist": sha256_file(allowlist_path)}
    expected_hashes = {
        "placement_manifest": APPROVED_PLACEMENT_MANIFEST_SHA256,
        "cpu_allowlist": APPROVED_CPU_ALLOWLIST_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise RuntimeError(f"approved placement hash drift: observed={observed_hashes}, expected={expected_hashes}")
    placement = json.loads(placement_path.read_text())
    allowlist = json.loads(allowlist_path.read_text())
    required_flags = ("allowlist_pass", "count_match", "cpu_control_only")
    if not all(bool(placement.get(name)) for name in required_flags) or not all(bool(allowlist.get(name)) for name in required_flags):
        raise RuntimeError("approved placement/allowlist is not marked passing")
    expected: dict[str, set[tuple[str, str]]] = {}
    for stage, payload in allowlist["stages"].items():
        allow_rows = {(row["name"], row["op_type"]) for row in payload["cpu_rows"]}
        manifest_rows = {
            (row["name"], row["op_type"])
            for row in placement["stages"][stage]["rows"]
            if row["provider"] == "CPUExecutionProvider"
        }
        if allow_rows != manifest_rows:
            raise RuntimeError(f"approved placement and CPU allowlist differ for {stage}")
        expected[stage] = allow_rows
    if not set(STAGES).issubset(expected):
        raise RuntimeError(f"approved CPU allowlist lacks required stages: {sorted(set(STAGES) - set(expected))}")
    return expected, {"paths": {"placement_manifest": str(placement_path), "cpu_allowlist": str(allowlist_path)}, "hashes": observed_hashes}


class ApprovedMixedStage:
    def __init__(self, stage: str, model_path: Path, profile_dir: Path, expected_cpu_rows: set[tuple[str, str]]) -> None:
        self.stage = stage
        self.model_path = model_path
        self.expected_cpu_rows = expected_cpu_rows
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_dir / f"walloss_revision7_{stage}")
        started = time.monotonic()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=PROVIDERS,
            disabled_optimizers=DISABLED_OPTIMIZERS,
        )
        self.session_create_seconds = time.monotonic() - started
        self.input_meta = {item.name: item for item in self.session.get_inputs()}
        self.output_meta = {item.name: item for item in self.session.get_outputs()}

    def run(self, feeds: Mapping[str, torch.Tensor | np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
        if set(feeds) != set(self.input_meta):
            missing = sorted(set(self.input_meta) - set(feeds))
            extra = sorted(set(feeds) - set(self.input_meta))
            raise ValueError(f"{self.stage} input mismatch: missing={missing}, extra={extra}")
        binding = self.session.io_binding()
        keepalive: list[Any] = []
        for name, meta in self.input_meta.items():
            value = np.ascontiguousarray(value_to_numpy(feeds[name]))
            expected_dtype = numpy_dtype(meta.type)
            if value.shape != fixed_shape(meta) or value.dtype != expected_dtype:
                raise TypeError(f"{self.stage}:{name} expected {fixed_shape(meta)}/{expected_dtype}, got {value.shape}/{value.dtype}")
            if meta.type == "tensor(bfloat16)":
                raw = value.view(np.uint16)
                ort_value = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(raw, TensorProto.BFLOAT16)
                keepalive.extend((value, raw, ort_value))
                binding.bind_ortvalue_input(name, ort_value)
            else:
                keepalive.append(value)
                binding.bind_cpu_input(name, value)

        outputs: dict[str, np.ndarray] = {}
        for name, meta in self.output_meta.items():
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
        self.session.run_with_iobinding(binding)
        return outputs, time.monotonic() - started

    def close(self, profile_dir: Path) -> dict[str, Any]:
        session_providers = self.session.get_providers()
        profile_path = Path(self.session.end_profiling()).resolve()
        if not profile_path.is_relative_to(profile_dir.resolve()):
            raise RuntimeError(f"ORT profile escaped approved profile directory: {profile_path}")
        events = json.loads(profile_path.read_text())
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
        unexpected_providers = {(name, op_type, provider) for name, op_type, provider in provider_rows if provider not in PROVIDERS}
        cuda_present = any(provider == "CUDAExecutionProvider" for _, _, provider in provider_rows)
        placement_passed = bool(session_providers == PROVIDERS and cpu_rows == self.expected_cpu_rows and not unexpected_providers and cuda_present)
        result = {
            "model": {"path": str(self.model_path), "sha256": sha256_file(self.model_path)},
            "profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
            "session_create_seconds": self.session_create_seconds,
            "session_providers": session_providers,
            "disabled_optimizers": sorted(DISABLED_OPTIMIZERS),
            "expected_cpu_node_count": len(self.expected_cpu_rows),
            "profiled_cpu_node_count": len(cpu_rows),
            "cpu_allowlist_exact": cpu_rows == self.expected_cpu_rows,
            "missing_cpu_rows": sorted(self.expected_cpu_rows - cpu_rows),
            "unexpected_cpu_rows": sorted(cpu_rows - self.expected_cpu_rows),
            "unexpected_provider_rows": sorted(unexpected_providers),
            "profiled_memcpy_rows": sorted(memcpy_rows),
            "cuda_provider_present": cuda_present,
            "approved_mixed_placement_passed": placement_passed,
        }
        del self.session
        gc.collect()
        return result


def reset_model(model: Any) -> None:
    model.rope_deltas = None
    model._infer_stable_cache.clear()


def stage_input_names(stage_name: str) -> tuple[str, ...]:
    from pi_cpp_walloss.io_manifest import default_stage_manifests

    stage = next(stage for stage in default_stage_manifests() if stage.name == stage_name)
    return tuple(spec.name for spec in stage.inputs)


def run_prefill(model: Any, prefix_outputs: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    from pi_cpp_walloss.torch_wrappers import PrefillWrapper

    names = stage_input_names("prefill")
    inputs = device_dict({name: prefix_outputs[name] for name in names}, device)
    with torch.inference_mode():
        outputs = PrefillWrapper(model)(inputs).tensors()
    return cpu_dict(outputs)


def run_complete_unpatched_reference(model: Any, prefix_inputs_cpu: Mapping[str, torch.Tensor], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    from pi_cpp_walloss.contract import CONTRACT
    from pi_cpp_walloss.torch_wrappers import PostfixStepWrapper, PrefillWrapper, PrefixEmbedWrapper
    from wall_x.model.core.action.normalizer import unnormalize_data_with_virtual_tail

    prefix_inputs = device_dict(prefix_inputs_cpu, device)
    seed = 20260819
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reset_model(model)
    with torch.inference_mode():
        internal_action = model.generate_flow_action(**export_runner._flow_kwargs(prefix_inputs, None))["predict_action"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generated_x0 = torch.randn(CONTRACT.x0_shape, device=device, dtype=torch.float32)
    if not torch.equal(generated_x0, prefix_inputs["x0"]):
        raise RuntimeError(
            f"legacy x0 does not reproduce command_parity seed {seed}: "
            f"generated={sha256_tensor(generated_x0)}, fixture={sha256_tensor(prefix_inputs['x0'])}"
        )
    reset_model(model)
    with torch.inference_mode():
        external_action = model.generate_flow_action(**export_runner._flow_kwargs(prefix_inputs, prefix_inputs["x0"]))["predict_action"]
    reset_model(model)
    with torch.inference_mode():
        prefix_outputs = PrefixEmbedWrapper(model)(prefix_inputs).tensors
        prefill_names = stage_input_names("prefill")
        prefill_inputs = {name: prefix_outputs[name] for name in prefill_names}
        prefill_result = PrefillWrapper(model)(prefill_inputs)
        prefill_outputs = prefill_result.tensors()
        times = model.action_preprocessor.get_inference_times(CONTRACT.flow_steps, device, torch.float32)
        if not torch.equal(times[0:1], prefix_inputs["t0"]):
            raise RuntimeError("legacy fixture t0 differs from the model flow schedule")
        x_i = prefill_result.x1
        trajectory = {"x1": x_i.detach().cpu()}
        postfix_inputs_by_step: dict[str, dict[str, torch.Tensor]] = {}
        postfix_outputs_by_step: dict[str, torch.Tensor] = {}
        last_postfix_inputs: dict[str, torch.Tensor] | None = None
        kv = {name: value for name, value in prefill_outputs.items() if name.startswith("prefix_kv.")}
        for index in range(1, CONTRACT.flow_steps):
            postfix_inputs = {
                "x_i": x_i,
                "t_i": times[index : index + 1],
                "dt_i": times[index + 1 : index + 2] - times[index : index + 1],
                "dof_mask": prefix_outputs["dof_mask"],
                "v_padding": prefix_outputs["v_padding"],
                "postfix_position_ids": prefix_outputs["postfix_position_ids"],
                "postfix_attention_mask": prefix_outputs["postfix_attention_mask"],
                **kv,
            }
            postfix_inputs_by_step[f"step{index}"] = cpu_dict(postfix_inputs)
            x_i = PostfixStepWrapper(model)(postfix_inputs)
            trajectory[f"x{index + 1}"] = x_i.detach().cpu()
            postfix_outputs_by_step[f"step{index}"] = x_i.detach().cpu()
            last_postfix_inputs = postfix_inputs
        if last_postfix_inputs is None:
            raise RuntimeError("Wall-OSS D10 contract produced no postfix steps")
        final_action = unnormalize_data_with_virtual_tail(
            model.action_preprocessor.normalizer_action, x_i, ["x2_normal"], prefix_inputs["dof_mask"]
        )
    fixture = {
        "prefix_inputs": cpu_dict(prefix_inputs),
        "prefix_outputs": cpu_dict(prefix_outputs),
        "prefill_inputs": cpu_dict(prefill_inputs),
        "prefill_outputs": cpu_dict(prefill_outputs),
        "postfix_inputs": cpu_dict(last_postfix_inputs),
        "postfix_output": x_i.detach().cpu(),
        "postfix_inputs_by_step": postfix_inputs_by_step,
        "postfix_outputs_by_step": postfix_outputs_by_step,
        "trajectory": trajectory,
        "normalized_action": x_i.detach().cpu(),
        "final_action": final_action.detach().cpu(),
        "internal_action": internal_action.detach().cpu(),
        "external_action": external_action.detach().cpu(),
        "wrapper_action": final_action.detach().cpu(),
        "seed": seed,
    }
    gates = {
        "seed": seed,
        "x0_sha256": sha256_tensor(prefix_inputs["x0"]),
        "internal_vs_external": compare_tensor(internal_action, external_action, threshold="external_x0_same_pytorch"),
        "external_vs_wrapper_action": compare_tensor(external_action, final_action, threshold="external_x0_same_pytorch"),
    }
    gates["passed"] = bool(gates["internal_vs_external"]["passed"] and gates["external_vs_wrapper_action"]["passed"])
    return fixture, gates


def run_export_patched_ladder(model: Any, fixture: Mapping[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    from pi_cpp_walloss.contract import CONTRACT
    from pi_cpp_walloss.torch_wrappers import PrefixEmbedWrapper

    prefix_inputs = device_dict(fixture["prefix_inputs"], device)
    vision_module = importlib.import_module(model.visual.__class__.__module__)
    permute_modules = (
        importlib.import_module(model.model.__class__.__module__),
        importlib.import_module("wall_x.model.core.action.moe"),
        importlib.import_module("wall_x.model.core.attention.joint"),
    )
    original_vision_globals = (vision_module.rot_pos_emb, vision_module.get_window_index)
    original_permutes = tuple(module.permute for module in permute_modules)
    attention_state = [
        {
            "attention": block.attn,
            "forward_present": "forward" in block.attn.__dict__,
            "forward": block.attn.__dict__.get("forward"),
            "full_attention_present": "_walloss_use_full_attention" in block.attn.__dict__,
            "full_attention": block.attn.__dict__.get("_walloss_use_full_attention"),
        }
        for block in model.visual.blocks
    ]
    vision_patch: dict[str, Any] | None = None
    mot_patch: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    try:
        with torch.inference_mode():
            vision_patch = export_runner._install_fixed_vision_export_ops(
                model, prefix_inputs["image_grid_thw"], prefix_inputs["pixel_values"]
            )
            mot_patch = export_runner._install_two_expert_permute_export_op(
                model, fixture["prefill_inputs"], fixture["prefill_outputs"]
            )
            reset_model(model)
            patched_prefix = PrefixEmbedWrapper(model)(prefix_inputs).tensors
        patched_prefill = run_prefill(model, patched_prefix, device)
        outputs = {"prefix_outputs": cpu_dict(patched_prefix), "prefill_outputs": patched_prefill}
    finally:
        vision_module.rot_pos_emb, vision_module.get_window_index = original_vision_globals
        for module, original in zip(permute_modules, original_permutes, strict=True):
            module.permute = original
        for state in attention_state:
            attention = state["attention"]
            for name, present_key, value_key in (
                ("forward", "forward_present", "forward"),
                ("_walloss_use_full_attention", "full_attention_present", "full_attention"),
            ):
                if state[present_key]:
                    setattr(attention, name, state[value_key])
                elif name in attention.__dict__:
                    delattr(attention, name)
    if outputs is None or vision_patch is None or mot_patch is None:
        raise RuntimeError("export patch ladder did not produce outputs")
    metrics = {
        "prefix": compare_prefix(fixture["prefix_outputs"], outputs["prefix_outputs"]),
        "prefill": compare_prefill(fixture["prefill_outputs"], outputs["prefill_outputs"]),
        "prefix_embedding_regions": embedding_region_metrics(
            fixture["prefix_outputs"]["prefill_inputs_embeds"],
            outputs["prefix_outputs"]["prefill_inputs_embeds"],
            fixture["prefix_inputs"]["input_ids"],
            prefix_length=CONTRACT.prefix_length,
            image_token_id=CONTRACT.image_token_id,
        ),
        "install": {
            "fixed_vision": vision_patch,
            "two_expert_permute": mot_patch,
            "module_global_state_restored": True,
        },
    }
    metrics["passed"] = bool(rows_pass(metrics["prefix"]) and rows_pass(metrics["prefill"]))
    metrics["passed"] = bool(metrics["passed"] and metrics["prefix_embedding_regions"]["passed"])
    return outputs, metrics


def validate_recovered_fixture(fixture: Mapping[str, Any], decoder_layers: int, flow_steps: int) -> dict[str, Any]:
    required = {
        "prefix_inputs", "prefix_outputs", "prefill_inputs", "prefill_outputs", "postfix_inputs", "postfix_output",
        "postfix_inputs_by_step", "postfix_outputs_by_step", "trajectory", "normalized_action", "final_action", "internal_action",
        "external_action", "wrapper_action",
    }
    missing = sorted(required - set(fixture))
    if missing:
        raise KeyError(f"recovered fixture missing sections: {missing}")
    kv_names = sorted(name for name in fixture["prefill_outputs"] if name.startswith("prefix_kv."))
    expected_trajectory = [f"x{index}" for index in range(1, flow_steps + 1)]
    expected_postfix = [f"step{index}" for index in range(1, flow_steps)]
    trajectory_keys = sorted(fixture["trajectory"], key=lambda name: int(name[1:]))
    postfix_keys = sorted(fixture["postfix_inputs_by_step"], key=lambda name: int(name[4:]))
    if len(kv_names) != decoder_layers * 2 or trajectory_keys != expected_trajectory or postfix_keys != expected_postfix:
        raise RuntimeError(
            f"recovered fixture completeness failed: kv={len(kv_names)}, trajectory={trajectory_keys}, postfix={postfix_keys}"
        )
    all_tensors: list[torch.Tensor] = []
    for section in ("prefix_inputs", "prefix_outputs", "prefill_inputs", "prefill_outputs", "postfix_inputs", "trajectory", "postfix_outputs_by_step"):
        all_tensors.extend(fixture[section].values())
    for step in fixture["postfix_inputs_by_step"].values():
        all_tensors.extend(step.values())
    all_tensors.extend(
        fixture[name] for name in ("postfix_output", "normalized_action", "final_action", "internal_action", "external_action", "wrapper_action")
    )
    finite = all(bool(torch.isfinite(value.float()).all()) for value in all_tensors)
    if not finite:
        raise RuntimeError("recovered fixture contains NaN or Inf")
    return {
        "kv_tensor_count": len(kv_names),
        "kv_layer_count": len({name.split(".")[1] for name in kv_names}),
        "trajectory_keys": trajectory_keys,
        "postfix_step_keys": postfix_keys,
        "all_finite": finite,
        "passed": True,
    }


def run_official_fixture_integrity(script: Path, fixture: Path, profile_dir: Path) -> dict[str, Any]:
    if not script.is_file():
        raise FileNotFoundError(f"official fixture integrity helper is missing: {script}")
    output = profile_dir / "recovered_fixture_integrity.json"
    if output.exists():
        raise FileExistsError(output)
    completed = subprocess.run(
        [sys.executable, str(script), "--fixture", str(fixture), "--output", str(output), "--require-trajectory"],
        check=False,
        capture_output=True,
        text=True,
    )
    if not output.is_file():
        raise RuntimeError(
            f"official fixture integrity helper produced no report: rc={completed.returncode}, "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    report = json.loads(output.read_text())
    return {
        "script": {"path": str(script), "sha256": sha256_file(script)},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": report,
        "paired_fixture_requested": False,
        "passed": bool(completed.returncode == 0 and report.get("passed")),
    }


def legacy_reference_ladder(legacy: Mapping[str, Any], recovered: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prefix": compare_prefix(legacy["prefix_outputs"], recovered["prefix_outputs"]),
        "prefill": compare_prefill(legacy["prefill_outputs"], recovered["prefill_outputs"]),
    }
    legacy_normalized = legacy.get("normalized_action", legacy["postfix_output"])
    result["normalized_action"] = compare_tensor(
        legacy_normalized, recovered["normalized_action"], threshold="final_normalized_action"
    )
    if "final_action" in legacy:
        result["final_action"] = compare_tensor(
            legacy["final_action"], recovered["final_action"], threshold="external_x0_same_pytorch"
        )
    result["passed"] = bool(
        rows_pass(result["prefix"])
        and rows_pass(result["prefill"])
        and result["normalized_action"]["passed"]
        and result.get("final_action", {"passed": True})["passed"]
    )
    return result


def compare_matrix(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {
        "A_vs_B_prefill_backend_on_pt_prefix": ("A", "B"),
        "A_vs_C_prefix_backend_with_pt_prefill": ("A", "C"),
        "A_vs_D_end_to_end": ("A", "D"),
        "B_vs_D_prefix_backend_with_ort_prefill": ("B", "D"),
        "C_vs_D_prefill_backend_on_ort_prefix": ("C", "D"),
    }
    comparisons = {name: compare_prefill(cells[left], cells[right]) for name, (left, right) in pairs.items()}
    return {
        "definitions": {
            "A": "PT prefill(PT prefix)",
            "B": "ORT prefill(PT prefix)",
            "C": "PT prefill(ORT prefix)",
            "D": "ORT prefill(ORT prefix)",
        },
        "cells": {name: section_inventory(outputs) for name, outputs in cells.items()},
        "comparisons": comparisons,
        "passed": all(rows_pass(rows) for rows in comparisons.values()),
    }


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v18-root", type=Path, required=True)
    parser.add_argument("--legacy-fixture", type=Path, required=True)
    parser.add_argument("--output-fixture", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--approved-placement-manifest", type=Path, required=True)
    parser.add_argument("--approved-cpu-allowlist", type=Path, required=True)
    parser.add_argument("--fixture-integrity-script", type=Path, default=Path(__file__).with_name("fixture_integrity.py"))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    from pi_cpp_walloss.contract import CONTRACT

    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.v18_root = args.v18_root.resolve()
    args.legacy_fixture = args.legacy_fixture.resolve()
    args.output_fixture = args.output_fixture.resolve()
    args.output_json = args.output_json.resolve()
    args.profile_dir = args.profile_dir.resolve()
    args.approved_placement_manifest = args.approved_placement_manifest.resolve()
    args.approved_cpu_allowlist = args.approved_cpu_allowlist.resolve()
    args.fixture_integrity_script = args.fixture_integrity_script.resolve()
    output_targets = (args.output_fixture, args.output_json, args.profile_dir)
    if len(set(output_targets)) != len(output_targets):
        raise ValueError("output fixture, JSON, and profile directory must be distinct")
    if args.output_fixture.is_relative_to(args.profile_dir) or args.output_json.is_relative_to(args.profile_dir):
        raise ValueError("output fixture and JSON must not be nested inside the profile directory")
    existing = [str(path) for path in output_targets if path.exists()]
    if existing:
        raise FileExistsError(f"Revision 7 outputs already exist: {existing}")
    if not args.checkpoint.is_dir() or not args.v18_root.is_dir() or not args.legacy_fixture.is_file():
        raise FileNotFoundError("checkpoint, V18 root, or legacy fixture is missing")
    if not (args.checkpoint / "model.safetensors").is_file() or not args.fixture_integrity_script.is_file():
        raise FileNotFoundError("checkpoint model.safetensors or official fixture integrity helper is missing")
    for stage in STAGES:
        if not (args.v18_root / f"{stage}.onnx").is_file():
            raise FileNotFoundError(args.v18_root / f"{stage}.onnx")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Revision 7 same-host diagnostics require a CUDA PyTorch device")
    if not set(PROVIDERS).issubset(ort.get_available_providers()):
        raise RuntimeError(f"ORT lacks an approved provider: {ort.get_available_providers()}")
    expected_cpu_rows, approved_placement = load_approved_placement(
        args.approved_placement_manifest, args.approved_cpu_allowlist
    )
    args.output_fixture.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.profile_dir.mkdir(parents=True)
    device = torch.device(args.device)
    legacy = torch.load(args.legacy_fixture, map_location="cpu", weights_only=True)
    if "prefix_inputs" not in legacy:
        raise KeyError("legacy fixture lacks prefix_inputs")
    prefix_inputs_cpu = cpu_dict(legacy["prefix_inputs"])

    # Phase 1: run and release the smaller V18 prefix session before loading the
    # PyTorch model.  Its CPU outputs are retained for C and D only.
    prefix_session = ApprovedMixedStage(
        "prefix_embed", args.v18_root / "prefix_embed.onnx", args.profile_dir, expected_cpu_rows["prefix_embed"]
    )
    ort_prefix_outputs, ort_prefix_seconds = prefix_session.run(
        {name: prefix_inputs_cpu[name] for name in prefix_session.input_meta}
    )
    prefix_placement = prefix_session.close(args.profile_dir)
    del prefix_session
    release_cuda()

    # Phase 2: all PyTorch work runs without a large ORT prefill session.
    model, _, unpatched_load_report = export_runner._build_model(args.checkpoint, device)
    recovered, generation_gates = run_complete_unpatched_reference(model, prefix_inputs_cpu, device)
    fixture_precheck = validate_recovered_fixture(recovered, CONTRACT.decoder_layers, CONTRACT.flow_steps)
    legacy_ladder = legacy_reference_ladder(legacy, recovered)
    prefix_pt_vs_ort = compare_prefix(recovered["prefix_outputs"], ort_prefix_outputs)
    prefix_pt_vs_ort_regions = embedding_region_metrics(
        recovered["prefix_outputs"]["prefill_inputs_embeds"],
        ort_prefix_outputs["prefill_inputs_embeds"],
        recovered["prefix_inputs"]["input_ids"],
        prefix_length=CONTRACT.prefix_length,
        image_token_id=CONTRACT.image_token_id,
    )
    cell_a = recovered["prefill_outputs"]
    cell_c = run_prefill(model, ort_prefix_outputs, device)
    torch.save(recovered, args.output_fixture)
    output_fixture_sha256 = sha256_file(args.output_fixture)
    reset_model(model)
    del model
    release_cuda()
    official_fixture_integrity = run_official_fixture_integrity(
        args.fixture_integrity_script, args.output_fixture, args.profile_dir
    )

    # Phase 3: the export-patched lane starts from a fresh model because both
    # approved export patches mutate module globals and attention methods.
    patched_model, _, patched_load_report = export_runner._build_model(args.checkpoint, device)
    _, patched_ladder = run_export_patched_ladder(patched_model, recovered, device)
    reset_model(patched_model)
    del patched_model
    release_cuda()

    # Phase 4: only after PyTorch has been destroyed do we create the large V18
    # prefill session.  It is reused for B and D and then immediately released.
    prefill_session = ApprovedMixedStage(
        "prefill", args.v18_root / "prefill.onnx", args.profile_dir, expected_cpu_rows["prefill"]
    )
    prefill_names = set(prefill_session.input_meta)
    cell_b, ort_prefill_pt_prefix_seconds = prefill_session.run(
        {name: recovered["prefix_outputs"][name] for name in prefill_names}
    )
    cell_d, ort_prefill_ort_prefix_seconds = prefill_session.run(
        {name: ort_prefix_outputs[name] for name in prefill_names}
    )
    prefill_placement = prefill_session.close(args.profile_dir)
    del prefill_session
    release_cuda()

    matrix = compare_matrix({"A": cell_a, "B": cell_b, "C": cell_c, "D": cell_d})
    from pi_cpp_walloss import contract as contract_module
    from pi_cpp_walloss import exporter as exporter_module
    from pi_cpp_walloss import torch_wrappers as wrappers_module

    source_modules = {
        "diagnostic_harness": Path(__file__).resolve(),
        "agx_export_runner": Path(export_runner.__file__).resolve(),
        "walloss_contract": Path(contract_module.__file__).resolve(),
        "walloss_exporter": Path(exporter_module.__file__).resolve(),
        "walloss_torch_wrappers": Path(wrappers_module.__file__).resolve(),
        "fixture_integrity": args.fixture_integrity_script,
    }
    v18_tree = sha256_tree(args.v18_root)
    placement_passed = bool(
        prefix_placement["approved_mixed_placement_passed"] and prefill_placement["approved_mixed_placement_passed"]
    )
    numerical_passed = bool(
        generation_gates["passed"]
        and fixture_precheck["passed"]
        and official_fixture_integrity["passed"]
        and legacy_ladder["passed"]
        and rows_pass(prefix_pt_vs_ort)
        and prefix_pt_vs_ort_regions["passed"]
        and patched_ladder["passed"]
        and matrix["passed"]
    )
    report = {
        "schema_version": "walloss.revision7.same-host-prefix-prefill-diagnostics.v1",
        "generated_at_epoch": time.time(),
        "host": platform.node(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "ort_version": ort.__version__,
        "cuda_device": str(device),
        "thresholds": THRESHOLDS,
        "artifact_identity": {
            "checkpoint": {
                "path": str(args.checkpoint),
                "model_safetensors_sha256": sha256_file(args.checkpoint / "model.safetensors"),
            },
            "v18_tree": v18_tree,
            "legacy_fixture": {"path": str(args.legacy_fixture), "sha256": sha256_file(args.legacy_fixture)},
            "output_fixture": {"path": str(args.output_fixture), "sha256": output_fixture_sha256},
            "source_modules": {
                name: {"path": str(path), "sha256": sha256_file(path)} for name, path in source_modules.items()
            },
            "v18_models": {
                stage: {"path": str(args.v18_root / f"{stage}.onnx"), "sha256": sha256_file(args.v18_root / f"{stage}.onnx")}
                for stage in STAGES
            },
        },
        "approved_placement": {
            **approved_placement,
            "expected_hashes": {
                "placement_manifest": APPROVED_PLACEMENT_MANIFEST_SHA256,
                "cpu_allowlist": APPROVED_CPU_ALLOWLIST_SHA256,
            },
            "provider_order": PROVIDERS,
            "disabled_optimizers": sorted(DISABLED_OPTIMIZERS),
        },
        "memory_safety_order": [
            "V18 ORT prefix session -> CPU outputs -> destroy session",
            "fresh unpatched PyTorch model under inference_mode -> CPU outputs -> save fixture -> destroy model",
            "fresh export-patched PyTorch model -> restore module globals/attention methods -> destroy model",
            "V18 ORT prefill session -> B/D -> destroy session",
        ],
        "load_report": {"unpatched": unpatched_load_report, "export_patched": patched_load_report},
        "fixture_inventory_precheck": fixture_precheck,
        "official_fixture_integrity": official_fixture_integrity,
        "fixture_inventory": {
            "prefix_inputs": section_inventory(recovered["prefix_inputs"]),
            "prefix_outputs": section_inventory(recovered["prefix_outputs"]),
            "prefill_inputs": section_inventory(recovered["prefill_inputs"]),
            "prefill_outputs": section_inventory(recovered["prefill_outputs"]),
            "trajectory": section_inventory(recovered["trajectory"]),
            "normalized_action": tensor_summary(recovered["normalized_action"]),
            "final_action": tensor_summary(recovered["final_action"]),
        },
        "external_x0_full_generate_vs_wrapper": generation_gates,
        "legacy_vs_unpatched_pytorch": legacy_ladder,
        "unpatched_vs_export_patched_pytorch": patched_ladder,
        "prefix_pt_vs_ort": {
            "execution_seconds": ort_prefix_seconds,
            "outputs": section_inventory(ort_prefix_outputs),
            "metrics": prefix_pt_vs_ort,
            "embedding_regions": prefix_pt_vs_ort_regions,
            "passed": bool(rows_pass(prefix_pt_vs_ort) and prefix_pt_vs_ort_regions["passed"]),
        },
        "prefix_prefill_2x2": {
            **matrix,
            "ort_execution_seconds": {
                "prefix": ort_prefix_seconds,
                "B_prefill_pt_prefix": ort_prefill_pt_prefix_seconds,
                "D_prefill_ort_prefix": ort_prefill_ort_prefix_seconds,
            },
        },
        "provider_placement": {
            "prefix_embed": prefix_placement,
            "prefill": prefill_placement,
            "passed": placement_passed,
        },
        "decision": {
            "fixture_inventory_precheck_passed": fixture_precheck["passed"],
            "fixture_recovery_passed": official_fixture_integrity["passed"],
            "external_x0_seam_passed": generation_gates["passed"],
            "same_host_pytorch_ladder_passed": bool(legacy_ladder["passed"] and patched_ladder["passed"]),
            "prefix_prefill_2x2_passed": matrix["passed"],
            "approved_provider_placement_passed": placement_passed,
            "revision7_5090_diagnostic_passed": bool(numerical_passed and placement_passed),
            "claim_limit": (
                "single approved Revision 7 fixture same-host PT/ORT prefix-prefill diagnostics only; "
                "not AGX, TensorRT, latency, server/client, closed-loop, or promotion evidence"
            ),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    if not report["decision"]["revision7_5090_diagnostic_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
