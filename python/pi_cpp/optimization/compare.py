"""Numerical comparison for FP and optimized model outputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def compare_array(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = candidate64 - reference64
    reference_norm = float(np.linalg.norm(reference64.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    denominator = max(reference_norm, np.finfo(np.float64).eps)
    dot = float(np.dot(reference64.ravel(), candidate64.ravel()))
    candidate_norm = float(np.linalg.norm(candidate64.ravel()))
    cosine_denominator = reference_norm * candidate_norm
    noise_power = float(np.mean(np.square(difference)))
    signal_power = float(np.mean(np.square(reference64)))
    sqnr = (
        math.inf
        if noise_power == 0.0
        else 10.0 * math.log10(max(signal_power, np.finfo(np.float64).tiny) / noise_power)
    )
    return {
        "shape_match": True,
        "shape": list(reference.shape),
        "reference_finite": bool(np.all(np.isfinite(reference64))),
        "candidate_finite": bool(np.all(np.isfinite(candidate64))),
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "l2": difference_norm,
        "relative_l2": difference_norm / denominator,
        "cosine": 1.0
        if difference_norm == 0.0
        else (dot / cosine_denominator if cosine_denominator else 0.0),
        "sqnr_db": sqnr,
    }


def compare_outputs(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError(
            f"output names differ: reference={sorted(reference)}, candidate={sorted(candidate)}"
        )
    tensors = {name: compare_array(reference[name], candidate[name]) for name in sorted(reference)}
    return {
        "output_names": sorted(reference),
        "all_shapes_match": all(value["shape_match"] for value in tensors.values()),
        "all_finite": all(
            value.get("reference_finite", False) and value.get("candidate_finite", False)
            for value in tensors.values()
        ),
        "tensors": tensors,
    }


def run_onnx(
    path: Path, inputs: dict[str, np.ndarray], providers: list[str]
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=providers)
    values = session.run(
        None, {name: np.ascontiguousarray(value) for name, value in inputs.items()}
    )
    return {output.name: value for output, value in zip(session.get_outputs(), values, strict=True)}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.ascontiguousarray(values[name]) for name in values.files}
