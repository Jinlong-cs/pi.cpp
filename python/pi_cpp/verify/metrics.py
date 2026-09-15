"""Acceptance metrics + manifest-gate evaluation for picpp verify.

The metric computation is the ported harness compare_actions / raw gate;
the thresholds come from the deployment manifest gates (family defaults +
per-package overrides), not from hardcoded contract constants.

Gate semantics (all fail-closed when a threshold is present):
- parity: aggregate relative_l2 (<=), cosine (>=), scaled max_abs (<=)
  against the JAX golden, plus the raw-space max_abs between the picpp
  runtime and the direct-TensorRT reference (<=).
- latency: p50/p95/mean upper bounds when the package declares them; the
  D10 protocol itself (warmup + runs) is enforced by the runner loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12


def cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference)) * float(np.linalg.norm(candidate))
    if denominator < EPS:
        return 1.0 if np.allclose(reference, candidate) else 0.0
    return float(np.dot(reference, candidate) / denominator)


def triplet(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], scale: np.ndarray) -> dict[str, Any]:
    """Per-case + aggregate rel_l2 / cosine / scaled max_abs for two surfaces."""
    cases = []
    for case_id, ref_values in reference.items():
        ref = (ref_values / scale).astype(np.float64)
        cand = (candidate[case_id] / scale).astype(np.float64)
        delta = cand - ref
        cases.append(
            {
                "case_id": case_id,
                "relative_l2": float(np.linalg.norm(delta) / max(float(np.linalg.norm(ref)), EPS)),
                "cosine": cosine(ref.ravel(), cand.ravel()),
                "max_abs_scaled": float(np.max(np.abs(delta))),
            }
        )
    ref_all = np.concatenate([reference[case_id] / scale for case_id in reference])
    cand_all = np.concatenate([candidate[case_id] / scale for case_id in reference])
    delta_all = cand_all - ref_all
    aggregate = {
        "relative_l2": float(np.linalg.norm(delta_all) / max(float(np.linalg.norm(ref_all)), EPS)),
        "cosine": cosine(ref_all.ravel(), cand_all.ravel()),
        "max_abs_scaled": float(np.max(np.abs(delta_all))),
    }
    return {
        "aggregate": aggregate,
        "worst_case_relative_l2": max(case["relative_l2"] for case in cases),
        "worst_scaled_max_abs": max(case["max_abs_scaled"] for case in cases),
        "cases": cases,
    }


def raw_max_abs(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> float:
    return float(
        np.max(np.concatenate([np.abs(reference[case_id] - candidate[case_id]).ravel() for case_id in reference]))
    )


def evaluate_parity(observed: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the triplet + raw surfaces against the manifest parity gates."""
    parity_gates = gates.get("parity") or {}
    checks: dict[str, Any] = {}
    aggregate = observed["aggregate"]
    if "rel_l2" in parity_gates:
        checks["aggregate_relative_l2"] = {
            "observed": aggregate["relative_l2"],
            "threshold": parity_gates["rel_l2"],
            "passed": aggregate["relative_l2"] <= parity_gates["rel_l2"],
        }
    if "cosine" in parity_gates:
        checks["aggregate_cosine"] = {
            "observed": aggregate["cosine"],
            "threshold": parity_gates["cosine"],
            "passed": aggregate["cosine"] >= parity_gates["cosine"],
        }
    if "max_abs_scaled" in parity_gates:
        checks["scaled_max_abs"] = {
            "observed": aggregate["max_abs_scaled"],
            "threshold": parity_gates["max_abs_scaled"],
            "passed": aggregate["max_abs_scaled"] <= parity_gates["max_abs_scaled"],
        }
    if "worst_rel_l2" in parity_gates:
        checks["worst_case_relative_l2"] = {
            "observed": observed["worst_case_relative_l2"],
            "threshold": parity_gates["worst_rel_l2"],
            "passed": observed["worst_case_relative_l2"] <= parity_gates["worst_rel_l2"],
        }
    if "raw" in parity_gates:
        checks["runtime_raw_max_abs"] = {
            "observed": observed["runtime_raw_max_abs"],
            "threshold": parity_gates["raw"],
            "passed": observed["runtime_raw_max_abs"] <= parity_gates["raw"],
        }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def evaluate_latency(latency: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the latency report against the manifest latency gates.

    The numeric bounds are optional: a package without them gets a
    report-only result (the family defaults record the D10 protocol, the
    deployment band is package evidence).
    """
    latency_gates = gates.get("latency") or {}
    checks: dict[str, Any] = {}
    for key in ("p50_ms", "p95_ms", "mean_ms"):
        bound = latency_gates.get(f"{key}_max")
        if bound is not None:
            checks[key] = {"observed": latency[key], "threshold": bound, "passed": latency[key] <= bound}
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "report_only": not checks,
    }


def load_latency_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "cosine",
    "evaluate_latency",
    "evaluate_parity",
    "load_latency_report",
    "raw_max_abs",
    "triplet",
]
