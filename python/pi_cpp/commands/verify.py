"""picpp verify: the acceptance ladder as a command.

parity: run the package engines over the frozen golden cases, compare the
surfaces against the JAX golden (scaled triplet) and against the saved
direct-TensorRT reference (raw-space), then evaluate the manifest gates
(family defaults + per-package overrides). Exit 0 when every check passes,
1 otherwise; the JSON report carries each observed/threshold pair.

latency: the D10 protocol from the gates (warmup + measure runs) over the
first case; the numeric bounds are optional (report-only without them).

The golden data contract follows the deployment protocol: cases.json +
frozen_inputs + the golden manifest with sha256-payload records; the
reference surfaces are the harness-saved actions_raw_16.npy files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal

import numpy as np
import tyro

from pi_cpp.package.gates import gates_for
from pi_cpp.verify.metrics import evaluate_latency, evaluate_parity, raw_max_abs, triplet

SURFACE_SHAPE = (50, 16)


def payload_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def golden_case_ids(golden_root: Path) -> tuple[str, ...]:
    cases = _read_json(golden_root / "cases.json")["golden_cases"]
    return tuple(case["case_id"] for case in cases)


def load_golden_raw32(golden_root: Path, case_id: str) -> np.ndarray:
    golden = _read_json(golden_root / "manifest.json")
    record = golden["authoritative_public_api_raw32"][case_id]
    path = golden_root / record["path"]
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if payload_hash(value) != record["sha256_payload"]:
        raise ValueError(f"golden payload hash mismatch for {case_id}")
    if value.shape != SURFACE_SHAPE:
        raise ValueError(f"golden surface shape {value.shape}, expected {SURFACE_SHAPE}")
    return np.ascontiguousarray(value)


def load_frozen_case(golden_root: Path, case_id: str) -> dict[str, np.ndarray]:
    case_dir = golden_root / "frozen_inputs" / "run_a" / case_id
    noise_files = list((golden_root / "frozen_inputs").glob("noise_seed_*.npy"))
    if len(noise_files) != 1:
        raise ValueError(f"expected exactly one noise_seed_*.npy in frozen_inputs, found {len(noise_files)}")
    paths = {
        "image": case_dir / "image_float32_nchw.npy",
        "image_mask": case_dir / "image_mask_bool.npy",
        "tokenized_prompt": case_dir / "token_ids_int64.npy",
        "tokenized_prompt_mask": case_dir / "token_mask_bool.npy",
        "x_t": noise_files[0],
        "state": case_dir / "normalized_state.npy",
        "embodiment_id": case_dir / "embodiment_id_int32.npy",
        "delay": case_dir / "rtc_delay_int32.npy",
        "action_prefix": case_dir / "rtc_action_prefix_normalized.npy",
    }
    values = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
    return {name: np.ascontiguousarray(value) for name, value in values.items()}


def unnormalize_actions(normalized: np.ndarray, norm_stats: Path) -> np.ndarray:
    values = np.asarray(normalized, dtype=np.float32).reshape(SURFACE_SHAPE)
    stats = _read_json(norm_stats)["norm_stats"]["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return np.ascontiguousarray((values + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01)


def pin_rtc_prefix(delay: np.ndarray, action_prefix: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Pin the executed prefix positions (idempotent; delay=0 leaves the chunk
    untouched). The native runner already applies the final pin on the host;
    this mirrors the JAX reference exactly."""
    delay_value = int(np.asarray(delay).reshape(-1)[0])
    if delay_value <= 0:
        return np.asarray(action, dtype=np.float32)
    pinned = np.array(action, dtype=np.float32, copy=True)
    if pinned.ndim == 3:
        pinned[0, :delay_value, :] = action_prefix[0, :delay_value, :]
    else:
        pinned[:delay_value, :] = action_prefix[0, :delay_value, :]
    return pinned


def _gates_for_package(package: Path) -> tuple[str, dict]:
    deployment = package / "deployment_manifest.json"
    if deployment.is_file():
        manifest = _read_json(deployment)
        family = manifest["model"]["family"]
        return family, gates_for(family, manifest.get("gates"))
    export = _read_json(package / "export_manifest.json")
    family = export["model_family"].split("_")[0]
    return family, gates_for(family, None)


def _load_per_dim_scale(contract: Path | None, golden_root: Path) -> np.ndarray:
    if contract is not None:
        return np.asarray(_read_json(contract)["per_dim_action_scale"], dtype=np.float64)
    candidate = golden_root / "model_contract.json"
    if candidate.is_file():
        return np.asarray(_read_json(candidate)["per_dim_action_scale"], dtype=np.float64)
    raise FileNotFoundError("per-dim action scale not found; pass --contract or place model_contract.json in the golden root")


def _engine_tensors(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    base = {
        name: inputs[name]
        for name in ("image", "image_mask", "tokenized_prompt", "tokenized_prompt_mask", "x_t", "state", "embodiment_id")
    }
    if "delay" in inputs:
        base["delay"] = np.ascontiguousarray(inputs["delay"], dtype=np.int64)
    if "action_prefix" in inputs:
        base["action_prefix"] = np.ascontiguousarray(inputs["action_prefix"], dtype=np.float32)
    return base


def run_parity(command: VerifyConfig, gates: dict, report: dict) -> int:
    import pi_cpp as picpp

    case_ids = golden_case_ids(command.golden)
    golden = {case_id: load_golden_raw32(command.golden, case_id) for case_id in case_ids}
    runner = picpp.Pi05OfflineRunner(command.package / "engines")
    runner.load()
    surfaces = {}
    for case_id in case_ids:
        inputs = load_frozen_case(command.golden, case_id)
        result = runner.run_once(_engine_tensors(inputs))
        action = np.asarray(result.action, dtype=np.float32).reshape(SURFACE_SHAPE)
        surfaces[case_id] = pin_rtc_prefix(inputs["delay"], inputs["action_prefix"], action)
    scale = _load_per_dim_scale(command.contract, command.golden)
    comparison = triplet(
        golden,
        {case_id: unnormalize_actions(surface, command.norm_stats) for case_id, surface in surfaces.items()},
        scale,
    )
    observed = {
        "aggregate": comparison["aggregate"],
        "runtime_raw_max_abs": raw_max_abs(_load_reference_surfaces(command.reference), surfaces) if command.reference else 0.0,
    }
    report["picpp_vs_golden"] = comparison
    report["runtime_raw_max_abs"] = observed["runtime_raw_max_abs"]
    report["gates"] = evaluate_parity(observed, gates)
    return 0 if report["gates"]["passed"] else 1


def _load_reference_surfaces(reference: Path) -> dict[str, np.ndarray]:
    surfaces = {}
    for case_dir in sorted(reference.iterdir()):
        path = case_dir / "actions_raw_16.npy"
        if path.is_file():
            surfaces[case_dir.name] = np.load(path, allow_pickle=False).astype(np.float32)
    return surfaces


def run_latency(command: VerifyConfig, gates: dict, report: dict) -> int:
    import pi_cpp as picpp

    protocol = gates.get("latency") or {}
    warmup = int(protocol.get("warmup", 10))
    runs = int(protocol.get("runs", 100))
    case_id = golden_case_ids(command.golden)[0]
    inputs = _engine_tensors(load_frozen_case(command.golden, case_id))
    runner = picpp.Pi05OfflineRunner(command.package / "engines")
    runner.load()
    for _ in range(warmup):
        runner.run_once(inputs)
    samples = []
    for _ in range(runs):
        start = perf_counter()
        runner.run_once(inputs)
        samples.append((perf_counter() - start) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    latency = {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "warmup_runs": warmup,
        "measure_runs": runs,
    }
    report["latency"] = latency
    report["gates"] = evaluate_latency(latency, gates)
    return 0 if report["gates"]["passed"] else 1


@dataclass(kw_only=True)
class VerifyConfig:
    mode: Annotated[Literal["parity", "latency"], tyro.conf.Positional]
    package: Path
    golden: Path
    norm_stats: Path
    contract: Path | None = None
    reference: Path | None = None
    out: Path | None = None


def run_verify(command: VerifyConfig) -> int:
    family, gates = _gates_for_package(command.package)
    report: dict = {"mode": command.mode, "family": family}
    if command.mode == "parity":
        exit_code = run_parity(command, gates, report)
    else:
        exit_code = run_latency(command, gates, report)
    payload = json.dumps(report, indent=2)
    if command.out is not None:
        command.out.parent.mkdir(parents=True, exist_ok=True)
        command.out.write_text(payload)
    else:
        print(payload)
    return exit_code
