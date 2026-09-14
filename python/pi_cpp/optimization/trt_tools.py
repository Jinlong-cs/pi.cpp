"""Structured TensorRT build, inspection, and stage benchmarking via trtexec."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from pi_cpp.optimization.contracts import TargetFingerprint


def read_shapes(path: Path | None) -> dict[str, tuple[int, ...]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("shape JSON must map input names to positive integer dimensions")
    shapes: dict[str, tuple[int, ...]] = {}
    for name, dimensions in raw.items():
        if (
            not isinstance(name, str)
            or not isinstance(dimensions, list)
            or not all(isinstance(value, int) and value > 0 for value in dimensions)
        ):
            raise TypeError(f"invalid shape entry: {name!r}={dimensions!r}")
        shapes[name] = tuple(dimensions)
    return shapes


def shape_argument(shapes: dict[str, tuple[int, ...]]) -> str | None:
    if not shapes:
        return None
    return ",".join(
        f"{name}:{'x'.join(str(value) for value in dimensions)}"
        for name, dimensions in sorted(shapes.items())
    )


def _run(command: list[str], log_path: Path) -> dict[str, Any]:
    if log_path.exists():
        raise FileExistsError(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}; see {log_path}")
    return {"command": command, "returncode": completed.returncode, "log": str(log_path)}


def _version(trtexec: str) -> str:
    completed = subprocess.run(
        [trtexec, "--version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"TensorRT v([^\]\s]+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"could not parse TensorRT version from {trtexec} --version")
    return match.group(1)


def _cuda_fingerprint() -> dict[str, str | None]:
    nvcc = shutil.which("nvcc")
    if nvcc:
        completed = subprocess.run(
            [nvcc, "--version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r"release\s+([0-9.]+)", completed.stdout)
        if match:
            return {"version": match.group(1), "source": "nvcc", "path": nvcc}

    for path in (Path("/usr/local/cuda/version.json"), Path("/usr/local/cuda/version.txt")):
        if not path.is_file():
            continue
        contents = path.read_text()
        if path.suffix == ".json":
            value = json.loads(contents).get("cuda", {}).get("version")
        else:
            match = re.search(r"CUDA Version\s+([0-9.]+)", contents)
            value = match.group(1) if match else None
        if value:
            return {"version": str(value), "source": path.name, "path": str(path)}

    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        match = re.search(r"CUDA Version:\s*([0-9.]+)", completed.stdout)
        if match:
            return {"version": match.group(1), "source": "nvidia-smi", "path": None}
    return {"version": None, "source": "unavailable", "path": None}


def target_fingerprint(name: str, architecture: str, trtexec: str) -> TargetFingerprint:
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        hardware["nvidia_smi"] = [
            line.strip() for line in query.stdout.splitlines() if line.strip()
        ]
    return TargetFingerprint(
        name=name,
        architecture=architecture,
        hardware=hardware,
        software={
            "cuda": _cuda_fingerprint(),
            "trtexec": str(Path(trtexec)),
            "trtexec_version": _version(trtexec),
        },
    )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    shapes: dict[str, tuple[int, ...]],
    *,
    fp16: bool,
    int8: bool,
    builder_optimization_level: int,
    workspace_mib: int,
    detailed: bool,
    trtexec: str,
    log_path: Path,
    layer_info_path: Path,
    target_name: str,
    architecture: str,
) -> dict[str, Any]:
    if engine_path.exists() or layer_info_path.exists():
        raise FileExistsError(engine_path if engine_path.exists() else layer_info_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    layer_info_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--builderOptimizationLevel={builder_optimization_level}",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--skipInference",
        f"--exportLayerInfo={layer_info_path}",
    ]
    shape_value = shape_argument(shapes)
    if shape_value:
        command.append(f"--shapes={shape_value}")
    if fp16:
        command.append("--fp16")
    if int8:
        command.append("--int8")
    if detailed:
        command.append("--profilingVerbosity=detailed")
    execution = _run(command, log_path)
    if not engine_path.is_file():
        raise FileNotFoundError(engine_path)
    return {
        "kind": "tensorrt_build",
        "timing_boundary": "engine build wall time is recorded in the trtexec log; inference is skipped",
        "target": target_fingerprint(target_name, architecture, trtexec).to_dict(),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "shapes": {name: list(value) for name, value in shapes.items()},
        "flags": {
            "fp16": fp16,
            "int8": int8,
            "builder_optimization_level": builder_optimization_level,
            "workspace_mib": workspace_mib,
            "detailed": detailed,
        },
        "layer_info": str(layer_info_path),
        "execution": execution,
    }


def inspect_engine(
    engine_path: Path,
    shapes: dict[str, tuple[int, ...]],
    *,
    trtexec: str,
    log_path: Path,
    layer_info_path: Path,
    target_name: str,
    architecture: str,
) -> dict[str, Any]:
    if layer_info_path.exists():
        raise FileExistsError(layer_info_path)
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        "--skipInference",
        "--profilingVerbosity=detailed",
        f"--exportLayerInfo={layer_info_path}",
    ]
    shape_value = shape_argument(shapes)
    if shape_value:
        command.append(f"--shapes={shape_value}")
    execution = _run(command, log_path)
    return {
        "kind": "tensorrt_engine_inspect",
        "target": target_fingerprint(target_name, architecture, trtexec).to_dict(),
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "shapes": {name: list(value) for name, value in shapes.items()},
        "layer_info": str(layer_info_path),
        "execution": execution,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _timing_summary(times_path: Path) -> dict[str, dict[str, float | int]]:
    raw = json.loads(times_path.read_text())
    if not isinstance(raw, list):
        raise TypeError(f"unexpected trtexec timing JSON in {times_path}")
    keys = sorted(
        {
            key
            for record in raw
            if isinstance(record, dict)
            for key, value in record.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: _summary(
            [
                float(record[key])
                for record in raw
                if isinstance(record, dict) and isinstance(record.get(key), (int, float))
            ]
        )
        for key in keys
    }


def benchmark_engine(
    engine_path: Path,
    shapes: dict[str, tuple[int, ...]],
    *,
    warmup_ms: int,
    repeats: int,
    trtexec: str,
    log_path: Path,
    times_path: Path,
    profile_path: Path,
    target_name: str,
    architecture: str,
) -> dict[str, Any]:
    for path in (times_path, profile_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--warmUp={warmup_ms}",
        f"--iterations={repeats}",
        "--duration=0",
        "--noDataTransfers",
        "--dumpProfile",
        "--separateProfileRun",
        f"--exportTimes={times_path}",
        f"--exportProfile={profile_path}",
    ]
    shape_value = shape_argument(shapes)
    if shape_value:
        command.append(f"--shapes={shape_value}")
    execution = _run(command, log_path)
    if not times_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("trtexec did not produce timing/profile JSON")
    return {
        "kind": "tensorrt_stage_benchmark",
        "timing_boundary": "single TensorRT engine compute using trtexec --noDataTransfers; excludes H2D/D2H, model preprocessing, inter-stage transfers, policy loop, server/client, and closed loop",
        "target": target_fingerprint(target_name, architecture, trtexec).to_dict(),
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "shapes": {name: list(value) for name, value in shapes.items()},
        "warmup_ms": warmup_ms,
        "requested_iterations": repeats,
        "timings": _timing_summary(times_path),
        "times": str(times_path),
        "profile": str(profile_path),
        "execution": execution,
    }
