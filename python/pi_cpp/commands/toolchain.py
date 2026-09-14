"""picpp graph / calibrate / build: the board build pipeline as tool behavior.

Each command writes a JSON report on stdout and exits 0 on success, 1 on
failure. The traps are encoded in the tool: the broken TRT INT8 calibrator
is refused by the recipe (fail-fast), the workspace size rides the recipe,
and the qdq path demands complete activation scales.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import onnx
import tyro

from pi_cpp.toolchain.build import build_engine
from pi_cpp.toolchain.calibrate import collect_scales
from pi_cpp.toolchain.graph import cast_cache_abi, insert_qdq
from pi_cpp.toolchain.recipe import BuildRecipe


@dataclass
class GraphConfig:
    action: Annotated[Literal["cast-caches", "qdq"], tyro.conf.Positional]
    onnx: Path
    out: Path
    stage: Literal["prefix", "suffix"] = "suffix"
    scales: Path | None = None
    mlp_contains: tuple[str, ...] = ("/mlp",)
    mlp_prefixes: tuple[str, ...] = ("/time_mlp",)


@dataclass
class CalibrateConfig:
    onnx: Path
    calib: Path
    out: Path
    mlp_contains: tuple[str, ...] = ("/mlp",)
    mlp_prefixes: tuple[str, ...] = ("/time_mlp",)


@dataclass
class BuildConfig:
    recipe: Path
    onnx_dir: Path | None = None
    out_dir: Path | None = None


def _save(model, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)


def _dump(report: dict) -> int:
    print(json.dumps(report, indent=2))
    return 0


def run_graph(command: GraphConfig) -> int:
    model = onnx.load(command.onnx, load_external_data=False)
    if command.action == "cast-caches":
        cast_cache_abi(model, stage=command.stage)
        _save(model, command.out)
        return _dump({"graph": command.action, "stage": command.stage, "out": str(command.out)})
    scales_path = command.scales
    if scales_path is None:
        raise ValueError("picpp graph qdq requires --scales (from picpp calibrate)")
    scales_data = np.load(scales_path)
    act_scales = {key: float(scales_data[key]) for key in scales_data.files}
    model, quantized = insert_qdq(
        model, act_scales=act_scales, name_contains=command.mlp_contains, name_prefixes=command.mlp_prefixes
    )
    _save(model, command.out)
    return _dump({"graph": command.action, "quantized_matmuls": quantized, "out": str(command.out)})


def run_calibrate(command: CalibrateConfig) -> int:
    report = collect_scales(
        command.onnx, command.calib, command.out, name_contains=command.mlp_contains, name_prefixes=command.mlp_prefixes
    )
    return _dump(report)


def run_build(command: BuildConfig) -> int:
    recipe = BuildRecipe.load(command.recipe)
    onnx_dir = command.onnx_dir or command.recipe.parent
    out_dir = command.out_dir or command.recipe.parent
    report = build_engine(recipe, onnx_dir, out_dir)
    return _dump(report)


def run_toolchain(command: GraphConfig | CalibrateConfig | BuildConfig) -> int:
    if isinstance(command, GraphConfig):
        return run_graph(command)
    if isinstance(command, CalibrateConfig):
        return run_calibrate(command)
    return run_build(command)
