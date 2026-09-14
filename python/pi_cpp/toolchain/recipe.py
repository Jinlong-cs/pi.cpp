"""Build recipes + the environment trap table for the picpp toolchain.

A recipe is the machine-readable form of a board build script: the input
ONNX, the precision policy (bf16/fp16 hybrid or calibrator-free qdq), the
workspace pool size, and the layer-forcing rule. The traps that burned the
board rounds become tool behavior:

- the workspace trap: 1GB builds starve the M=51 gemm tactics; the recipe
  carries the workspace and a 4GB default,
- the calibrator trap: TRT 10.3's INT8 entropy calibrator asserts
  (commonEmitDebugTensor) on bf16 networks in this JetPack env; a recipe
  that asks for the calibrator is refused with the trap message,
- the scale-consistency trap: the qdq recipe needs an activation scale for
  every selected MatMul; missing scales fail fast in the graph transform.

Every built engine feeds the deployment-manifest lock: the builder returns
the engine's size_bytes + sha256 so ``picpp package lock`` can seal it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECIPE_SCHEMA = "picpp.build.recipe.v1"
DEFAULT_WORKSPACE_GB = 4
DEFAULT_MLP_CONTAINS = ("/mlp",)
DEFAULT_MLP_PREFIXES = ("/time_mlp",)

# Known environment traps: the operation -> why it is refused. A recipe that
# selects a trapped path fails fast with this message instead of burning a
# board round on the broken TRT calibrator.
ENV_TRAPS: dict[str, str] = {
    "trt_int8_calibrator": (
        "TRT 10.3 INT8 entropy calibrator is broken on this JetPack env "
        "(commonEmitDebugTensor assertion on bf16 networks); use the "
        "calibrator-free qdq recipe with picpp calibrate + picpp graph qdq"
    ),
}

_VALID_PRECISIONS = ("bf16", "fp16", "qdq")


@dataclass(frozen=True)
class BuildRecipe:
    onnx: str
    output: str
    precision: str = "bf16"
    workspace_gb: int = DEFAULT_WORKSPACE_GB
    obey_precision: bool = True
    fp32_force_except: str | None = None  # layer-name marker kept at the recipe precision
    mlp_contains: tuple[str, ...] = DEFAULT_MLP_CONTAINS
    mlp_prefixes: tuple[str, ...] = DEFAULT_MLP_PREFIXES
    act_scales: str | None = None  # npz path from picpp calibrate (qdq only)
    flags: tuple[str, ...] = ()

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> BuildRecipe:
        precision = str(raw.get("precision", "bf16"))
        if precision == "int8":
            raise ValueError(ENV_TRAPS["trt_int8_calibrator"])
        if precision not in _VALID_PRECISIONS:
            raise ValueError(f"recipe precision must be one of {_VALID_PRECISIONS}, got {precision!r}")
        recipe = BuildRecipe(
            onnx=str(raw["onnx"]),
            output=str(raw["output"]),
            precision=precision,
            workspace_gb=int(raw.get("workspace_gb", DEFAULT_WORKSPACE_GB)),
            obey_precision=bool(raw.get("obey_precision", True)),
            fp32_force_except=raw.get("fp32_force_except"),
            mlp_contains=tuple(raw.get("mlp_contains", DEFAULT_MLP_CONTAINS)),
            mlp_prefixes=tuple(raw.get("mlp_prefixes", DEFAULT_MLP_PREFIXES)),
            act_scales=raw.get("act_scales"),
            flags=tuple(raw.get("flags", ())),
        )
        recipe.check_traps()
        return recipe

    @staticmethod
    def load(path: str | Path) -> BuildRecipe:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return BuildRecipe.from_dict(raw)

    def check_traps(self) -> None:
        if self.workspace_gb < 1:
            raise ValueError("workspace_gb must be >= 1")
        if self.precision == "qdq" and not self.act_scales:
            raise ValueError("the qdq recipe requires act_scales from picpp calibrate")


def engine_lock_entry(path: Path) -> dict[str, Any]:
    """The lock payload for a built engine: size + sha256."""
    data = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


__all__ = [
    "DEFAULT_WORKSPACE_GB",
    "ENV_TRAPS",
    "RECIPE_SCHEMA",
    "BuildRecipe",
    "engine_lock_entry",
]
