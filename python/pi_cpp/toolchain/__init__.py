"""The picpp board toolchain: graph transforms, calibration, engine builds.

All board-side dependencies (tensorrt, onnxruntime) are imported lazily so
the host-side graph transforms and tests run anywhere onnx is installed.
"""

from pi_cpp.toolchain.graph import cast_cache_abi, count_mlp_matmuls, insert_qdq, topo_sort
from pi_cpp.toolchain.recipe import (
    DEFAULT_WORKSPACE_GB,
    ENV_TRAPS,
    RECIPE_SCHEMA,
    BuildRecipe,
    engine_lock_entry,
)

__all__ = [
    "DEFAULT_WORKSPACE_GB",
    "ENV_TRAPS",
    "RECIPE_SCHEMA",
    "BuildRecipe",
    "cast_cache_abi",
    "count_mlp_matmuls",
    "engine_lock_entry",
    "insert_qdq",
    "topo_sort",
]
