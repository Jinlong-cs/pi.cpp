"""Board-side TRT engine builder (the ported build_hybrid_suffix_4g.py +
build_suffix_qdq.py build step, driven by a BuildRecipe).

Runs where TensorRT lives (the AGX); ``import tensorrt`` stays lazy so the
host-side tests never need it. Returns the engine lock entry (size + sha256)
so the deployment manifest can seal the artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pi_cpp.toolchain.recipe import BuildRecipe, engine_lock_entry


def _is_mlp_layer_name(name: str, recipe: BuildRecipe) -> bool:
    return any(marker in name for marker in recipe.mlp_contains) or any(
        name.startswith(prefix) for prefix in recipe.mlp_prefixes
    )


def build_engine(recipe: BuildRecipe, onnx_dir: Path, out_dir: Path) -> dict[str, Any]:
    import tensorrt as trt

    recipe.check_traps()
    onnx_path = onnx_dir / recipe.onnx
    if not onnx_path.is_file():
        raise FileNotFoundError(f"recipe onnx not found: {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            raise ValueError(f"failed to parse {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, recipe.workspace_gb << 30)
    if recipe.precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    if recipe.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if recipe.obey_precision:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    forced = 0
    kept = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.num_outputs == 0:
            continue
        dtype = layer.get_output(0).dtype
        if dtype in (trt.float32, trt.float16, trt.bfloat16):
            if recipe.fp32_force_except and recipe.fp32_force_except in layer.name or recipe.precision == "qdq" and _is_mlp_layer_name(layer.name, recipe):
                kept += 1
            else:
                layer.precision = trt.float32
                forced += 1
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build returned no engine")
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = out_dir / recipe.output
    engine_path.write_bytes(bytes(serialized))
    stats = {
        "engine": engine_path.name,
        "size_bytes": engine_path.stat().st_size,
        "fp32_forced_layers": forced,
        "kept_layers": kept,
        "workspace_gb": recipe.workspace_gb,
        "precision": recipe.precision,
    }
    stats.update(engine_lock_entry(engine_path))
    return stats
