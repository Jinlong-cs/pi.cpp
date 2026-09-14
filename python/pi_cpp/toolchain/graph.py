"""Pure-ONNX graph transforms for the picpp build pipeline.

These are the ported board scripts as tool behavior, with fail-fast
consistency checks instead of silent best effort:

- ``cast_cache_abi``: the cache ABI bf16 -> fp16 transform (cast_caches_fp16.py).
  Every bf16 value is exactly representable in fp16 (same 8-bit exponent,
  more mantissa bits), so the transform preserves values bit-for-bit. The
  TRT INT8 calibrator chokes on bf16 network inputs; both stages then speak
  fp16 with the same 2-byte footprint.
- ``insert_qdq``: the calibrator-free QDQ transform (build_suffix_qdq.py) —
  symmetric per-tensor int8 QuantizeLinear/DequantizeLinear pairs around the
  selected MatMuls, per-channel weight scales (max_abs(W)/127), activation
  scales supplied by ``picpp calibrate``. Missing scales fail fast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from onnx import ModelProto, TensorProto, helper, numpy_helper

_CACHE_PREFIX = "cache_"
_EXPECTED_CACHES = 36


def topo_sort(nodes: list[Any]) -> list[Any]:
    """Deterministic topological sort; raises on cycles (fail-fast)."""
    produced = {o: node for node in nodes for o in node.output}
    placed: set[int] = set()
    ordered: list[Any] = []
    pending = list(nodes)
    while pending:
        progress = False
        for node in list(pending):
            if all(i not in produced or id(produced[i]) in placed for i in node.input):
                ordered.append(node)
                placed.add(id(node))
                pending.remove(node)
                progress = True
        if not progress:
            raise ValueError("graph contains a cycle; refusing to emit a broken model")
    return ordered


def cast_cache_abi(model: ModelProto, *, stage: str) -> ModelProto:
    """Rewrite the cache ABI bf16 -> fp16 for the prefix or suffix stage."""
    if stage not in ("prefix", "suffix"):
        raise ValueError(f"stage must be prefix or suffix, got {stage!r}")
    graph = model.graph
    if stage == "prefix":
        cache_tensors = [out for out in graph.output if out.name.startswith(_CACHE_PREFIX)]
    else:
        cache_tensors = [inp for inp in graph.input if inp.name.startswith(_CACHE_PREFIX)]
    if len(cache_tensors) != _EXPECTED_CACHES:
        raise ValueError(f"{stage} must declare {_EXPECTED_CACHES} cache tensors, found {len(cache_tensors)}")
    for tensor in cache_tensors:
        if tensor.type.tensor_type.elem_type != TensorProto.BFLOAT16:
            raise ValueError(f"cache tensor {tensor.name} must be bf16, got {tensor.type.tensor_type.elem_type}")
    for tensor in cache_tensors:
        internal = f"{tensor.name}_bf16"
        if stage == "prefix":
            for node in graph.node:
                for i, output in enumerate(node.output):
                    if output == tensor.name:
                        node.output[i] = internal
            graph.node.append(
                helper.make_node("Cast", [internal], [tensor.name], to=TensorProto.FLOAT16, name=f"cast_out_{tensor.name}")
            )
        else:
            for node in graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == tensor.name:
                        node.input[i] = internal
            graph.node.append(
                helper.make_node("Cast", [tensor.name], [internal], to=TensorProto.BFLOAT16, name=f"cast_in_{tensor.name}")
            )
        tensor.type.tensor_type.elem_type = TensorProto.FLOAT16
    ordered = topo_sort(list(graph.node))
    graph.ClearField("node")
    graph.node.extend(ordered)
    return model


def _is_mlp_matmul(node: Any, name_contains: Sequence[str], name_prefixes: Sequence[str]) -> bool:
    if node.op_type != "MatMul":
        return False
    name = node.name or ""
    return any(marker in name for marker in name_contains) or any(name.startswith(prefix) for prefix in name_prefixes)


def resolve_weight(graph: Any, name: str) -> np.ndarray:
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    raise ValueError(f"cannot resolve weight {name}")


def _add_const(graph: Any, name: str, value: np.ndarray) -> str:
    graph.initializer.append(numpy_helper.from_array(value, name=name))
    return name


def insert_qdq(
    model: ModelProto,
    *,
    act_scales: Mapping[str, float],
    name_contains: Sequence[str] = ("/mlp",),
    name_prefixes: Sequence[str] = ("/time_mlp",),
) -> tuple[ModelProto, int]:
    """Wrap every selected MatMul in symmetric int8 Q/DQ pairs.

    Weights are quantized per-channel (axis 0) from the resolved constant
    (walking back through a Transpose); activations use the supplied scales
    (``picpp calibrate`` output). Returns the rewritten model + the number of
    quantized matmuls. Raises when an activation scale is missing.
    """
    graph = model.graph
    matmuls = [node for node in graph.node if _is_mlp_matmul(node, name_contains, name_prefixes)]
    missing = [node.input[0] for node in matmuls if node.input[0] not in act_scales]
    if missing:
        raise ValueError(f"missing activation scales for {missing[:3]}")
    counter = iter(range(10**6))
    for matmul in matmuls:
        weight_in = matmul.input[1]
        weight_raw = None
        for node in graph.node:
            if weight_in in node.output:
                if node.op_type == "Transpose" and len(node.input) == 1:
                    weight_raw = resolve_weight(graph, node.input[0])
                break
        if weight_raw is None:
            raise ValueError(f"weight source for {weight_in} not found")
        per_row = np.maximum(np.abs(weight_raw).max(axis=-1).astype(np.float32), 1e-6) / 127.0
        sidx = next(counter)
        s_name = f"/q/{matmul.name}_s_w_{sidx}"
        z_name = f"/q/{matmul.name}_z_w_{sidx}"
        q_name = f"/q/{matmul.name}_q_w_{sidx}"
        dq_name = f"/q/{matmul.name}_dq_w_{sidx}"
        _add_const(graph, s_name, per_row)
        _add_const(graph, z_name, np.zeros(len(per_row), np.int8))
        graph.node.append(helper.make_node("QuantizeLinear", [weight_in, s_name, z_name], [q_name], name=f"/q/{matmul.name}_q_w_{sidx}", axis=1))
        graph.node.append(helper.make_node("DequantizeLinear", [q_name, s_name, z_name], [dq_name], name=f"/q/{matmul.name}_dq_w_{sidx}", axis=1))
        matmul.input[1] = dq_name
    ordered = topo_sort(list(graph.node))
    graph.ClearField("node")
    graph.node.extend(ordered)
    return model, len(matmuls)


def count_mlp_matmuls(model: ModelProto, *, name_contains: Sequence[str] = ("/mlp",), name_prefixes: Sequence[str] = ("/time_mlp",)) -> int:
    return sum(1 for node in model.graph.node if _is_mlp_matmul(node, name_contains, name_prefixes))


__all__ = [
    "cast_cache_abi",
    "count_mlp_matmuls",
    "insert_qdq",
    "resolve_weight",
    "topo_sort",
]
