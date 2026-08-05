"""Selective static QDQ insertion and precision-guard regeneration."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from pi_cpp.optimization.onnx_graph import (
    LinearCandidate,
    find_linear_candidates,
    initializer_map,
)


@dataclass(frozen=True)
class QDQStageConfig:
    enabled: bool
    activation_scale: float
    per_channel_weights: bool
    include_name_patterns: tuple[str, ...]
    exclude_name_patterns: tuple[str, ...]
    keep_fp16_name_patterns: tuple[str, ...]
    expected_quantized_nodes: int | None = None
    expected_guarded_nodes: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QDQStageConfig:
        return cls(
            enabled=bool(value.get("enabled", True)),
            activation_scale=float(value.get("activation_scale", 0.1)),
            per_channel_weights=bool(value.get("per_channel_weights", True)),
            include_name_patterns=tuple(value.get("include_name_patterns", [".*"])),
            exclude_name_patterns=tuple(value.get("exclude_name_patterns", [])),
            keep_fp16_name_patterns=tuple(value.get("keep_fp16_name_patterns", [])),
            expected_quantized_nodes=int(value["expected_quantized_nodes"])
            if value.get("expected_quantized_nodes") is not None
            else None,
            expected_guarded_nodes=int(value["expected_guarded_nodes"])
            if value.get("expected_guarded_nodes") is not None
            else None,
        )

    def with_guards(self, patterns: tuple[str, ...]) -> QDQStageConfig:
        return replace(
            self,
            keep_fp16_name_patterns=self.keep_fp16_name_patterns + patterns,
            expected_quantized_nodes=None,
            expected_guarded_nodes=None,
        )


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _activation_qdq(
    value_name: str, scale: float
) -> tuple[list[onnx.NodeProto], str, list[onnx.TensorProto]]:
    base = _sanitize(value_name)
    scale_name = f"{base}_qdq_act_scale"
    zero_name = f"{base}_qdq_act_zero"
    quantized_name = f"{base}_qdq_act_quant"
    dequantized_name = f"{base}_qdq_act_dequant"
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [value_name, scale_name, zero_name],
            [quantized_name],
            name=f"{base}_Q",
        ),
        helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name, zero_name],
            [dequantized_name],
            name=f"{base}_DQ",
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray(scale, dtype=np.float32), scale_name),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), zero_name),
    ]
    return nodes, dequantized_name, initializers


def _weight_array(
    candidate: LinearCandidate, initializers: dict[str, onnx.TensorProto]
) -> np.ndarray:
    value = numpy_helper.to_array(initializers[candidate.source_initializer]).astype(np.float32)
    if candidate.weight_transform is not None:
        value = np.transpose(value, axes=candidate.weight_transform)
    if value.ndim != 2:
        raise ValueError(f"{candidate.node_name}: expected rank-2 linear weight, got {value.shape}")
    return value


def _weight_dq(
    candidate: LinearCandidate, value: np.ndarray, per_channel: bool
) -> tuple[list[onnx.NodeProto], str, list[onnx.TensorProto]]:
    base = f"{_sanitize(candidate.source_initializer)}_qdq_weight"
    quantized_name = f"{base}_int8"
    scale_name = f"{base}_scale"
    zero_name = f"{base}_zero"
    dequantized_name = f"{base}_dequant"
    if per_channel:
        reduce_axes = tuple(axis for axis in range(value.ndim) if axis != candidate.channel_axis)
        absmax = np.max(np.abs(value), axis=reduce_axes).astype(np.float32)
        scale = np.maximum(absmax / 127.0, np.float32(1.0e-8))
        broadcast_shape = [1] * value.ndim
        broadcast_shape[candidate.channel_axis] = scale.shape[0]
        quantized = np.clip(np.rint(value / scale.reshape(broadcast_shape)), -127, 127).astype(
            np.int8
        )
        zero = np.zeros(scale.shape, dtype=np.int8)
        node = helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name, zero_name],
            [dequantized_name],
            name=f"{base}_DQ",
            axis=candidate.channel_axis,
        )
    else:
        absmax = float(np.max(np.abs(value)))
        scale = np.asarray(max(absmax / 127.0, 1.0e-8), dtype=np.float32)
        quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
        zero = np.asarray(0, dtype=np.int8)
        node = helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name, zero_name],
            [dequantized_name],
            name=f"{base}_DQ",
        )
    initializers = [
        numpy_helper.from_array(quantized, quantized_name),
        numpy_helper.from_array(scale, scale_name),
        numpy_helper.from_array(zero, zero_name),
    ]
    return [node], dequantized_name, initializers


def prune_dead_graph(graph: onnx.GraphProto) -> None:
    producers = {
        output: index for index, node in enumerate(graph.node) for output in node.output if output
    }
    needed_values = {output.name for output in graph.output}
    needed_nodes: set[int] = set()
    stack = list(needed_values)
    while stack:
        value = stack.pop()
        index = producers.get(value)
        if index is None or index in needed_nodes:
            continue
        needed_nodes.add(index)
        for input_name in graph.node[index].input:
            if input_name and input_name not in needed_values:
                needed_values.add(input_name)
                stack.append(input_name)
    nodes = [node for index, node in enumerate(graph.node) if index in needed_nodes]
    initializers = [
        initializer for initializer in graph.initializer if initializer.name in needed_values
    ]
    value_info = [value for value in graph.value_info if value.name in needed_values]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    del graph.value_info[:]
    graph.value_info.extend(value_info)


def quantize_model(
    model: onnx.ModelProto, config: QDQStageConfig
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    if not config.enabled:
        raise ValueError("QDQ is disabled for this stage")
    result = copy.deepcopy(model)
    graph = result.graph
    candidates = [
        candidate
        for candidate in find_linear_candidates(result)
        if candidate.weight_precision == "fp"
    ]
    candidate_by_index = {candidate.node_index: candidate for candidate in candidates}
    initializers = initializer_map(graph)
    activation_outputs: dict[str, str] = {}
    activation_initializers: dict[str, onnx.TensorProto] = {}
    weight_outputs: dict[tuple[str, tuple[int, ...] | None, int], str] = {}
    weight_initializers: dict[str, onnx.TensorProto] = {}
    selected: list[str] = []
    guarded: list[str] = []
    excluded: list[str] = []
    new_nodes: list[onnx.NodeProto] = []

    for index, node in enumerate(graph.node):
        candidate = candidate_by_index.get(index)
        if candidate is None:
            new_nodes.append(node)
            continue
        name = candidate.node_name
        if config.include_name_patterns and not _matches(name, config.include_name_patterns):
            excluded.append(name)
            new_nodes.append(node)
            continue
        if _matches(name, config.exclude_name_patterns):
            excluded.append(name)
            new_nodes.append(node)
            continue
        if _matches(name, config.keep_fp16_name_patterns):
            guarded.append(name)
            new_nodes.append(node)
            continue

        if candidate.activation_input not in activation_outputs:
            nodes, output, values = _activation_qdq(
                candidate.activation_input, config.activation_scale
            )
            activation_outputs[candidate.activation_input] = output
            new_nodes.extend(nodes)
            activation_initializers.update({value.name: value for value in values})
        key = (candidate.source_initializer, candidate.weight_transform, candidate.channel_axis)
        if key not in weight_outputs:
            nodes, output, values = _weight_dq(
                candidate, _weight_array(candidate, initializers), config.per_channel_weights
            )
            weight_outputs[key] = output
            new_nodes.extend(nodes)
            for value in values:
                if value.name in weight_initializers:
                    raise ValueError(f"duplicate generated weight initializer: {value.name}")
                weight_initializers[value.name] = value
        node.input[0] = activation_outputs[candidate.activation_input]
        node.input[1] = weight_outputs[key]
        selected.append(name)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(activation_initializers.values())
    graph.initializer.extend(weight_initializers.values())
    before = {
        "nodes": len(graph.node),
        "initializers": len(graph.initializer),
        "op_counts": dict(Counter(node.op_type for node in graph.node)),
    }
    prune_dead_graph(graph)
    after = {
        "nodes": len(graph.node),
        "initializers": len(graph.initializer),
        "op_counts": dict(Counter(node.op_type for node in graph.node)),
    }
    if (
        config.expected_quantized_nodes is not None
        and len(selected) != config.expected_quantized_nodes
    ):
        raise ValueError(
            f"expected {config.expected_quantized_nodes} quantized nodes, found {len(selected)}"
        )
    if config.expected_guarded_nodes is not None and len(guarded) != config.expected_guarded_nodes:
        raise ValueError(
            f"expected {config.expected_guarded_nodes} guarded nodes, found {len(guarded)}"
        )
    return result, {
        "candidate_count": len(candidates),
        "quantized_node_count": len(selected),
        "quantized_nodes": selected,
        "guarded_node_count": len(guarded),
        "guarded_nodes": guarded,
        "excluded_node_count": len(excluded),
        "excluded_nodes": excluded,
        "unique_activation_qdq_tensors": len(activation_outputs),
        "unique_weight_dq_tensors": len(weight_outputs),
        "activation_scale": config.activation_scale,
        "per_channel_weights": config.per_channel_weights,
        "before_prune": before,
        "after_prune": after,
    }


def save_model(
    model: onnx.ModelProto, output: Path, external_data_file: str | None, check: bool = True
) -> None:
    if output.exists():
        raise FileExistsError(output)
    if external_data_file is not None and Path(external_data_file).name != external_data_file:
        raise ValueError("external_data_file must be a filename relative to the output ONNX")
    output.parent.mkdir(parents=True, exist_ok=True)
    if external_data_file is None:
        onnx.save_model(model, output)
    else:
        external_path = output.parent / external_data_file
        if external_path.exists():
            raise FileExistsError(external_path)
        onnx.save_model(
            model,
            output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_file,
            size_threshold=1024,
            convert_attribute=False,
        )
    if check:
        onnx.checker.check_model(str(output))


def strip_qdq_pairs(
    model: onnx.ModelProto, name_patterns: tuple[str, ...] = ()
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    result = copy.deepcopy(model)
    graph = result.graph
    consumers: dict[str, list[tuple[int, onnx.NodeProto]]] = defaultdict(list)
    for index, node in enumerate(graph.node):
        for input_name in node.input:
            if input_name:
                consumers[input_name].append((index, node))
    removed: set[int] = set()
    replacements: dict[str, str] = {}
    pairs: list[dict[str, str]] = []
    for index, quantize in enumerate(graph.node):
        if (
            quantize.op_type != "QuantizeLinear"
            or len(quantize.input) < 1
            or len(quantize.output) != 1
        ):
            continue
        uses = consumers.get(quantize.output[0], [])
        if not uses or not all(
            node.op_type == "DequantizeLinear" and node.input[0] == quantize.output[0]
            for _, node in uses
        ):
            continue
        names = [quantize.name] + [node.name for _, node in uses]
        if name_patterns and not any(_matches(name, name_patterns) for name in names):
            continue
        removed.add(index)
        for consumer_index, dequantize in uses:
            if len(dequantize.output) != 1:
                raise ValueError(f"{dequantize.name}: expected one DequantizeLinear output")
            removed.add(consumer_index)
            replacements[dequantize.output[0]] = quantize.input[0]
            pairs.append({"quantize": quantize.name, "dequantize": dequantize.name})

    def resolve(value: str) -> str:
        seen: set[str] = set()
        while value in replacements:
            if value in seen:
                raise RuntimeError(f"replacement cycle at {value}")
            seen.add(value)
            value = replacements[value]
        return value

    nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(graph.node):
        if index in removed:
            continue
        clone = copy.deepcopy(node)
        for input_index, input_name in enumerate(clone.input):
            if input_name:
                clone.input[input_index] = resolve(input_name)
        nodes.append(clone)
    del graph.node[:]
    graph.node.extend(nodes)
    for output in graph.output:
        output.name = resolve(output.name)
    for value in graph.value_info:
        value.name = resolve(value.name)
    prune_dead_graph(graph)
    return result, {
        "stripped_pair_count": len(pairs),
        "pairs": pairs,
        "remaining_nodes": len(graph.node),
    }
