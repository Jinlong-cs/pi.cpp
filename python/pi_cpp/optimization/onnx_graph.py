"""ONNX graph inspection and constant-weight linear detection."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnx


@dataclass(frozen=True)
class LinearCandidate:
    node_index: int
    node_name: str
    op_type: str
    activation_input: str
    weight_input: str
    source_initializer: str
    weight_transform: tuple[int, ...] | None
    channel_axis: int
    weight_precision: str


def attribute_int(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def attribute_ints(node: onnx.NodeProto, name: str) -> tuple[int, ...] | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return tuple(int(value) for value in attribute.ints)
    return None


def node_by_output(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in graph.node for output in node.output if output}


def initializer_map(graph: onnx.GraphProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def find_linear_candidates(model: onnx.ModelProto) -> list[LinearCandidate]:
    graph = model.graph
    producers = node_by_output(graph)
    initializers = initializer_map(graph)
    candidates: list[LinearCandidate] = []
    for index, node in enumerate(graph.node):
        if node.op_type == "MatMul" and len(node.input) >= 2:
            weight_input = node.input[1]
            source = weight_input if weight_input in initializers else None
            transform: tuple[int, ...] | None = None
            precision = "fp"
            producer = producers.get(weight_input)
            if (
                source is None
                and producer is not None
                and producer.op_type == "DequantizeLinear"
                and producer.input[0] in initializers
            ):
                source = producer.input[0]
                precision = "qdq"
            if (
                source is None
                and producer is not None
                and producer.op_type == "Cast"
                and len(producer.input) == 1
            ):
                producer = producers.get(producer.input[0])
            if (
                source is None
                and producer is not None
                and producer.op_type == "Transpose"
                and len(producer.input) == 1
            ):
                source = producer.input[0] if producer.input[0] in initializers else None
                transform = attribute_ints(producer, "perm")
                if source is not None and transform is None:
                    rank = len(initializers[source].dims)
                    transform = tuple(reversed(range(rank)))
            if source is not None:
                candidates.append(
                    LinearCandidate(
                        index,
                        node.name,
                        node.op_type,
                        node.input[0],
                        weight_input,
                        source,
                        transform,
                        1,
                        precision,
                    )
                )
        elif node.op_type == "Gemm" and len(node.input) >= 2:
            producer = producers.get(node.input[1])
            source = node.input[1] if node.input[1] in initializers else None
            precision = "fp"
            if (
                source is None
                and producer is not None
                and producer.op_type == "DequantizeLinear"
                and producer.input[0] in initializers
            ):
                source = producer.input[0]
                precision = "qdq"
            if source is None:
                continue
            trans_b = attribute_int(node, "transB")
            candidates.append(
                LinearCandidate(
                    index,
                    node.name,
                    node.op_type,
                    node.input[0],
                    node.input[1],
                    source,
                    None,
                    0 if trans_b else 1,
                    precision,
                )
            )
    return candidates


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def tensor_dtype(value: onnx.ValueInfoProto) -> str:
    return onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(path: Path, model: onnx.ModelProto) -> list[dict[str, Any]]:
    locations = {path.name}
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    result: list[dict[str, Any]] = []
    for location in sorted(locations):
        artifact = path if location == path.name else path.parent / location
        result.append(
            {
                "path": str(artifact),
                "exists": artifact.is_file(),
                "bytes": artifact.stat().st_size if artifact.is_file() else None,
                "sha256": sha256_file(artifact) if artifact.is_file() else None,
            }
        )
    return result


def inspect_model(path: Path) -> dict[str, Any]:
    model = onnx.load_model(path, load_external_data=False)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    op_counts = Counter(node.op_type for node in graph.node)
    candidates = find_linear_candidates(model)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    qdq_pairs = sum(
        1
        for node in graph.node
        if node.op_type == "QuantizeLinear"
        and len(node.output) == 1
        and any(
            consumer.op_type == "DequantizeLinear" for consumer in consumers.get(node.output[0], [])
        )
    )
    return {
        "path": str(path),
        "artifacts": artifact_files(path, model),
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": opset.domain, "version": int(opset.version)} for opset in model.opset_import
        ],
        "producer": {"name": model.producer_name, "version": model.producer_version},
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "inputs": [
            {"name": value.name, "dtype": tensor_dtype(value), "shape": tensor_shape(value)}
            for value in graph.input
            if value.name not in initializer_names
        ],
        "outputs": [
            {"name": value.name, "dtype": tensor_dtype(value), "shape": tensor_shape(value)}
            for value in graph.output
        ],
        "op_counts": dict(sorted(op_counts.items())),
        "qdq_pair_count": qdq_pairs,
        "constant_weight_linear_count": len(candidates),
        "constant_weight_linear_op_counts": dict(
            sorted(Counter(candidate.op_type for candidate in candidates).items())
        ),
        "constant_weight_linear_precision_counts": dict(
            sorted(Counter(candidate.weight_precision for candidate in candidates).items())
        ),
        "constant_weight_linear_candidates": [asdict(candidate) for candidate in candidates],
    }
