"""PI0.5 compact712 INT8 lineage operators.

Ports the recorded AGX compact712 build tools (pi05_qdq_per_layer_20260627) into the
optimization SDK. The sealed compact712 headers pin the sequence-layout contract: the
968-token prefix is [cam0 256][cam1 256][pad-cam 256][prompt 200]; compaction keeps the
first `real_camera_tokens` slots of each real camera plus the full prompt and drops the
pad camera. Tier arithmetic: 2 * real_camera_tokens + 200 (64->328, 96->392, 128->456,
192->584, 256->712). The 256 tier matches the sealed `pi05_compact712_retained_indices`
initializer byte-for-byte (asserted by tests).

Sparse-delta contract (SDK-native, format `flat_indices_int_delta_and_float32_replacement_v2`):
the manifest lists every narrowed array as `{name: {dtype, shape, changed}}`; the delta
NPZ holds `{name}_indices` (int64) and `{name}_values` (array dtype) for every changed
array. Int arrays are updated by adding the delta with int8 wrap; float32 arrays are
updated by replacement.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FULL_PREFIX_TOKENS = 968
FULL_SUFFIX_TOKENS = 1018
PROMPT_TOKENS = 200
VISUAL_RETAIN_CHOICES = (64, 96, 128, 192, 256)
COMPACT_TOKENS = {64: 328, 96: 392, 128: 456, 192: 584, 256: 712}
SEQUENCE_AUDIT_VALUES = {424, 456, 474, 506, 512, 584, 634, 712, 762, 768, 968, 1018}
SPARSE_DELTA_FORMAT = "flat_indices_int_delta_and_float32_replacement_v2"
FFN_LAYER_COUNT = 17
FFN_ONNX_PREFIX = "model_paligemma_with_expert_paligemma_model_language_model_layers_{layer}_mlp"
PREFIX_EMBED_OUTPUT_AXES = {
    "prefix_embs": (1,),
    "prefix_pad_masks": (1,),
    "prefix_position_ids": (1,),
    "prefix_attention_mask_4d": (2, 3),
}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def retained_prefix_indices(real_camera_tokens: int) -> np.ndarray:
    if real_camera_tokens not in VISUAL_RETAIN_CHOICES:
        raise ValueError(f"unsupported real-camera token count: {real_camera_tokens}")
    visual = np.concatenate(
        (
            np.arange(0, real_camera_tokens, dtype=np.int64),
            np.arange(256, 256 + real_camera_tokens, dtype=np.int64),
        )
    )
    return np.concatenate((visual, np.arange(768, 968, dtype=np.int64)))


def _qdq_counts(graph: onnx.GraphProto) -> dict[str, int]:
    return {
        op_type: sum(node.op_type == op_type for node in graph.node)
        for op_type in ("QuantizeLinear", "DequantizeLinear")
    }


def _replace_shape_dimensions(
    graph: onnx.GraphProto,
    replacements: dict[int, int],
    *,
    include_inputs: bool = True,
    include_value_info: bool = True,
) -> list[dict[str, object]]:
    edits = []
    values = [*graph.output]
    if include_inputs:
        values.extend(graph.input)
    if include_value_info:
        values.extend(graph.value_info)
    for value in values:
        for axis, dimension in enumerate(value.type.tensor_type.shape.dim):
            if not dimension.HasField("dim_value") or dimension.dim_value not in replacements:
                continue
            old = dimension.dim_value
            dimension.dim_value = replacements[old]
            edits.append({"axis": axis, "name": value.name, "old": old, "new": replacements[old]})
    return edits


def _replace_constant_values(
    graph: onnx.GraphProto, replacements: dict[int, int]
) -> list[dict[str, object]]:
    edits = []
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            tensor = attribute.t
            if tensor.data_location == TensorProto.EXTERNAL:
                continue
            array = numpy_helper.to_array(tensor)
            if not np.issubdtype(array.dtype, np.integer):
                continue
            updated = array.copy()
            counts = {}
            for old, new in replacements.items():
                count = int(np.count_nonzero(updated == old))
                if count:
                    updated[updated == old] = new
                    counts[str(old)] = count
            if not counts:
                continue
            tensor.CopyFrom(numpy_helper.from_array(updated, tensor.name))
            edits.append({"name": node.name, "output": list(node.output), "counts": counts})
    return edits


def _remaining_constant_values(
    graph: onnx.GraphProto, targets: set[int]
) -> list[dict[str, object]]:
    rows = []
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            tensor = attribute.t
            if tensor.data_location == TensorProto.EXTERNAL:
                continue
            array = numpy_helper.to_array(tensor)
            if not np.issubdtype(array.dtype, np.integer):
                continue
            found = sorted(targets.intersection(int(value) for value in array.flat))
            if found:
                rows.append({"name": node.name, "values": found})
    return rows


def _rename_produced_tensor(graph: onnx.GraphProto, old: str, new: str) -> int:
    producers = 0
    for node in graph.node:
        for index, output in enumerate(node.output):
            if output == old:
                node.output[index] = new
                producers += 1
    if producers != 1:
        raise ValueError(f"expected one producer for {old}, found {producers}")
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
    return producers


def _compact_prefix_embed_outputs(
    graph: onnx.GraphProto, real_camera_tokens: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    retained = retained_prefix_indices(real_camera_tokens)
    compact_tokens = int(retained.size)
    indices_name = f"pi05_compact{compact_tokens}_retained_indices"
    if any(initializer.name == indices_name for initializer in graph.initializer):
        raise ValueError(f"initializer already exists: {indices_name}")
    graph.initializer.append(numpy_helper.from_array(retained, indices_name))
    if {output.name for output in graph.output} != set(PREFIX_EMBED_OUTPUT_AXES):
        raise ValueError("prefix_embed outputs differ from the deployment contract")
    edits = []
    for output_name, axes in PREFIX_EMBED_OUTPUT_AXES.items():
        source = f"{output_name}_full{FULL_PREFIX_TOKENS}"
        _rename_produced_tensor(graph, output_name, source)
        for axis_index, axis in enumerate(axes):
            target = (
                output_name if axis_index == len(axes) - 1 else f"{output_name}_compact_axis{axis}"
            )
            graph.node.append(
                helper.make_node(
                    "Gather",
                    [source, indices_name],
                    [target],
                    axis=axis,
                    name=f"pi05_compact{compact_tokens}_{output_name}_axis{axis}",
                )
            )
            source = target
        edits.append({"name": output_name, "axes": list(axes)})
    shape_edits = _replace_shape_dimensions(
        graph, {FULL_PREFIX_TOKENS: compact_tokens}, include_inputs=False, include_value_info=False
    )
    return edits, shape_edits


def compact_sequence(
    model: onnx.ModelProto, stage: str, real_camera_tokens: int
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    if stage not in ("prefix_embed", "prefix_lm", "suffix_step"):
        raise ValueError(f"unknown stage: {stage}")
    graph = model.graph
    before_qdq = _qdq_counts(graph)
    compact_tokens = COMPACT_TOKENS[real_camera_tokens]
    compact_suffix_tokens = compact_tokens + 50

    output_edits = []
    shape_edits = []
    constant_edits = []
    if stage == "prefix_embed":
        output_edits, shape_edits = _compact_prefix_embed_outputs(graph, real_camera_tokens)
        remaining_targets = set()
    elif stage == "prefix_lm":
        replacements = {FULL_PREFIX_TOKENS: compact_tokens}
        shape_edits = _replace_shape_dimensions(graph, replacements)
        constant_edits = _replace_constant_values(graph, replacements)
        remaining_targets = {FULL_PREFIX_TOKENS}
    else:
        replacements = {
            FULL_PREFIX_TOKENS: compact_tokens,
            FULL_SUFFIX_TOKENS: compact_suffix_tokens,
        }
        shape_edits = _replace_shape_dimensions(graph, replacements)
        constant_edits = _replace_constant_values(graph, replacements)
        remaining_targets = {FULL_PREFIX_TOKENS, FULL_SUFFIX_TOKENS}

    after_qdq = _qdq_counts(graph)
    if after_qdq != before_qdq:
        raise ValueError(f"QDQ node counts changed: {before_qdq} -> {after_qdq}")
    remaining = _remaining_constant_values(graph, remaining_targets)
    if remaining:
        raise ValueError(f"old sequence constants remain: {remaining}")
    return model, {
        "classification": "pi05_compact_sequence_onnx_rewrite",
        "compact_prefix_tokens": compact_tokens,
        "compact_suffix_tokens": compact_suffix_tokens,
        "real_camera_tokens": real_camera_tokens,
        "stage": stage,
        "output_edits": output_edits,
        "shape_edit_count": len(shape_edits),
        "shape_edits": shape_edits,
        "constant_edit_count": len(constant_edits),
        "constant_edits": constant_edits,
        "qdq_counts": after_qdq,
        "remaining_old_constants": remaining,
    }


def audit_sequence(models: list[Path]) -> dict[str, object]:
    def value_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
        dimensions = value.type.tensor_type.shape.dim
        return [
            dimension.dim_value
            if dimension.HasField("dim_value")
            else dimension.dim_param
            if dimension.HasField("dim_param")
            else None
            for dimension in dimensions
        ]

    def small_integer_values(tensor: onnx.TensorProto) -> list[int] | None:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            return None
        elements = int(np.prod(tensor.dims)) if tensor.dims else 1
        if elements > 64:
            return None
        array = numpy_helper.to_array(tensor)
        if not np.issubdtype(array.dtype, np.integer):
            return None
        return [int(value) for value in array.reshape(-1)]

    def audit_model(path: Path) -> dict[str, object]:
        model = onnx.load(str(path), load_external_data=False)
        graph = model.graph
        values = [*graph.input, *graph.output, *graph.value_info]
        shaped_values = []
        for value in values:
            shape = value_shape(value)
            if any(dimension in SEQUENCE_AUDIT_VALUES for dimension in shape):
                shaped_values.append({"name": value.name, "shape": shape})
        constants = []
        for initializer in graph.initializer:
            found = small_integer_values(initializer)
            if found is not None and SEQUENCE_AUDIT_VALUES.intersection(found):
                constants.append({"kind": "initializer", "name": initializer.name, "values": found})
        for node in graph.node:
            if node.op_type != "Constant":
                continue
            for attribute in node.attribute:
                if attribute.type != onnx.AttributeProto.TENSOR:
                    continue
                found = small_integer_values(attribute.t)
                if found is not None and SEQUENCE_AUDIT_VALUES.intersection(found):
                    constants.append(
                        {
                            "kind": "constant_node",
                            "name": node.name,
                            "output": list(node.output),
                            "values": found,
                        }
                    )
        return {
            "path": str(path),
            "graph_inputs": [
                {"name": value.name, "shape": value_shape(value)} for value in graph.input
            ],
            "graph_outputs": [
                {"name": value.name, "shape": value_shape(value)} for value in graph.output
            ],
            "sequence_constants": constants,
            "sequence_shaped_values": shaped_values,
        }

    return {
        "classification": "pi05_compact_sequence_onnx_audit",
        "target_values": sorted(SEQUENCE_AUDIT_VALUES),
        "models": [audit_model(path) for path in models],
    }


def rewrite_io_dimensions(
    payload: object, replacements: dict[int, int], expected_count: int
) -> tuple[object, int]:
    def walk(value: object) -> tuple[object, int]:
        if isinstance(value, list):
            items = []
            count = 0
            for item in value:
                rewritten, item_count = walk(item)
                items.append(rewritten)
                count += item_count
            return items, count
        if isinstance(value, dict):
            result = {}
            count = 0
            for key, item in value.items():
                rewritten, item_count = walk(item)
                result[key] = rewritten
                count += item_count
            return result, count
        if isinstance(value, int) and value in replacements:
            return replacements[value], 1
        return value, 0

    rewritten, count = walk(payload)
    if count != expected_count:
        raise ValueError(f"expected {expected_count} replacements, found {count}")
    return rewritten, count


def _replace_input(graph: onnx.GraphProto, old: str, new: str) -> int:
    count = 0
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
                count += 1
    for output in graph.output:
        if output.name == old:
            output.name = new
            count += 1
    return count


def _remove_unused_initializers(graph: onnx.GraphProto) -> None:
    used = {value for node in graph.node for value in node.input if value}
    used.update(output.name for output in graph.output)
    kept = [initializer for initializer in graph.initializer if initializer.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def edit_activation_qdq(
    model: onnx.ModelProto,
    remove_patterns: tuple[str, ...],
    set_scales: tuple[str, ...],
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    patterns = [re.compile(value) for value in remove_patterns]
    scale_specs = []
    for value in set_scales:
        pattern, scale = value.rsplit("=", 1)
        scale_specs.append((re.compile(pattern), float(scale)))
    graph = model.graph

    by_input: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.input:
            by_input[node.input[0]] = node

    remove_names: set[str] = set()
    removed: list[dict[str, object]] = []
    for node in list(graph.node):
        if node.op_type != "QuantizeLinear":
            continue
        if not any(pattern.search(node.name) for pattern in patterns):
            continue
        if len(node.input) < 3 or not node.output:
            continue
        dq = by_input.get(node.output[0])
        if dq is None or not dq.output:
            continue
        replacements = _replace_input(graph, dq.output[0], node.input[0])
        remove_names.add(node.name)
        remove_names.add(dq.name)
        removed.append(
            {
                "quantize": node.name,
                "dequantize": dq.name,
                "original_input": node.input[0],
                "removed_output": dq.output[0],
                "replacements": replacements,
                "scale": node.input[1],
                "zero": node.input[2],
            }
        )
    if remove_names:
        kept_nodes = [node for node in graph.node if node.name not in remove_names]
        del graph.node[:]
        graph.node.extend(kept_nodes)
        _remove_unused_initializers(graph)

    initializers = {initializer.name: initializer for initializer in graph.initializer}
    scale_edits = []
    for node in graph.node:
        if node.op_type != "QuantizeLinear" or len(node.input) < 2:
            continue
        for pattern, value in scale_specs:
            if not pattern.search(node.name):
                continue
            scale_name = node.input[1]
            initializer = initializers.get(scale_name)
            if initializer is None:
                continue
            old = numpy_helper.to_array(initializer)
            if old.size != 1:
                continue
            initializer.CopyFrom(
                numpy_helper.from_array(np.asarray(value, dtype=np.float32), scale_name)
            )
            scale_edits.append(
                {
                    "quantize": node.name,
                    "scale": scale_name,
                    "old": float(old.reshape(-1)[0]),
                    "new": value,
                }
            )
    return model, {
        "classification": "pi05_activation_qdq_edit",
        "removed_qdq": removed,
        "removed_count": len(removed),
        "scale_edits": scale_edits,
        "scale_edit_count": len(scale_edits),
    }


def read_keep_indices(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text())
    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path}: missing keep-index layers")
    result = {}
    for raw_layer, row in layers.items():
        layer = int(raw_layer)
        keep = np.asarray(row["keep_indices"], dtype=np.int64)
        if keep.ndim != 1 or not keep.size or np.unique(keep).size != keep.size:
            raise ValueError(f"{path}: layer{layer} keep indices are invalid")
        result[layer] = keep
    return result


def subset_positions(base_keep: np.ndarray, target_keep: np.ndarray) -> np.ndarray:
    position_by_channel = {int(channel): index for index, channel in enumerate(base_keep)}
    missing = [int(channel) for channel in target_keep if int(channel) not in position_by_channel]
    if missing:
        raise ValueError(f"target keep indices are absent from base keep indices: {missing[:8]}")
    return np.asarray(
        [position_by_channel[int(channel)] for channel in target_keep], dtype=np.int64
    )


def narrow_projection(
    weight: np.ndarray, scale: np.ndarray, zero: np.ndarray, positions: np.ndarray, projection: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if weight.ndim != 2 or scale.ndim != 1 or zero.shape != scale.shape:
        raise ValueError("QDQ weight/scale/zero metadata mismatch")
    if projection in {"gate", "up"}:
        if weight.shape[1] != scale.size:
            raise ValueError(f"{projection} output axis differs from QDQ scale")
        return (
            np.ascontiguousarray(weight[:, positions]),
            np.ascontiguousarray(scale[positions]),
            np.ascontiguousarray(zero[positions]),
        )
    if projection == "down":
        if weight.shape[0] <= int(positions.max(initial=-1)):
            raise ValueError("down input axis is smaller than keep-index positions")
        return (
            np.ascontiguousarray(weight[positions, :]),
            np.ascontiguousarray(scale),
            np.ascontiguousarray(zero),
        )
    raise ValueError(f"unknown projection: {projection}")


def ffn_onnx_names(layer: int) -> dict[str, str]:
    base = FFN_ONNX_PREFIX.format(layer=layer)
    return {
        f"{projection}_{suffix}": f"{base}_{projection}_proj_weight_qdq_weight_{onnx_suffix}"
        for projection in ("gate", "up", "down")
        for suffix, onnx_suffix in (("weight", "int8"), ("scale", "scale"), ("zero", "zero"))
    }


def _parse_layers(text: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in text.split(",") if item)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be a nonempty unique comma-separated list")
    return layers


def narrow_ffn(
    model: onnx.ModelProto,
    base_keep_indices: Path,
    target_keep_indices: Path,
    layers: str = ",".join(str(layer) for layer in range(FFN_LAYER_COUNT)),
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    layer_list = _parse_layers(layers)
    base_keep = read_keep_indices(base_keep_indices)
    target_keep = read_keep_indices(target_keep_indices)
    if set(base_keep) != set(layer_list) or set(target_keep) != set(layer_list):
        raise ValueError("base and target keep manifests must exactly match declared layers")
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    report_layers = {}
    for layer in layer_list:
        positions = subset_positions(base_keep[layer], target_keep[layer])
        names = ffn_onnx_names(layer)
        missing = sorted(set(names.values()) - set(initializers))
        if missing:
            raise KeyError(f"layer{layer} is missing QDQ initializers: {missing}")
        projections = {}
        for projection in ("gate", "up", "down"):
            weight, scale, zero = narrow_projection(
                numpy_helper.to_array(initializers[names[f"{projection}_weight"]]),
                numpy_helper.to_array(initializers[names[f"{projection}_scale"]]),
                numpy_helper.to_array(initializers[names[f"{projection}_zero"]]),
                positions,
                projection,
            )
            for name, value in (
                (names[f"{projection}_weight"], weight),
                (names[f"{projection}_scale"], scale),
                (names[f"{projection}_zero"], zero),
            ):
                initializers[name].CopyFrom(numpy_helper.from_array(value, name))
            projections[projection] = {
                "weight_shape": list(weight.shape),
                "scale_shape": list(scale.shape),
                "zero_shape": list(zero.shape),
            }
        report_layers[str(layer)] = {
            "base_kept": int(base_keep[layer].size),
            "target_kept": int(target_keep[layer].size),
            "removed": int(base_keep[layer].size - target_keep[layer].size),
            "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "projections": projections,
        }
    return model, {
        "classification": "pi05_narrow_ffn_qdq",
        "base_keep_indices": str(base_keep_indices),
        "base_keep_indices_sha256": file_sha256(base_keep_indices),
        "target_keep_indices": str(target_keep_indices),
        "target_keep_indices_sha256": file_sha256(target_keep_indices),
        "layers": report_layers,
    }


def canonical_sha256(arrays: dict[str, np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(str(array.dtype).encode())
        hasher.update(b"\0")
        hasher.update(",".join(str(size) for size in array.shape).encode())
        hasher.update(b"\0")
        hasher.update(np.ascontiguousarray(array).tobytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def apply_sparse_delta(
    model: onnx.ModelProto,
    delta_path: Path,
    manifest_path: Path,
    base_keep_indices: Path,
    target_keep_indices: Path,
) -> tuple[onnx.ModelProto, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != SPARSE_DELTA_FORMAT:
        raise ValueError("sparse delta must use the scale-aware v2 format")
    if file_sha256(delta_path) != manifest["delta_file_sha256"]:
        raise ValueError("sparse delta file SHA256 differs from manifest")
    base_keep = read_keep_indices(base_keep_indices)
    target_keep = read_keep_indices(target_keep_indices)
    if set(base_keep) != set(range(FFN_LAYER_COUNT)) or set(target_keep) != set(
        range(FFN_LAYER_COUNT)
    ):
        raise ValueError("keep manifests must contain prefix FFN layers0..16")

    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    arrays = {}
    onnx_by_array = {}
    report_layers = {}
    for layer in range(FFN_LAYER_COUNT):
        positions = subset_positions(base_keep[layer], target_keep[layer])
        names = ffn_onnx_names(layer)
        missing = sorted(set(names.values()) - set(initializers))
        if missing:
            raise KeyError(f"layer{layer} is missing QDQ initializers: {missing}")
        arrays[f"layer{layer}_keep"] = target_keep[layer].astype(np.int32)
        projections = {}
        for projection in ("gate", "up", "down"):
            weight, scale, zero = narrow_projection(
                numpy_helper.to_array(initializers[names[f"{projection}_weight"]]),
                numpy_helper.to_array(initializers[names[f"{projection}_scale"]]),
                numpy_helper.to_array(initializers[names[f"{projection}_zero"]]),
                positions,
                projection,
            )
            for suffix, onnx_name, value in (
                ("w", names[f"{projection}_weight"], weight),
                ("scale", names[f"{projection}_scale"], scale),
                ("zero", names[f"{projection}_zero"], zero),
            ):
                package_name = f"layer{layer}_{projection}_{suffix}"
                arrays[package_name] = value
                onnx_by_array[package_name] = onnx_name
            projections[projection] = {
                "weight_shape": list(weight.shape),
                "scale_shape": list(scale.shape),
                "zero_shape": list(zero.shape),
            }
        report_layers[str(layer)] = {
            "base_kept": int(base_keep[layer].size),
            "target_kept": int(target_keep[layer].size),
            "removed": int(base_keep[layer].size - target_keep[layer].size),
            "projections": projections,
        }

    if set(arrays) != set(manifest["entries"]):
        raise ValueError("narrowed ONNX QDQ arrays differ from sparse delta manifest")
    base_canonical = canonical_sha256(arrays)
    if base_canonical != manifest["base_canonical_sha256"]:
        raise ValueError("narrowed ONNX canonical SHA256 differs from sparse delta")

    with np.load(delta_path, allow_pickle=False) as pack:
        delta = {name: pack[name] for name in pack.files}
    target_arrays = {}
    for name, value in arrays.items():
        entry = manifest["entries"][name]
        changed = int(entry["changed"])
        if not changed:
            target_arrays[name] = value
            continue
        indices = np.asarray(delta[f"{name}_indices"], dtype=np.int64)
        updates = np.asarray(delta[f"{name}_values"])
        if (
            indices.ndim != 1
            or updates.ndim != 1
            or indices.size != changed
            or updates.size != changed
        ):
            raise ValueError(
                f"{name}: sparse delta shape mismatch ({indices.shape}, {updates.shape}) vs changed={changed}"
            )
        target = value.copy()
        flat = target.reshape(-1)
        if value.dtype == np.float32:
            flat[indices] = updates.astype(np.float32)
        elif value.dtype == np.int8:
            flat[indices] = (flat[indices].astype(np.int16) + updates.astype(np.int16)).astype(
                np.int8
            )
        else:
            raise ValueError(f"{name}: unsupported sparse delta dtype {value.dtype}")
        target_arrays[name] = target
    target_canonical = canonical_sha256(target_arrays)
    if target_canonical != manifest["target_canonical_sha256"]:
        raise ValueError("target ONNX canonical SHA256 differs from sparse delta")
    for package_name, onnx_name in onnx_by_array.items():
        initializers[onnx_name].CopyFrom(
            numpy_helper.from_array(target_arrays[package_name], onnx_name)
        )

    return model, {
        "classification": "pi05_apply_sparse_ffn_qdq_delta",
        "delta": str(delta_path),
        "delta_sha256": file_sha256(delta_path),
        "manifest": str(manifest_path),
        "base_keep_indices": str(base_keep_indices),
        "target_keep_indices": str(target_keep_indices),
        "base_canonical_sha256": base_canonical,
        "target_canonical_sha256": target_canonical,
        "changed_arrays": manifest["changed_arrays"],
        "changed_values": manifest["changed_values"],
        "layers": report_layers,
        "qdq_counts": _qdq_counts(graph),
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
    }
