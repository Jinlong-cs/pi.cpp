"""Unit tests for the PI0.5 compact712 optimization operators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from pi_cpp.optimization.pi05 import (
    COMPACT_TOKENS,
    apply_sparse_delta,
    audit_sequence,
    canonical_sha256,
    compact_sequence,
    edit_activation_qdq,
    ffn_onnx_names,
    narrow_ffn,
    retained_prefix_indices,
    rewrite_io_dimensions,
)
from pi_cpp.optimization.qdq import save_model

SEALED_RETAINED_SHA256 = "f5a1faefc3fceb3357ff482ddbba80b634533a7d9a3c96c64ea682aea0edcb8d"


def _make_qdq(prefix: str, input_name: str, output_name: str) -> list[onnx.NodeProto]:
    return [
        helper.make_node(
            "QuantizeLinear",
            [input_name, f"{prefix}_scale", f"{prefix}_zero"],
            [f"{prefix}_quant"],
            name=f"{prefix}_Q",
        ),
        helper.make_node(
            "DequantizeLinear",
            [f"{prefix}_quant", f"{prefix}_scale", f"{prefix}_zero"],
            [output_name],
            name=f"{prefix}_DQ",
        ),
    ]


def test_retained_prefix_indices_tiers() -> None:
    for tier, expected in COMPACT_TOKENS.items():
        indices = retained_prefix_indices(tier)
        assert indices.dtype == np.int64
        assert indices.size == expected
        assert np.array_equal(indices[:tier], np.arange(0, tier))
        assert np.array_equal(indices[tier : 2 * tier], np.arange(256, 256 + tier))
        assert np.array_equal(indices[2 * tier :], np.arange(768, 968))
    indices = retained_prefix_indices(256)
    assert hashlib.sha256(indices.tobytes()).hexdigest() == SEALED_RETAINED_SHA256


def test_retained_prefix_indices_rejects_unknown_tier() -> None:
    try:
        retained_prefix_indices(128)
    except ValueError as error:
        assert "unsupported" in str(error)


def _prefix_embed_fixture() -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info("src_embs", onnx.TensorProto.FLOAT, [1, 968, 1152]),
        helper.make_tensor_value_info("src_masks", onnx.TensorProto.FLOAT, [1, 968]),
        helper.make_tensor_value_info("src_positions", onnx.TensorProto.INT64, [1, 968]),
        helper.make_tensor_value_info("src_attention", onnx.TensorProto.FLOAT, [1, 1, 968, 968]),
    ]
    outputs = [
        helper.make_tensor_value_info("prefix_embs", onnx.TensorProto.FLOAT, [1, 968, 1152]),
        helper.make_tensor_value_info("prefix_pad_masks", onnx.TensorProto.FLOAT, [1, 968]),
        helper.make_tensor_value_info("prefix_position_ids", onnx.TensorProto.INT64, [1, 968]),
        helper.make_tensor_value_info(
            "prefix_attention_mask_4d", onnx.TensorProto.FLOAT, [1, 1, 968, 968]
        ),
    ]
    nodes = [
        helper.make_node("Identity", [inputs[i].name], [outputs[i].name], name=f"produce_{i}")
        for i in range(4)
    ]
    nodes.extend(_make_qdq("aux", "src_masks", "aux_dequant"))
    graph = helper.make_graph(
        nodes,
        "prefix_embed_fixture",
        inputs,
        outputs,
        initializer=[
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "aux_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), "aux_zero"),
        ],
    )
    return helper.make_model(graph)


def test_compact_sequence_prefix_embed() -> None:
    model = _prefix_embed_fixture()
    candidate, summary = compact_sequence(model, "prefix_embed", 256)
    assert summary["compact_prefix_tokens"] == 712
    assert summary["qdq_counts"] == {"QuantizeLinear": 1, "DequantizeLinear": 1}
    graph = candidate.graph
    assert any(
        initializer.name == "pi05_compact712_retained_indices" for initializer in graph.initializer
    )
    assert any(node.name == "pi05_compact712_prefix_embs_axis1" for node in graph.node)
    for output in graph.output:
        dims = [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
        assert 968 not in dims
        assert 712 in dims
    assert {output.name for output in graph.output} == {
        "prefix_embs",
        "prefix_pad_masks",
        "prefix_position_ids",
        "prefix_attention_mask_4d",
    }


def _prefix_lm_fixture() -> onnx.ModelProto:
    node = helper.make_node(
        "Constant",
        [],
        ["positions"],
        name="positions_constant",
        value=numpy_helper.from_array(np.asarray([968], dtype=np.int64), "positions_value"),
    )
    graph = helper.make_graph(
        [node],
        "prefix_lm_fixture",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 968, 2048])],
        [helper.make_tensor_value_info("positions", onnx.TensorProto.INT64, [1])],
        value_info=[helper.make_tensor_value_info("mid", onnx.TensorProto.FLOAT, [1, 968])],
    )
    return helper.make_model(graph)


def test_compact_sequence_prefix_lm() -> None:
    candidate, summary = compact_sequence(_prefix_lm_fixture(), "prefix_lm", 128)
    assert summary["compact_prefix_tokens"] == 456
    constant = next(node for node in candidate.graph.node if node.op_type == "Constant")
    assert numpy_helper.to_array(constant.attribute[0].t).tolist() == [456]
    input_dims = [
        dimension.dim_value for dimension in candidate.graph.input[0].type.tensor_type.shape.dim
    ]
    assert input_dims == [1, 456, 2048]


def _suffix_fixture() -> onnx.ModelProto:
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["a"],
            name="a_constant",
            value=numpy_helper.from_array(np.asarray([968], dtype=np.int64), "a_value"),
        ),
        helper.make_node(
            "Constant",
            [],
            ["b"],
            name="b_constant",
            value=numpy_helper.from_array(np.asarray([1018], dtype=np.int64), "b_value"),
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "suffix_fixture",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1018, 2048])],
        [
            helper.make_tensor_value_info("a", onnx.TensorProto.INT64, [1]),
            helper.make_tensor_value_info("b", onnx.TensorProto.INT64, [1]),
        ],
    )
    return helper.make_model(graph)


def test_compact_sequence_suffix_step() -> None:
    candidate, summary = compact_sequence(_suffix_fixture(), "suffix_step", 256)
    assert summary["compact_prefix_tokens"] == 712
    assert summary["compact_suffix_tokens"] == 762
    values = sorted(
        numpy_helper.to_array(node.attribute[0].t).tolist()[0] for node in candidate.graph.node
    )
    assert values == [712, 762]
    input_dims = [
        dimension.dim_value for dimension in candidate.graph.input[0].type.tensor_type.shape.dim
    ]
    assert input_dims == [1, 762, 2048]


def test_edit_activation_qdq(tmp_path: Path) -> None:
    nodes = _make_qdq("hot", "x", "hot_dequant")
    nodes += _make_qdq("cold", "hot_dequant", "out")
    nodes.append(helper.make_node("Identity", ["out"], ["y"], name="sink"))
    graph = helper.make_graph(
        nodes,
        "edit_fixture",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4])],
        initializer=[
            numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "hot_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), "hot_zero"),
            numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "cold_scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), "cold_zero"),
        ],
    )
    candidate, summary = edit_activation_qdq(
        helper.make_model(graph), (r"hot_Q",), (r"cold_Q=0.125",)
    )
    assert summary["removed_count"] == 1
    assert summary["removed_qdq"][0]["quantize"] == "hot_Q"
    assert summary["scale_edit_count"] == 1
    assert summary["scale_edits"][0] == {
        "quantize": "cold_Q",
        "scale": "cold_scale",
        "old": 0.25,
        "new": 0.125,
    }
    names = {node.name for node in candidate.graph.node}
    assert "hot_Q" not in names and "hot_DQ" not in names
    cold_q = next(node for node in candidate.graph.node if node.name == "cold_Q")
    assert cold_q.input[0] == "x"
    initializers = {initializer.name: initializer for initializer in candidate.graph.initializer}
    assert numpy_helper.to_array(initializers["cold_scale"]).item() == 0.125
    assert "hot_scale" not in initializers
    save_model(candidate, tmp_path / "edited.onnx", None, check=True)


def _ffn_fixture(hidden: int = 16, intermediate: int = 8) -> onnx.ModelProto:
    initializers = []
    for layer in range(17):
        names = ffn_onnx_names(layer)
        for projection, shape, scale_axis in (
            ("gate", (hidden, intermediate), 1),
            ("up", (hidden, intermediate), 1),
            ("down", (intermediate, hidden), 1),
        ):
            rng = np.random.default_rng(layer * 3 + scale_axis)
            weight = rng.integers(-127, 127, size=shape, dtype=np.int8)
            initializers.append(numpy_helper.from_array(weight, names[f"{projection}_weight"]))
            scale_shape = shape[scale_axis]
            initializers.append(
                numpy_helper.from_array(
                    rng.random(scale_shape, dtype=np.float32), names[f"{projection}_scale"]
                )
            )
            initializers.append(
                numpy_helper.from_array(
                    np.zeros(scale_shape, dtype=np.int8), names[f"{projection}_zero"]
                )
            )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="sink")],
        "ffn_fixture",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4])],
        initializer=initializers,
    )
    return helper.make_model(graph)


def _keep_json(tmp_path: Path, name: str, layers: int, keep: list[int]) -> Path:
    payload = {"layers": {str(layer): {"keep_indices": keep} for layer in range(layers)}}
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_narrow_ffn(tmp_path: Path) -> None:
    base_keep = _keep_json(tmp_path, "base_keep.json", 17, list(range(8)))
    target_keep = _keep_json(tmp_path, "target_keep.json", 17, [1, 3, 5, 7])
    candidate, summary = narrow_ffn(_ffn_fixture(), base_keep, target_keep)
    assert summary["layers"]["0"]["base_kept"] == 8
    assert summary["layers"]["0"]["target_kept"] == 4
    initializers = {initializer.name: initializer for initializer in candidate.graph.initializer}
    assert numpy_helper.to_array(initializers[ffn_onnx_names(0)["gate_weight"]]).shape == (16, 4)
    assert numpy_helper.to_array(initializers[ffn_onnx_names(0)["gate_scale"]]).shape == (4,)
    assert numpy_helper.to_array(initializers[ffn_onnx_names(0)["down_weight"]]).shape == (4, 16)
    save_model(candidate, tmp_path / "narrowed.onnx", None, check=True)


def _narrowed_base_arrays(model: onnx.ModelProto, keep: list[int]) -> dict[str, np.ndarray]:
    positions = np.asarray(keep, dtype=np.int64)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    arrays = {}
    for layer in range(17):
        arrays[f"layer{layer}_keep"] = np.asarray(keep, dtype=np.int32)
        for projection in ("gate", "up", "down"):
            names = ffn_onnx_names(layer)
            weight = numpy_helper.to_array(initializers[names[f"{projection}_weight"]])
            scale = numpy_helper.to_array(initializers[names[f"{projection}_scale"]])
            zero = numpy_helper.to_array(initializers[names[f"{projection}_zero"]])
            if projection in {"gate", "up"}:
                arrays[f"layer{layer}_{projection}_w"] = np.ascontiguousarray(weight[:, positions])
                arrays[f"layer{layer}_{projection}_scale"] = np.ascontiguousarray(scale[positions])
                arrays[f"layer{layer}_{projection}_zero"] = np.ascontiguousarray(zero[positions])
            else:
                arrays[f"layer{layer}_{projection}_w"] = np.ascontiguousarray(weight[positions, :])
                arrays[f"layer{layer}_{projection}_scale"] = np.ascontiguousarray(scale)
                arrays[f"layer{layer}_{projection}_zero"] = np.ascontiguousarray(zero)
    return arrays


def test_apply_sparse_delta(tmp_path: Path) -> None:
    base_keep = _keep_json(tmp_path, "base_keep.json", 17, list(range(8)))
    target_keep = _keep_json(tmp_path, "target_keep.json", 17, [1, 3, 5, 7])
    model = _ffn_fixture()
    base_arrays = _narrowed_base_arrays(model, [1, 3, 5, 7])

    delta_arrays = {}
    delta_arrays["layer0_gate_scale_indices"] = np.asarray([0, 3], dtype=np.int64)
    delta_arrays["layer0_gate_scale_values"] = np.asarray([0.01, 0.02], dtype=np.float32)
    delta_arrays["layer1_down_w_indices"] = np.asarray([1, 5, 9], dtype=np.int64)
    delta_arrays["layer1_down_w_values"] = np.asarray([1, -2, 40], dtype=np.int8)

    entries = {}
    for name, value in base_arrays.items():
        entries[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "changed": 0}
    entries["layer0_gate_scale"]["changed"] = 2
    entries["layer1_down_w"]["changed"] = 3

    target_arrays = {}
    for name, value in base_arrays.items():
        target = value.copy()
        indices_name = f"{name}_indices"
        if indices_name in delta_arrays:
            indices = delta_arrays[indices_name]
            updates = delta_arrays[f"{name}_values"]
            if value.dtype == np.float32:
                target[indices] = updates.astype(np.float32)
            else:
                flat = target.reshape(-1)
                flat[indices] = (flat[indices].astype(np.int16) + updates.astype(np.int16)).astype(
                    np.int8
                )
        target_arrays[name] = target

    np.savez(tmp_path / "delta.npz", **delta_arrays)
    delta_path = tmp_path / "delta.npz"
    manifest = {
        "format": "flat_indices_int_delta_and_float32_replacement_v2",
        "entries": entries,
        "changed_arrays": 2,
        "changed_values": 5,
        "base_canonical_sha256": canonical_sha256(base_arrays),
        "target_canonical_sha256": canonical_sha256(target_arrays),
        "delta_file_sha256": hashlib.sha256(delta_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    candidate, summary = apply_sparse_delta(
        model, delta_path, manifest_path, base_keep, target_keep
    )
    assert summary["changed_arrays"] == 2
    assert summary["changed_values"] == 5
    result = {initializer.name: initializer for initializer in candidate.graph.initializer}
    gate_scale = numpy_helper.to_array(result[ffn_onnx_names(0)["gate_scale"]])
    assert gate_scale[0] == 0.01 and gate_scale[3] == 0.02
    assert numpy_helper.to_array(result[ffn_onnx_names(0)["gate_scale"]])[1] != 0.01
    assert np.array_equal(
        numpy_helper.to_array(result[ffn_onnx_names(1)["down_weight"]]),
        target_arrays["layer1_down_w"],
    )
    save_model(candidate, tmp_path / "delta_applied.onnx", None, check=True)


def test_apply_sparse_delta_rejects_wrong_delta_sha(tmp_path: Path) -> None:
    base_keep = _keep_json(tmp_path, "base_keep.json", 17, list(range(8)))
    target_keep = _keep_json(tmp_path, "target_keep.json", 17, [1, 3, 5, 7])
    model = _ffn_fixture()
    base_arrays = _narrowed_base_arrays(model, [1, 3, 5, 7])
    entries = {
        name: {"dtype": str(value.dtype), "shape": list(value.shape), "changed": 0}
        for name, value in base_arrays.items()
    }
    manifest = {
        "format": "flat_indices_int_delta_and_float32_replacement_v2",
        "entries": entries,
        "changed_arrays": 0,
        "changed_values": 0,
        "base_canonical_sha256": canonical_sha256(base_arrays),
        "target_canonical_sha256": canonical_sha256(base_arrays),
        "delta_file_sha256": "0" * 64,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    delta_path = tmp_path / "delta.npz"
    np.savez(delta_path)
    try:
        apply_sparse_delta(model, delta_path, manifest_path, base_keep, target_keep)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "SHA256" in str(error)


def test_rewrite_io_dimensions() -> None:
    payload = {"prefix_embs": {"shape": [1, 968, 1152]}, "cache": {"shapes": [[1, 968], [1, 968]]}}
    rewritten, count = rewrite_io_dimensions(payload, {968: 712}, 3)
    assert count == 3
    assert rewritten["prefix_embs"]["shape"] == [1, 712, 1152]
    assert rewritten["cache"]["shapes"] == [[1, 712], [1, 712]]
    try:
        rewrite_io_dimensions(payload, {968: 712}, 2)
        raise AssertionError("expected ValueError")
    except ValueError as error:
        assert "expected 2 replacements, found 3" in str(error)


def test_audit_sequence(tmp_path: Path) -> None:
    model = _prefix_lm_fixture()
    path = tmp_path / "prefix_lm.onnx"
    onnx.save(model, str(path))
    report = audit_sequence([path])
    assert report["classification"] == "pi05_compact_sequence_onnx_audit"
    entry = report["models"][0]
    assert any(item["name"] == "positions_constant" for item in entry["sequence_constants"])
    assert any(
        value["name"] == "mid" and 968 in value["shape"]
        for value in entry["sequence_shaped_values"]
    )
