from __future__ import annotations

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper
from pi_cpp.toolchain.graph import cast_cache_abi, count_mlp_matmuls, insert_qdq, topo_sort
from pi_cpp.toolchain.recipe import BuildRecipe, engine_lock_entry

CACHES = 36


def _cache_tensor(name: str, elem_type: int) -> helper.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, [1, 32, 64])


def _prefix_model() -> ModelProto:
    nodes = [helper.make_node("Identity", [f"src_{i}"], [f"cache_{i:02d}"], name=f"prod_{i}") for i in range(CACHES)]
    inputs = [helper.make_tensor_value_info(f"src_{i}", TensorProto.BFLOAT16, [1, 32, 64]) for i in range(CACHES)]
    outputs = [_cache_tensor(f"cache_{i:02d}", TensorProto.BFLOAT16) for i in range(CACHES)]
    return helper.make_model(helper.make_graph(nodes, "prefix", inputs, outputs))


def _suffix_model() -> ModelProto:
    nodes = [helper.make_node("Identity", [f"cache_{i:02d}"], [f"out_{i}"], name=f"cons_{i}") for i in range(CACHES)]
    inputs = [_cache_tensor(f"cache_{i:02d}", TensorProto.BFLOAT16) for i in range(CACHES)]
    outputs = [helper.make_tensor_value_info(f"out_{i}", TensorProto.BFLOAT16, [1, 32, 64]) for i in range(CACHES)]
    return helper.make_model(helper.make_graph(nodes, "suffix", inputs, outputs))


def test_topo_sort_and_cycle():
    a = helper.make_node("Identity", ["x"], ["y"], name="a")
    b = helper.make_node("Identity", ["y"], ["z"], name="b")
    assert [n.name for n in topo_sort([b, a])] == ["a", "b"]
    cycle = helper.make_node("Identity", ["z"], ["x"], name="c")
    with pytest.raises(ValueError, match="cycle"):
        topo_sort([a, b, cycle])


def test_cast_cache_abi_prefix():
    model = cast_cache_abi(_prefix_model(), stage="prefix")
    graph = model.graph
    assert all(o.type.tensor_type.elem_type == TensorProto.FLOAT16 for o in graph.output)
    casts = [n for n in graph.node if n.op_type == "Cast"]
    assert len(casts) == CACHES
    producers = [n for n in graph.node if n.op_type == "Identity"]
    assert all(n.output == [f"cache_{i:02d}_bf16"] for i, n in enumerate(producers))


def test_cast_cache_abi_suffix():
    model = cast_cache_abi(_suffix_model(), stage="suffix")
    graph = model.graph
    assert all(i.type.tensor_type.elem_type == TensorProto.FLOAT16 for i in graph.input)
    casts = [n for n in graph.node if n.op_type == "Cast"]
    assert len(casts) == CACHES
    assert all(cast.attribute[0].i == TensorProto.BFLOAT16 for cast in casts)
    consumers = [n for n in graph.node if n.op_type == "Identity"]
    assert all(n.input == [f"cache_{i:02d}_bf16"] for i, n in enumerate(consumers))


def test_cast_cache_abi_fail_fast():
    with pytest.raises(ValueError, match="36 cache tensors"):
        model = _prefix_model()
        del model.graph.output[:1]
        cast_cache_abi(model, stage="prefix")
    with pytest.raises(ValueError, match="must be bf16"):
        model = _prefix_model()
        model.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT16
        cast_cache_abi(model, stage="prefix")
    with pytest.raises(ValueError, match="stage must be"):
        cast_cache_abi(_prefix_model(), stage="nope")


def _qdq_model() -> ModelProto:
    weight = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["w_raw"], value=numpy_helper.from_array(weight)),
            helper.make_node("Transpose", ["w_raw"], ["w_t"], name="w_t"),
            helper.make_node("MatMul", ["act_0", "w_t"], ["mm_out"], name="/mlp/gate/MatMul"),
        ],
        "qdq",
        [helper.make_tensor_value_info("act_0", TensorProto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info("mm_out", TensorProto.FLOAT, [1, 64])],
    )
    return helper.make_model(graph)


def test_insert_qdq():
    model, quantized = insert_qdq(_qdq_model(), act_scales={"act_0": 0.5})
    assert quantized == 1
    assert count_mlp_matmuls(model) == 1
    q_nodes = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    dq_nodes = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]
    assert len(q_nodes) == 1 and len(dq_nodes) == 1
    matmul = next(n for n in model.graph.node if n.op_type == "MatMul")
    assert matmul.input[1] == dq_nodes[0].output[0]
    scale_name = q_nodes[0].input[1]
    scales = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    expected = np.maximum(np.abs(np.arange(32 * 64, dtype=np.float32).reshape(32, 64)).max(axis=-1), 1e-6) / 127.0
    assert np.allclose(scales[scale_name], expected)


def test_insert_qdq_missing_scale_fails():
    with pytest.raises(ValueError, match="missing activation scales"):
        insert_qdq(_qdq_model(), act_scales={})


def test_recipe_traps(tmp_path):
    with pytest.raises(ValueError, match="commonEmitDebugTensor"):
        BuildRecipe.from_dict({"onnx": "a.onnx", "output": "a.engine", "precision": "int8"})
    with pytest.raises(ValueError, match="requires act_scales"):
        BuildRecipe.from_dict({"onnx": "a.onnx", "output": "a.engine", "precision": "qdq"})
    recipe = BuildRecipe.from_dict({"onnx": "a.onnx", "output": "a.engine", "precision": "qdq", "act_scales": "s.npz"})
    assert recipe.workspace_gb == 4
    assert recipe.check_traps() is None
    path = tmp_path / "recipe.json"
    path.write_text('{"onnx": "a.onnx", "output": "a.engine"}')
    assert BuildRecipe.load(path).precision == "bf16"


def test_engine_lock_entry(tmp_path):
    engine = tmp_path / "suffix.engine"
    engine.write_bytes(b"\x00\x01" * 100)
    entry = engine_lock_entry(engine)
    assert entry["size_bytes"] == 200
    assert len(entry["sha256"]) == 64
