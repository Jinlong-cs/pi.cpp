"""Board-side activation-scale collection for the qdq recipe (the ported
collect_mlp_scales.py): per-tensor max_abs ranges for the selected MatMul
activations over the calibration cases, emitted as symmetric int8 scales
(max_abs/127) in an npz.

The port preserves every trap workaround from the board script as tool
behavior: the whole graph is converted bf16 -> fp32 for ORT-CPU (lossless,
so the collected ranges equal the fp32-stream ranges), and the Sin/Cos
int64 rotary angles get fp32 casts. ``import onnxruntime`` stays lazy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def _mlp_activation_tensors(graph: Any, name_contains: tuple[str, ...], name_prefixes: tuple[str, ...]) -> list[str]:
    activations = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        name = node.name or ""
        if any(marker in name for marker in name_contains) or any(name.startswith(prefix) for prefix in name_prefixes):
            activations.append(node.input[0])
    return activations


def _resolve_dtype(graph: Any, tensor: str) -> int:
    type_map = {vi.name: vi.type.tensor_type.elem_type for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    for init in graph.initializer:
        type_map[init.name] = init.data_type
    producer = {o: n for n in graph.node for o in n.output}

    def resolve(name: str, depth: int = 0) -> int:
        if depth >= 20:
            raise ValueError(f"type resolve depth exceeded for {name}")
        if name in type_map:
            return type_map[name]
        node = producer.get(name)
        if node is None:
            raise ValueError(f"no producer for {name}")
        if node.op_type == "Cast":
            dtype = node.attribute[0].i
        elif node.op_type == "Constant":
            dtype = node.attribute[0].t.data_type
        else:
            dtype = resolve(node.input[0], depth + 1)
        type_map[name] = dtype
        return dtype

    return resolve(tensor)


def _cast_graph_to_fp32(graph: Any) -> None:
    """ORT-CPU lacks bf16 kernels; bf16 -> fp32 is lossless."""
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    new_inits = []
    for init in graph.initializer:
        if init.data_type == onnx.TensorProto.BFLOAT16:
            array = onnx.numpy_helper.to_array(init).astype(np.float32)
            new_inits.append(onnx.numpy_helper.from_array(array, init.name))
        else:
            new_inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.BFLOAT16:
                    attr.i = onnx.TensorProto.FLOAT
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == onnx.TensorProto.BFLOAT16:
                    array = onnx.numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(array))
        if node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.BFLOAT16:
                    array = onnx.numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(onnx.numpy_helper.from_array(array))


def _patch_sin_cos_int64(graph: Any) -> None:
    """ORT-CPU has no Sin/Cos for int64 rotary angles: insert fp32 casts."""
    patched = 0
    casted_once: dict[str, str] = {}
    for node in list(graph.node):
        if node.op_type not in ("Sin", "Cos"):
            continue
        source = node.input[0]
        casted = casted_once.get(source)
        if casted is None:
            casted = f"{source}_f32"
            graph.node.append(onnx.helper.make_node("Cast", [source], [casted], to=onnx.TensorProto.FLOAT, name=f"/cast_{node.name}"))
            casted_once[source] = casted
        node.input[0] = casted
        patched += 1
    if patched:
        print(f"patched {patched} Sin/Cos int64 inputs")


def collect_scales(
    onnx_path: Path,
    calib_npz: Path,
    out_npz: Path,
    *,
    name_contains: tuple[str, ...] = ("/mlp",),
    name_prefixes: tuple[str, ...] = ("/time_mlp",),
    dump_path: Path | None = None,
) -> dict[str, Any]:
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    activations = _mlp_activation_tensors(graph, name_contains, name_prefixes)
    existing = {o.name for o in graph.output}
    for tensor in activations:
        if tensor in existing:
            continue
        dtype = _resolve_dtype(graph, tensor)
        graph.output.append(onnx.helper.make_tensor_value_info(tensor, dtype, None))
    _cast_graph_to_fp32(graph)
    _patch_sin_cos_int64(graph)
    probe_path = dump_path or onnx_path.with_name(f"{onnx_path.stem}_probe.onnx")
    onnx.save(model, probe_path)
    del model

    session = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
    data = np.load(calib_npz)
    names = sorted(data.files)
    feeds = {name: data[name] for name in names}
    case_count = feeds[names[0]].shape[0]
    maxima = {tensor: 0.0 for tensor in activations}
    t0 = time.time()
    for i in range(case_count):
        feed = {name: values[i] for name, values in feeds.items()}
        outputs = session.run(activations, feed)
        for tensor, output in zip(activations, outputs, strict=True):
            maxima[tensor] = max(maxima[tensor], float(np.abs(np.asarray(output)).max()))
    scales = {tensor: np.float32(max(maxima[tensor], 1e-6) / 127.0) for tensor in activations}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **scales)
    return {
        "activations": len(activations),
        "cases": case_count,
        "scale_min": float(min(scales.values())),
        "scale_max": float(max(scales.values())),
        "scales": out_npz.name,
        "probe_graph": probe_path.name,
        "elapsed_s": round(time.time() - t0, 1),
    }
