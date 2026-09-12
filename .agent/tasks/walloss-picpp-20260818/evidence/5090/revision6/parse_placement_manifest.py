#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import onnx

ROOT = Path("/work")
LOG = ROOT / "logs" / "placement_session_only.log"
PACKAGE = ROOT / "inbound_v18" / "onnx_real_legacy_v18"
OUT = ROOT / "results" / "placement_manifest.json"
CPU_OUT = ROOT / "results" / "cpu_allowlist.json"

PLACEMENT = re.compile(r"Node\(s\) placed on \[(CPUExecutionProvider|CUDAExecutionProvider)\].*Number of nodes: (\d+)")
ALL_PLACED = re.compile(r"All nodes placed on \[(CUDAExecutionProvider|CPUExecutionProvider)\].*Number of nodes: (\d+)")
NODE = re.compile(r"\s+(\w+) \(([^)]*)\)")
MEMCPY = re.compile(r"Add (MemcpyFromHost|MemcpyToHost) (?:after|before) (.+)")

DENY = {
    "MatMul", "Gemm", "Conv", "Attention", "MultiHeadAttention", "Softmax",
    "LayerNormalization", "SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization",
    "RMSNormalization", "BatchNormalization", "GroupNormalization", "InstanceNormalization",
    "ReduceMean", "ReduceSum", "Pow", "Sqrt", "Exp", "Log", "Sin", "Cos",
}
CONTROL_DTYPES = {"BOOL", "INT32", "INT64"}


def dtype_name(elem_type: int) -> str:
    return onnx.TensorProto.DataType.Name(elem_type)


def graph_inventory(stage: str) -> dict[str, dict]:
    optimized = ROOT / "tmp" / f"{stage}.optimized.onnx"
    model = onnx.load(str(optimized if optimized.exists() else PACKAGE / f"{stage}.onnx"), load_external_data=False)
    types = {}
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.type.HasField("tensor_type"):
            types[value.name] = dtype_name(value.type.tensor_type.elem_type)
    for tensor in model.graph.initializer:
        types[tensor.name] = dtype_name(tensor.data_type)
    raw_nodes = list(model.graph.node)
    producer = {output: node for node in raw_nodes for output in node.output}

    def infer(name: str, active: set[str] | None = None) -> str:
        if name in types:
            return types[name]
        active = set() if active is None else active
        if name in active or name not in producer:
            return "UNKNOWN"
        active.add(name)
        node = producer[name]
        op = node.op_type
        result = "UNKNOWN"
        if op == "Shape" or op == "Size" or op == "NonZero":
            result = "INT64"
        elif op in {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Not"}:
            result = "BOOL"
        elif op == "Cast":
            attr = next((item for item in node.attribute if item.name == "to"), None)
            result = dtype_name(attr.i) if attr is not None else "UNKNOWN"
        elif op == "Constant":
            attr = next((item for item in node.attribute if item.name == "value"), None)
            result = dtype_name(attr.t.data_type) if attr is not None and attr.HasField("t") else "UNKNOWN"
        elif op == "ConstantOfShape":
            attr = next((item for item in node.attribute if item.name == "value"), None)
            result = dtype_name(attr.t.data_type) if attr is not None and attr.HasField("t") else "FLOAT"
        elif op == "Where":
            result = infer(node.input[1], active) if len(node.input) > 1 else "UNKNOWN"
            if result == "UNKNOWN" and len(node.input) > 2:
                result = infer(node.input[2], active)
        elif op in {"Gather", "GatherElements", "GatherND", "Unsqueeze", "Squeeze", "Reshape", "Transpose", "Slice", "Concat", "Expand", "Pad", "Identity", "Flatten", "ScatterElements", "ScatterND", "CumSum", "MemcpyFromHost", "MemcpyToHost", "Split", "Add", "Sub", "Mul", "Div", "Min", "Max"}:
            result = infer(node.input[0], active) if node.input else "UNKNOWN"
        types[name] = result
        active.remove(name)
        return result

    for node in raw_nodes:
        for output in node.output:
            infer(output)
    nodes = {node.name: {"op_type": node.op_type, "inputs": list(node.input), "outputs": list(node.output)} for node in raw_nodes}
    return {"types": types, "nodes": nodes}


def main() -> None:
    text = LOG.read_text(errors="replace")
    placements_by_stage = {}
    stage = None
    placements = []
    current_provider = None
    current_expected_count = None
    for line in text.splitlines():
        if line.startswith("BEGIN_STAGE "):
            stage = line.split()[1]
            placements = []
            placements_by_stage[stage] = placements
            current_provider = None
            continue
        if line.startswith("END_STAGE "):
            stage = None
            current_provider = None
            continue
        match = PLACEMENT.search(line)
        if match:
            current_provider = match.group(1)
            current_expected_count = int(match.group(2))
            placements.append({"provider": current_provider, "expected_count": current_expected_count, "nodes": []})
            continue
        match = ALL_PLACED.search(line)
        if match:
            current_provider = match.group(1)
            current_expected_count = int(match.group(2))
            placements.append({"provider": current_provider, "expected_count": current_expected_count, "nodes": []})
            continue
        if placements and current_provider:
            node = NODE.search(line)
            if node:
                placements[-1]["nodes"].append({"op_type": node.group(1), "name": node.group(2)})
    stages = {}
    for stage in ("prefix_embed", "prefill", "postfix_step"):
        graph = graph_inventory(stage)
        stage_rows = []
        for group in placements_by_stage.get(stage, []):
            for row in group["nodes"]:
                if row["name"] not in graph["nodes"]:
                    continue
                node = graph["nodes"][row["name"]]
                tensor_types = sorted({graph["types"].get(name, "UNKNOWN") for name in [*node["inputs"], *node["outputs"]]})
                stage_rows.append({"provider": group["provider"], "name": row["name"], "op_type": node["op_type"], "tensor_types": tensor_types, "inputs": node["inputs"], "outputs": node["outputs"]})
        stages[stage] = {"rows": stage_rows, "counts": {provider: sum(1 for row in stage_rows if row["provider"] == provider) for provider in ("CPUExecutionProvider", "CUDAExecutionProvider")}}
    memcpy = []
    memcpy_stage = None
    for line in text.splitlines():
        if line.startswith("BEGIN_STAGE "):
            memcpy_stage = line.split()[1]
        elif line.startswith("END_STAGE "):
            memcpy_stage = None
        else:
            match = MEMCPY.search(line)
            if match:
                memcpy.append({"stage": memcpy_stage, "direction": match.group(1), "boundary": match.group(2)})
    all_rows = [row for stage in stages.values() for row in stage["rows"]]
    denied_cpu = [row for row in all_rows if row["provider"] == "CPUExecutionProvider" and (row["op_type"] in DENY or any(dtype not in CONTROL_DTYPES for dtype in row["tensor_types"]))]
    counts_from_log = []
    for stage_name, groups in placements_by_stage.items():
        for group in groups:
            counts_from_log.append({"stage": stage_name, "provider": group["provider"], "expected_count": group["expected_count"], "parsed_node_lines": len(group["nodes"])})
    report = {"schema_version": "walloss.revision6.placement-manifest.v1", "counts_from_log": counts_from_log, "stages": stages, "memcpy_boundaries": memcpy, "cpu_control_only": not denied_cpu, "denied_cpu_rows": denied_cpu, "expected_cpu_counts": {"prefix_embed": 8, "prefill": 3481, "postfix_step": 0}}
    expected_total = {"prefix_embed": 1823, "prefill": 17094, "postfix_step": 4184}
    report["count_match"] = all(
        stages[stage]["counts"]["CPUExecutionProvider"] == report["expected_cpu_counts"][stage]
        and (
            sum(stages[stage]["counts"].values()) == expected_total[stage]
            or stage == "postfix_step" and any(
                row["stage"] == stage and row["provider"] == "CUDAExecutionProvider" and row["expected_count"] == expected_total[stage]
                for row in counts_from_log
            )
        )
        for stage in stages
    )
    report["allowlist_pass"] = bool(report["count_match"] and report["cpu_control_only"])
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    OUT.write_text(payload)
    cpu_allowlist = {
        "schema_version": "walloss.revision6.cpu-allowlist.v1",
        "source_manifest": str(OUT),
        "allowlist_pass": report["allowlist_pass"],
        "count_match": report["count_match"],
        "cpu_control_only": report["cpu_control_only"],
        "memcpy_boundaries": memcpy,
        "stages": {
            stage: {
                "counts": value["counts"],
                "cpu_rows": [row for row in value["rows"] if row["provider"] == "CPUExecutionProvider"],
            }
            for stage, value in stages.items()
        },
    }
    CPU_OUT.write_text(json.dumps(cpu_allowlist, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"count_match": report["count_match"], "cpu_control_only": report["cpu_control_only"], "allowlist_pass": report["allowlist_pass"], "memcpy_count": len(memcpy)}, indent=2, sort_keys=True))
    if not report["allowlist_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
