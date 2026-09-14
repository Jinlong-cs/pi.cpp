#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import platform
import time
from pathlib import Path

import onnxruntime as ort

ROOT = Path("/work")
PACKAGE = ROOT / "inbound_v18" / "onnx_real_legacy_v18"
OUT = ROOT / "results" / "placement_session_only_summary.json"


def main() -> None:
    report = {
        "schema_version": "walloss.revision6.session-only-placement.v1",
        "host": platform.node(),
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "provider_order": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "disabled_optimizers": ["SimplifiedLayerNormFusion"],
        "inference_called": False,
        "stages": {},
    }
    for stage in ("prefix_embed", "prefill", "postfix_step"):
        print(f"BEGIN_STAGE {stage}", flush=True)
        path = PACKAGE / f"{stage}.onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 0
        options.log_verbosity_level = 1
        options.optimized_model_filepath = str(ROOT / "tmp" / f"{stage}.optimized.onnx")
        started = time.monotonic()
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            disabled_optimizers={"SimplifiedLayerNormFusion"},
        )
        report["stages"][stage] = {
            "session_created": True,
            "seconds": time.monotonic() - started,
            "session_providers": session.get_providers(),
            "input_count": len(session.get_inputs()),
            "output_count": len(session.get_outputs()),
        }
        del session
        gc.collect()
        print(f"END_STAGE {stage}", flush=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
