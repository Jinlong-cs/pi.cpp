"""Standalone CLI for offline pi.cpp optimization workflows."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import onnx

from pi_cpp.optimization.compare import compare_outputs, load_npz, run_onnx
from pi_cpp.optimization.contracts import read_bundle, write_json
from pi_cpp.optimization.onnx_graph import inspect_model
from pi_cpp.optimization.qdq import QDQStageConfig, quantize_model, save_model, strip_qdq_pairs
from pi_cpp.optimization.trt_tools import (
    benchmark_engine,
    build_engine,
    inspect_engine,
    read_shapes,
)


def _absent(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(path)
    return path


def _recipe(path: Path, stage: str) -> tuple[dict[str, object], QDQStageConfig]:
    model, target, plan, evidence = read_bundle(path)
    return {
        "recipe": str(path),
        "model_spec": asdict(model),
        "target_fingerprint": asdict(target),
        "optimization_plan": asdict(plan),
        "recorded_evidence": [record.to_dict() for record in evidence],
        "stage": stage,
    }, QDQStageConfig.from_dict(plan.stage(stage))


def _add_trt_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--shapes-json", type=Path)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect", help="Inspect an ONNX graph and constant-weight linear candidates."
    )
    inspect.add_argument("--onnx", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)

    for name in ("qdq", "rollback"):
        qdq = commands.add_parser(name, help="Generate a selective QDQ graph from an FP source.")
        qdq.add_argument("--source-onnx", type=Path, required=True)
        qdq.add_argument("--output-onnx", type=Path, required=True)
        qdq.add_argument("--recipe", type=Path, required=True)
        qdq.add_argument("--stage", required=True)
        qdq.add_argument("--summary", type=Path, required=True)
        qdq.add_argument("--external-data-file")
        qdq.add_argument("--keep-fp16", action="append", default=[])
        qdq.add_argument("--skip-check", action="store_true")

    strip = commands.add_parser(
        "strip-qdq", help="Remove isolated activation Q/DQ pairs for an analysis-only control."
    )
    strip.add_argument("--input-onnx", type=Path, required=True)
    strip.add_argument("--output-onnx", type=Path, required=True)
    strip.add_argument("--summary", type=Path, required=True)
    strip.add_argument("--name-pattern", action="append", default=[])
    strip.add_argument("--external-data-file")
    strip.add_argument("--skip-check", action="store_true")

    compare = commands.add_parser(
        "compare", help="Run two ONNX models on the same NPZ inputs and compare outputs."
    )
    compare.add_argument("--reference-onnx", type=Path, required=True)
    compare.add_argument("--candidate-onnx", type=Path, required=True)
    compare.add_argument("--inputs-npz", type=Path, required=True)
    compare.add_argument("--provider", action="append", default=[])
    compare.add_argument("--output", type=Path, required=True)

    build = commands.add_parser(
        "trt-build", help="Build a TensorRT engine with target-local trtexec."
    )
    build.add_argument("--onnx", type=Path, required=True)
    _add_trt_common(build)
    build.add_argument("--fp16", action="store_true")
    build.add_argument("--int8", action="store_true")
    build.add_argument("--builder-optimization-level", type=int, choices=range(6), default=3)
    build.add_argument("--workspace-mib", type=int, default=4096)
    build.add_argument("--detailed", action="store_true")
    build.add_argument("--layer-info", type=Path, required=True)

    engine_inspect = commands.add_parser(
        "trt-inspect", help="Export TensorRT engine layer information."
    )
    _add_trt_common(engine_inspect)
    engine_inspect.add_argument("--layer-info", type=Path, required=True)

    bench = commands.add_parser(
        "trt-bench", help="Benchmark one TensorRT engine stage with explicit timing boundaries."
    )
    _add_trt_common(bench)
    bench.add_argument("--warmup-ms", type=int, default=1000)
    bench.add_argument("--repeats", type=int, default=100)
    bench.add_argument("--times", type=Path, required=True)
    bench.add_argument("--profile", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        write_json(_absent(args.output), inspect_model(args.onnx))
    elif args.command in {"qdq", "rollback"}:
        recipe, config = _recipe(args.recipe, args.stage)
        if args.keep_fp16:
            config = config.with_guards(tuple(args.keep_fp16))
        model = onnx.load_model(args.source_onnx, load_external_data=True)
        candidate, summary = quantize_model(model, config)
        save_model(candidate, args.output_onnx, args.external_data_file, check=not args.skip_check)
        write_json(
            _absent(args.summary),
            {
                **recipe,
                "source_onnx": str(args.source_onnx),
                "output_onnx": str(args.output_onnx),
                **summary,
            },
        )
    elif args.command == "strip-qdq":
        model = onnx.load_model(args.input_onnx, load_external_data=True)
        candidate, summary = strip_qdq_pairs(model, tuple(args.name_pattern))
        save_model(candidate, args.output_onnx, args.external_data_file, check=not args.skip_check)
        write_json(
            _absent(args.summary),
            {"input_onnx": str(args.input_onnx), "output_onnx": str(args.output_onnx), **summary},
        )
    elif args.command == "compare":
        providers = args.provider or ["CPUExecutionProvider"]
        inputs = load_npz(args.inputs_npz)
        reference = run_onnx(args.reference_onnx, inputs, providers)
        candidate = run_onnx(args.candidate_onnx, inputs, providers)
        write_json(
            _absent(args.output),
            {
                "reference_onnx": str(args.reference_onnx),
                "candidate_onnx": str(args.candidate_onnx),
                "inputs_npz": str(args.inputs_npz),
                "providers": providers,
                **compare_outputs(reference, candidate),
            },
        )
    elif args.command == "trt-build":
        value = build_engine(
            args.onnx,
            args.engine,
            read_shapes(args.shapes_json),
            fp16=args.fp16,
            int8=args.int8,
            builder_optimization_level=args.builder_optimization_level,
            workspace_mib=args.workspace_mib,
            detailed=args.detailed,
            trtexec=args.trtexec,
            log_path=args.log,
            layer_info_path=args.layer_info,
            target_name=args.target_name,
            architecture=args.architecture,
        )
        write_json(_absent(args.output), value)
    elif args.command == "trt-inspect":
        value = inspect_engine(
            args.engine,
            read_shapes(args.shapes_json),
            trtexec=args.trtexec,
            log_path=args.log,
            layer_info_path=args.layer_info,
            target_name=args.target_name,
            architecture=args.architecture,
        )
        write_json(_absent(args.output), value)
    elif args.command == "trt-bench":
        value = benchmark_engine(
            args.engine,
            read_shapes(args.shapes_json),
            warmup_ms=args.warmup_ms,
            repeats=args.repeats,
            trtexec=args.trtexec,
            log_path=args.log,
            times_path=args.times,
            profile_path=args.profile,
            target_name=args.target_name,
            architecture=args.architecture,
        )
        write_json(_absent(args.output), value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
