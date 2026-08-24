"""Latency command implementation for the pi.cpp CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pi_cpp as picpp
from rich.console import Console

console = Console()

WARMUP_ITERS = 10
MEASURE_ITERS = 10
SEED = 1234


def _run_pi05_latency(*, model_dir: Path | None, prompt: str) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_pi05_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.random((3, height, width), dtype=np.float32) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=prompt, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "prefix_embed_ms": [],
        "prefix_lm_ms": [],
        "suffix_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=prompt, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_pi06_airbot_latency(*, model_dir: Path | None, prompt: str) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_pi06_airbot_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=prompt, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "prefix_embed_ms": [],
        "prefix_lm_ms": [],
        "suffix_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=prompt, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_fastwam_latency(*, model_dir: Path | None, prompt: str) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_fastwam_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.random((3, height, width), dtype=np.float32) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=prompt, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "vae_image_encoder_ms": [],
        "video_prefill_ms": [],
        "action_loop_ms": [],
        "action_decode_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=prompt, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_semanticvla_latency(*, model_dir: Path | None) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_semanticvla_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "backbone_ms": [],
        "action_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_evo1_latency(*, model_dir: Path | None) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_evo1_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "backbone_ms": [],
        "action_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["output_finite"] = bool(np.isfinite(actions).all())
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_smolvla_latency(*, model_dir: Path | None) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_smolvla_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.random((3, height, width), dtype=np.float32) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "prefix_embed_ms": [],
        "prefix_lm_ms": [],
        "suffix_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_dit4dit_latency(*, model_dir: Path | None, prompt: str) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_dit4dit_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=prompt, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "vae_ms": [],
        "feature_ms": [],
        "action_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=prompt, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_groot_latency(*, model_dir: Path | None, prompt: str) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_groot_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]
    state = np.zeros(runner.state_dim, dtype=np.float32)

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=prompt, state=state)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "backbone_ms": [],
        "action_loop_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=prompt, state=state)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def _run_starvla_latency(*, model_dir: Path | None) -> dict:
    show_progress = os.environ.get("PICPP_LATENCY_PROGRESS") == "1"
    rng = np.random.default_rng(SEED)
    runner = picpp.build_starvla_runner(model_dir=model_dir)
    height, width = runner.image_size
    images = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(runner.num_cameras)]

    for index in range(WARMUP_ITERS):
        runner.run_once(images=images, prompt=runner.default_prompt)
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "warmup", "done": index + 1, "total": WARMUP_ITERS}
            )

    wrapper_ms = []
    stage_ms: dict[str, list[float]] = {
        "preprocess_ms": [],
        "infer_ms": [],
        "policy_ms": [],
        "postprocess_ms": [],
    }
    for index in range(MEASURE_ITERS):
        start = perf_counter()
        actions = runner.run_once(images=images, prompt=runner.default_prompt)
        wrapper_ms.append((perf_counter() - start) * 1000.0)
        for key in stage_ms:
            stage_ms[key].append(float(runner.metadata[key]))
        if show_progress:
            console.print_json(
                data={"event": "latency_progress", "phase": "measure", "done": index + 1, "total": MEASURE_ITERS}
            )

    metadata = dict(runner.metadata)
    metadata["output_shape"] = list(actions.shape)
    metadata["wrapper_ms"] = float(sum(wrapper_ms) / len(wrapper_ms))
    for key, values in stage_ms.items():
        metadata[key] = float(sum(values) / len(values))
    return metadata


def run_latency(*, model: str, model_dir: Path | None, prompt: str) -> int:
    if model == "pi05":
        metadata = _run_pi05_latency(model_dir=model_dir, prompt=prompt)
    elif model == "pi06_airbot":
        metadata = _run_pi06_airbot_latency(model_dir=model_dir, prompt=prompt)
    elif model == "fastwam":
        metadata = _run_fastwam_latency(model_dir=model_dir, prompt=prompt)
    elif model == "semanticvla":
        metadata = _run_semanticvla_latency(model_dir=model_dir)
    elif model == "evo1":
        metadata = _run_evo1_latency(model_dir=model_dir)
    elif model == "smolvla":
        metadata = _run_smolvla_latency(model_dir=model_dir)
    elif model == "dit4dit":
        metadata = _run_dit4dit_latency(model_dir=model_dir, prompt=prompt)
    elif model == "groot":
        metadata = _run_groot_latency(model_dir=model_dir, prompt=prompt)
    elif model == "starvla":
        metadata = _run_starvla_latency(model_dir=model_dir)
    else:
        raise ValueError(f"unsupported latency model: {model}")

    metadata["command"] = "latency"
    metadata["model"] = model
    metadata["warmup"] = WARMUP_ITERS
    metadata["iters"] = MEASURE_ITERS
    profile_path = Path(str(metadata["model_dir"])).expanduser() / "latency.json"
    metadata["latency_profile_path"] = str(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(metadata, indent=2) + "\n")

    console.print_json(data=metadata)
    return 0
