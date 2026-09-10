"""Single-shot open-loop inference command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

import pi_cpp as picpp

console = Console()

PI06_HETEROGENEOUS_CASE_TENSORS = (
    ("image", "image_float32_nchw.npy"),
    ("image_mask", "image_mask_bool.npy"),
    ("tokenized_prompt", "token_ids_int64.npy"),
    ("tokenized_prompt_mask", "token_mask_bool.npy"),
    ("state", "normalized_state.npy"),
    ("embodiment_id", "embodiment_id_int32.npy"),
)

PI06_RTC_CASE_TENSORS = PI06_HETEROGENEOUS_CASE_TENSORS + (
    ("delay", "rtc_delay_int32.npy"),
    ("action_prefix", "rtc_action_prefix_normalized.npy"),
)


def _load_case_tensors(case_dir: Path) -> dict[str, np.ndarray]:
    tensors = {}
    for name, filename in PI06_HETEROGENEOUS_CASE_TENSORS:
        path = case_dir / filename
        if not path.exists():
            raise ValueError(f"PI0.6 heterogeneous case tensor {filename} missing in {case_dir}")
        tensors[name] = np.load(path)
    return tensors


def _run_pi06_heterogeneous_infer(
    *,
    model_dir: Path | None,
    case_dir: Path | None,
    prompt: str | None,
    state: str | None,
    images: list[Path],
    embodiment_id: int | None,
    noise_file: Path | None,
    output: Path | None,
) -> int:
    runner = picpp.build_pi06_heterogeneous_runner(model_dir=model_dir)
    if case_dir is not None:
        tensors = _load_case_tensors(Path(case_dir))
        noise_path = noise_file if noise_file is not None else Path(case_dir) / "noise_seed.npy"
        if noise_path.exists():
            noise = np.load(noise_path)
            expected = (1, runner.action_horizon, runner.action_dim)
            if noise.shape == (runner.action_horizon, runner.action_dim):
                noise = noise[None]
            if noise.shape != expected:
                raise ValueError(f"noise file {noise_path} has shape {noise.shape}, expected {expected}")
            tensors["x_t"] = np.ascontiguousarray(noise, dtype=np.float32)
        elif "x_t" not in tensors:
            raise ValueError(f"case {case_dir} has no noise: pass --noise-file or add x_t")
        actions = runner.run_abi(tensors)
        metadata = dict(runner.metadata)
        metadata["case_dir"] = str(case_dir)
    else:
        if prompt is None or state is None or not images:
            raise ValueError(
                "PI0.6 heterogeneous live inference needs --prompt, --state, and at least one --image "
                "(or use --case-dir for frozen-case replay)"
            )
        state_values = [float(part) for part in state.split(",")]
        if len(state_values) != runner.state_dim:
            raise ValueError(f"--state must contain {runner.state_dim} comma-separated floats, got {len(state_values)}")
        image_arrays = [np.asarray(_load_image(path), dtype=np.uint8) for path in images]
        if len(image_arrays) != runner.num_cameras:
            raise ValueError(
                f"PI0.6 heterogeneous expects {runner.num_cameras} images in {runner.spec.camera_order}, got {len(image_arrays)}"
            )
        noise = None
        if noise_file is not None:
            noise = np.load(noise_file)
        actions = runner.run_once(
            images=image_arrays,
            prompt=prompt,
            state=np.asarray(state_values, dtype=np.float32),
            embodiment_id=embodiment_id,
            noise=noise,
        )
        metadata = dict(runner.metadata)
        metadata["noise_mode"] = "explicit" if noise_file is not None else "seeded_stream"

    if output is not None:
        np.save(output, actions)
        metadata["output"] = str(output)
    console.print_json(
        data={
            "model": "pi06_heterogeneous",
            "action_horizon": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "actions": actions.tolist(),
            **metadata,
        }
    )
    return 0


def _run_pi06_rtc_infer(
    *,
    model_dir: Path | None,
    case_dir: Path | None,
    prompt: str | None,
    state: str | None,
    images: list[Path],
    embodiment_id: int | None,
    noise_file: Path | None,
    output: Path | None,
    delay: int | None,
    action_prefix_file: Path | None,
) -> int:
    runner = picpp.build_pi06_rtc_runner(model_dir=model_dir)
    if case_dir is not None:
        tensors = {}
        for name, filename in PI06_RTC_CASE_TENSORS:
            path = Path(case_dir) / filename
            if not path.exists():
                raise ValueError(f"PI0.6 RTC case tensor {filename} missing in {case_dir}")
            tensors[name] = np.load(path)
        noise_path = noise_file if noise_file is not None else Path(case_dir) / "noise_seed.npy"
        if noise_path.exists():
            noise = np.load(noise_path)
            expected = (1, runner.action_horizon, runner.action_dim)
            if noise.shape == (runner.action_horizon, runner.action_dim):
                noise = noise[None]
            if noise.shape != expected:
                raise ValueError(f"noise file {noise_path} has shape {noise.shape}, expected {expected}")
            tensors["x_t"] = np.ascontiguousarray(noise, dtype=np.float32)
        elif "x_t" not in tensors:
            raise ValueError(f"case {case_dir} has no noise: pass --noise-file or add x_t")
        actions = runner.run_abi(tensors)
        metadata = dict(runner.metadata)
        metadata["case_dir"] = str(case_dir)
    else:
        if prompt is None or state is None or not images:
            raise ValueError(
                "PI0.6 RTC live inference needs --prompt, --state, and at least one --image "
                "(or use --case-dir for frozen-case replay)"
            )
        state_values = [float(part) for part in state.split(",")]
        if len(state_values) != runner.state_dim:
            raise ValueError(f"--state must contain {runner.state_dim} comma-separated floats, got {len(state_values)}")
        image_arrays = [np.asarray(_load_image(path), dtype=np.uint8) for path in images]
        if len(image_arrays) != runner.num_cameras:
            raise ValueError(
                f"PI0.6 RTC expects {runner.num_cameras} images in {runner.spec.camera_order}, got {len(image_arrays)}"
            )
        noise = None
        if noise_file is not None:
            noise = np.load(noise_file)
        action_prefix = None
        if action_prefix_file is not None:
            action_prefix = np.load(action_prefix_file)
            if action_prefix.shape == (runner.action_dim,):
                action_prefix = action_prefix[None]
        actions = runner.run_once(
            images=image_arrays,
            prompt=prompt,
            state=np.asarray(state_values, dtype=np.float32),
            embodiment_id=embodiment_id,
            noise=noise,
            delay=delay,
            action_prefix=action_prefix,
        )
        metadata = dict(runner.metadata)
        metadata["noise_mode"] = "explicit" if noise_file is not None else "seeded_stream"

    if output is not None:
        np.save(output, actions)
        metadata["output"] = str(output)
    console.print_json(
        data={
            "model": "pi06_rtc",
            "action_horizon": int(actions.shape[0]),
            "action_dim": int(actions.shape[1]),
            "actions": actions.tolist(),
            **metadata,
        }
    )
    return 0


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image)


def run_infer(
    *,
    model: str,
    model_dir: Path | None,
    case_dir: Path | None,
    prompt: str | None,
    state: str | None,
    images: list[Path],
    embodiment_id: int | None,
    noise_file: Path | None,
    output: Path | None,
    delay: int | None,
    action_prefix_file: Path | None,
) -> int:
    if model == "pi06_heterogeneous":
        return _run_pi06_heterogeneous_infer(
            model_dir=model_dir,
            case_dir=case_dir,
            prompt=prompt,
            state=state,
            images=images,
            embodiment_id=embodiment_id,
            noise_file=noise_file,
            output=output,
        )
    if model == "pi06_rtc":
        return _run_pi06_rtc_infer(
            model_dir=model_dir,
            case_dir=case_dir,
            prompt=prompt,
            state=state,
            images=images,
            embodiment_id=embodiment_id,
            noise_file=noise_file,
            output=output,
            delay=delay,
            action_prefix_file=action_prefix_file,
        )
    raise ValueError(f"unsupported infer model: {model}")
