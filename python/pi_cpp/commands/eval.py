"""Eval command."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import track

import pi_cpp as picpp
from pi_cpp.eval.gt import default_dataset_root
from pi_cpp.eval.metrics import action_metrics

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
TASK_INDEX_KEY = "task_index"
ACTION_KEY = "action"
PI05_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
PI05_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
FASTWAM_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
FASTWAM_ROBOTWIN_IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
SEMANTICVLA_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
SEMANTICVLA_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
SMOLVLA_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
SMOLVLA_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
DIT4DIT_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
DIT4DIT_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
GROOT_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
GROOT_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
STARVLA_LIBERO_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
STARVLA_LIBERO_IMAGE2_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)

console = Console()


LIBERO_V21_DATASETS = (
    ("aopolin-lv/libero_spatial_no_noops_lerobot_v21", "v2.1"),
    ("aopolin-lv/libero_object_no_noops_lerobot_v21", "v2.1"),
    ("aopolin-lv/libero_goal_no_noops_lerobot_v21", "v2.1"),
    ("aopolin-lv/libero_10_no_noops_lerobot_v21", "v2.1"),
)


def _load_dataset(repo_id: str, revision: str, episodes: list[int] | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=default_dataset_root(repo_id),
        episodes=episodes,
        revision=revision,
        download_videos=True,
        video_backend="pyav",
    )


def _load_eval_datasets(dataset: str):
    specs = LIBERO_V21_DATASETS if dataset == "lerobot/libero" else ((dataset, "main"),)
    datasets = []
    for repo_id, revision in specs:
        episodes = list(range(100)) if dataset == "lerobot/libero" else None
        lerobot_dataset = _load_dataset(repo_id, revision, episodes=episodes)
        episode_starts, episode_ends = _episode_ranges(lerobot_dataset)
        if dataset == "lerobot/libero":
            episode_starts = episode_starts[:100]
            episode_ends = episode_ends[:100]
        datasets.append((lerobot_dataset, episode_starts, episode_ends))
    return datasets


def _eval_episodes(eval_datasets):
    episode_index = 0
    for lerobot_dataset, episode_starts, episode_ends in eval_datasets:
        for episode_start, episode_end in track(
            zip(episode_starts, episode_ends, strict=True),
            total=len(episode_starts),
            description=f"eval {lerobot_dataset.repo_id}",
        ):
            yield episode_index, lerobot_dataset, int(episode_start), int(episode_end)
            episode_index += 1


def _episode_ranges(lerobot_dataset) -> tuple[list[int], list[int]]:
    if hasattr(lerobot_dataset, "episode_data_index"):
        return (
            lerobot_dataset.episode_data_index["from"].tolist(),
            lerobot_dataset.episode_data_index["to"].tolist(),
        )

    starts = []
    ends = []
    path = default_dataset_root(lerobot_dataset.repo_id) / "meta" / "episodes.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        starts.append(int(row["dataset_from_index"]))
        ends.append(int(row["dataset_to_index"]))
    return starts, ends


def _task_text(lerobot_dataset, item: dict) -> str:
    if "task" in item:
        return str(item["task"]).strip()

    task_index = int(item[TASK_INDEX_KEY])
    tasks = lerobot_dataset.meta.tasks
    if hasattr(tasks, "index") and hasattr(tasks, "columns") and "task_index" in tasks.columns:
        return str(tasks.index[tasks["task_index"] == task_index][0]).strip()
    return str(tasks[task_index]).strip()


def _print_eval_progress(
    *,
    enabled: bool,
    model: str,
    dataset: str,
    episode_index: int,
    episodes: int,
    episode_samples: int,
    samples: int,
    totals: dict[str, float],
) -> None:
    if not enabled:
        return
    console.print_json(
        data={
            "event": "eval_episode_result",
            "model": model,
            "dataset": dataset,
            "episode_index": episode_index,
            "episodes": episodes,
            "episodes_done": episode_index + 1,
            "episode_samples": episode_samples,
            "samples": samples,
            "mse": totals["mse"] / samples if samples else 0.0,
            "l1": totals["l1"] / samples if samples else 0.0,
        }
    )


def _print_eval_start(
    *,
    enabled: bool,
    model: str,
    dataset: str,
    episodes: int,
    action_horizon: int,
    action_steps: int | None = None,
) -> None:
    if not enabled:
        return
    data: dict[str, object] = {
        "event": "eval_start",
        "model": model,
        "dataset": dataset,
        "episodes": episodes,
        "action_horizon": action_horizon,
    }
    if action_steps is not None:
        data["action_steps"] = action_steps
    console.print_json(data=data)


def _run_pi05_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_pi05_runner(model_dir=model_dir)
    action_horizon = runner.spec.action_horizon
    action_steps = runner.spec.action_steps
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="pi05",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
        action_steps=action_steps,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            PI05_LIBERO_IMAGE_KEYS
            if PI05_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else PI05_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_steps
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="pi05",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [
                lerobot_dataset.hf_dataset[index + offset] for offset in range(action_steps)
            ]
            item = lerobot_dataset[index]
            images = [item[key] for key in image_keys]
            state = np.ascontiguousarray(np.asarray(item[STATE_KEY]), dtype=np.float32)
            prompt = _task_text(lerobot_dataset, item)
            deploy_action = runner.run_once(images=images, prompt=prompt, state=state)[:action_steps]
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="pi05",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "pi05",
            "dataset": dataset,
            "action_horizon": action_horizon,
            "action_steps": action_steps,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_fastwam_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_fastwam_runner(model_dir=model_dir)
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="fastwam",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = FASTWAM_LIBERO_IMAGE_KEYS if runner.num_cameras == 2 else FASTWAM_ROBOTWIN_IMAGE_KEYS
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="fastwam",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            state = np.ascontiguousarray(np.asarray(item[STATE_KEY]), dtype=np.float32)
            prompt = _task_text(lerobot_dataset, item)
            deploy_action = runner.run_once(images=images, prompt=prompt, state=state)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="fastwam",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "fastwam",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_semanticvla_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_semanticvla_runner(model_dir=model_dir)
    prompt_by_task = {
        " ".join(str(row["task"]).strip().lower().split()): str(row["task"])
        for row in runner.prompt_cache_index
    }
    action_min = np.asarray(eval_datasets[0][0].meta.stats[ACTION_KEY]["min"], dtype=np.float32).reshape(-1)
    gt_gripper_is_signed = bool(action_min[-1] < 0.0)
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="semanticvla",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            SEMANTICVLA_LIBERO_IMAGE_KEYS
            if SEMANTICVLA_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else SEMANTICVLA_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="semanticvla",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            task = _task_text(lerobot_dataset, item)
            prompt = prompt_by_task[" ".join(task.strip().lower().split())]
            deploy_action = runner.run_once(images=images, prompt=prompt, state=None)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            if gt_gripper_is_signed:
                deploy_action = deploy_action.copy()
                deploy_action[:, -1] = 1.0 - 2.0 * (deploy_action[:, -1] > 0.5)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="semanticvla",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "semanticvla",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_smolvla_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_smolvla_runner(model_dir=model_dir)
    prompt_by_task = {
        " ".join(str(row["task"]).strip().lower().split()): str(row["task"])
        for row in runner.prompt_cache_index
    }
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="smolvla",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            SMOLVLA_LIBERO_IMAGE_KEYS
            if SMOLVLA_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else SMOLVLA_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="smolvla",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            state = np.ascontiguousarray(np.asarray(item[STATE_KEY]), dtype=np.float32)
            task = _task_text(lerobot_dataset, item)
            prompt = prompt_by_task[" ".join(task.strip().lower().split())]
            deploy_action = runner.run_once(images=images, prompt=prompt, state=state)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="smolvla",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "smolvla",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_dit4dit_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_dit4dit_runner(model_dir=model_dir)
    prompt_by_task = {
        " ".join(str(row["task"]).strip().lower().split()): str(row["task"])
        for row in runner.prompt_cache_index
    }
    action_min = np.asarray(eval_datasets[0][0].meta.stats[ACTION_KEY]["min"], dtype=np.float32).reshape(-1)
    gt_gripper_is_signed = bool(action_min[-1] < 0.0)
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="dit4dit",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            DIT4DIT_LIBERO_IMAGE_KEYS
            if DIT4DIT_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else DIT4DIT_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="dit4dit",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            state = np.ascontiguousarray(np.asarray(item[STATE_KEY]), dtype=np.float32)
            task = _task_text(lerobot_dataset, item)
            prompt = prompt_by_task[" ".join(task.strip().lower().split())]
            deploy_action = runner.run_once(images=images, prompt=prompt, state=state)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            if gt_gripper_is_signed:
                deploy_action = deploy_action.copy()
                deploy_action[:, -1] = 1.0 - 2.0 * (deploy_action[:, -1] > 0.5)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="dit4dit",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "dit4dit",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_groot_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_groot_runner(model_dir=model_dir)
    action_min = np.asarray(eval_datasets[0][0].meta.stats[ACTION_KEY]["min"], dtype=np.float32).reshape(-1)
    gt_gripper_is_signed = bool(action_min[-1] < 0.0)
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="groot",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            GROOT_LIBERO_IMAGE_KEYS
            if GROOT_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else GROOT_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="groot",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            state = np.ascontiguousarray(np.asarray(item[STATE_KEY]), dtype=np.float32)
            task = _task_text(lerobot_dataset, item)
            deploy_action = runner.run_once(images=images, prompt=task, state=state)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            if gt_gripper_is_signed:
                deploy_action = deploy_action.copy()
                deploy_action[:, -1] = 1.0 - 2.0 * (deploy_action[:, -1] > 0.5)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="groot",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "groot",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def _run_starvla_eval(*, dataset: str, model_dir: Path | None, progress: bool) -> int:
    eval_datasets = _load_eval_datasets(dataset)
    runner = picpp.build_starvla_runner(model_dir=model_dir)
    prompt_by_task = {
        " ".join(str(row["task"]).strip().lower().split()): str(row["task"])
        for row in runner.prompt_cache_index
    }
    action_min = np.asarray(eval_datasets[0][0].meta.stats[ACTION_KEY]["min"], dtype=np.float32).reshape(-1)
    gt_gripper_is_signed = bool(action_min[-1] < 0.0)
    action_horizon = runner.action_horizon
    episodes = sum(len(episode_starts) for _, episode_starts, _ in eval_datasets)
    totals = {"mse": 0.0, "l1": 0.0}
    samples = 0
    _print_eval_start(
        enabled=progress,
        model="starvla",
        dataset=dataset,
        episodes=episodes,
        action_horizon=action_horizon,
    )

    for episode_index, lerobot_dataset, first_frame, episode_end in _eval_episodes(eval_datasets):
        image_keys = (
            STARVLA_LIBERO_IMAGE_KEYS
            if STARVLA_LIBERO_IMAGE_KEYS[1] in lerobot_dataset.features
            else STARVLA_LIBERO_IMAGE2_KEYS
        )
        last_frame = episode_end - action_horizon
        episode_samples = 0
        if last_frame < first_frame:
            _print_eval_progress(
                enabled=progress,
                model="starvla",
                dataset=lerobot_dataset.repo_id,
                episode_index=episode_index,
                episodes=episodes,
                episode_samples=episode_samples,
                samples=samples,
                totals=totals,
            )
            continue

        for index in range(first_frame, last_frame + 1):
            chunk_items = [lerobot_dataset[index + offset] for offset in range(action_horizon)]
            item = chunk_items[0]
            images = [item[key] for key in image_keys]
            task = _task_text(lerobot_dataset, item)
            prompt = prompt_by_task[" ".join(task.strip().lower().split())]
            deploy_action = runner.run_once(images=images, prompt=prompt)
            gt_action = [
                np.asarray(item[ACTION_KEY], dtype=np.float32).reshape(-1)
                for item in chunk_items[: deploy_action.shape[0]]
            ]
            gt_action = np.asarray(gt_action, dtype=np.float32)
            if gt_gripper_is_signed:
                deploy_action = deploy_action.copy()
                deploy_action[:, -1] = 1.0 - 2.0 * (deploy_action[:, -1] > 0.5)
            metrics = action_metrics(deploy_action[:, : gt_action.shape[1]], gt_action)
            totals["mse"] += float(metrics["mse"])
            totals["l1"] += float(metrics["mae"])
            samples += 1
            episode_samples += 1

        _print_eval_progress(
            enabled=progress,
            model="starvla",
            dataset=lerobot_dataset.repo_id,
            episode_index=episode_index,
            episodes=episodes,
            episode_samples=episode_samples,
            samples=samples,
            totals=totals,
        )

    console.print_json(
        data={
            "model": "starvla",
            "dataset": dataset,
            "samples": samples,
            "mse": totals["mse"] / samples,
            "l1": totals["l1"] / samples,
        }
    )
    return 0


def run_eval(*, model: str, dataset: str, model_dir: Path | None) -> int:
    progress = os.environ.get("PICPP_EVAL_PROGRESS") == "1"
    if model == "pi05":
        return _run_pi05_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "fastwam":
        return _run_fastwam_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "semanticvla":
        return _run_semanticvla_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "smolvla":
        return _run_smolvla_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "dit4dit":
        return _run_dit4dit_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "groot":
        return _run_groot_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    if model == "starvla":
        return _run_starvla_eval(dataset=dataset, model_dir=model_dir, progress=progress)
    raise ValueError(f"unsupported eval model: {model}")
