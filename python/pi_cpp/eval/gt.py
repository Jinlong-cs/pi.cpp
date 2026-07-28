"""LeRobot ground-truth action loading for offline eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def default_dataset_root(repo_id: str) -> Path:
    import os

    data_dir = Path(os.environ.get("PI_CPP_DATA_DIR", Path.cwd() / "data")).expanduser()
    return data_dir / "raw" / repo_slug(repo_id)


def load_lerobot_dataset(
    *,
    repo_id: str,
    episode_index: int,
    revision: str,
    video_backend: str,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id,
        root=default_dataset_root(repo_id),
        episodes=[episode_index],
        revision=revision,
        download_videos=True,
        video_backend=video_backend,
    )


def item_action(item: dict[str, Any], *, action_key: str = "action") -> np.ndarray:
    value = item[action_key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def action_chunk_from_dataset(
    dataset: Any,
    *,
    frame_index: int,
    horizon: int,
    action_dim: int,
    action_key: str = "action",
) -> np.ndarray:
    if frame_index + horizon > len(dataset):
        raise ValueError(
            f"GT action chunk crosses episode boundary: frame={frame_index}, "
            f"horizon={horizon}, length={len(dataset)}"
        )
    source = getattr(dataset, "hf_dataset", dataset)
    actions = [
        item_action(source[frame_index + offset], action_key=action_key)[:action_dim]
        for offset in range(horizon)
    ]
    return np.ascontiguousarray(np.stack(actions, axis=0).astype(np.float32))


def sample_frame_indices(*, dataset_length: int, horizon: int, count: int, start: int, stride: int) -> list[int]:
    last_start = dataset_length - horizon
    if last_start < start:
        raise ValueError(
            f"dataset episode is too short for horizon={horizon}: length={dataset_length}, start={start}"
        )
    if stride <= 0:
        available = last_start - start + 1
        if count >= available:
            return list(range(start, last_start + 1))
        if count == 1:
            return [start]
        span = last_start - start
        return [start + round(index * span / (count - 1)) for index in range(count)]

    indices: list[int] = []
    frame = start
    while frame <= last_start and len(indices) < count:
        indices.append(frame)
        frame += stride
    if not indices:
        raise ValueError("no eval samples selected")
    return indices
