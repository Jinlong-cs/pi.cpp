"""Action chunk metrics for offline pi.cpp evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_action_chunk(value: Any) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2:
        raise ValueError(f"action chunk must have shape [H,D] or [1,H,D], got {action.shape}")
    return np.ascontiguousarray(action)


def align_action_chunks(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = as_action_chunk(prediction)
    gt = as_action_chunk(target)
    horizon = min(pred.shape[0], gt.shape[0])
    action_dim = min(pred.shape[1], gt.shape[1])
    return pred[:horizon, :action_dim], gt[:horizon, :action_dim]


def action_metrics(prediction: Any, target: Any) -> dict[str, float | list[int]]:
    pred, gt = align_action_chunks(prediction, target)
    diff = pred - gt
    abs_diff = np.abs(diff)
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(abs_diff)),
        "max_abs": float(np.max(abs_diff)),
        "shape": [int(pred.shape[0]), int(pred.shape[1])],
    }

