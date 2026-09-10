"""PI0.6 RTC (real-time chunking) deployment contract over the split runner.

Same heterogeneous_3view backbone as pi06_heterogeneous (continuous
quantile-normalized state, [50,16] flow loop, canonical output layout), plus
the training-time-RTC suffix ABI: ``delay`` [1] int and ``action_prefix``
[1,50,16] float32 in the same quantile-normalized canonical space as x_t.
The executed prefix positions are pinned back to the prefix after every
denoising step (the runner applies the final pin on the host), and delay=0
reduces to the plain pi06_heterogeneous semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pi_cpp.pi06_heterogeneous import (
    DEFAULT_PI06_HETEROGENEOUS_MODEL_DIR,
    Pi06HeterogeneousRunnerWrapper,
    Pi06HeterogeneousSpec,
)

DEFAULT_PI06_RTC_MODEL_DIR = Path("assets/pi06_rtc")
# The cloth_*_airbot action layout: embodiment 0 (fold_cloth_v4) maps
# 6+1 joints/gripper into the canonical 16 slots (missing joint slot and
# right gripper slot zero-padded); embodiment 1 is identity. Same table as
# CLOTH_ACTION_INDEX_MAPS in training/config.py.
ACTION_INDEX_MAPS = (
    (0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10, 11, 12, -1, 13),
    tuple(range(16)),
)


class Pi06RtcRunnerWrapper(Pi06HeterogeneousRunnerWrapper):
    """pi06_heterogeneous + the RTC delay/action_prefix suffix inputs."""

    def __init__(self, model_dir: str | Path = DEFAULT_PI06_RTC_MODEL_DIR) -> None:
        super().__init__(model_dir=model_dir)

    @staticmethod
    def _load_spec(manifest: dict[str, Any]) -> Pi06HeterogeneousSpec:
        spec = Pi06HeterogeneousRunnerWrapper._load_spec(manifest)
        rtc = manifest.get("rtc") or {}
        if not rtc.get("training_time_rtc", False):
            raise ValueError("pi06_rtc requires manifest rtc.training_time_rtc=True")
        return spec

    @property
    def rtc_max_delay(self) -> int:
        return 4

    def _validate_input_shapes(self) -> None:
        super()._validate_input_shapes()
        expected = {
            "delay": [1],
            "action_prefix": [1, self.spec.action_horizon, self.spec.internal_action_dim],
        }
        actual = {name: list(self._input_shapes[name]) for name in expected}
        if actual != expected:
            raise ValueError(f"PI0.6 RTC input shapes do not match the fixed ABI: {actual}")

    def _map_action_prefix(self, values: np.ndarray, embodiment_id: int) -> np.ndarray:
        index_map = ACTION_INDEX_MAPS[int(embodiment_id)]
        mapped = np.zeros((values.shape[0], self.spec.internal_action_dim), dtype=np.float32)
        for out_index, source_index in enumerate(index_map):
            if source_index >= 0:
                mapped[:, out_index] = values[:, source_index]
        return mapped

    def _prepare_rtc_inputs(
        self,
        delay: Any,
        action_prefix: Any,
        embodiment_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map raw-space executed prefix -> the engine ABI (canonical,
        quantile-normalized, horizon-padded), mirroring the JAX policy.infer
        input transform (airbot map -> quantile normalize -> zero pad)."""
        if delay is None:
            delay_value = 0
        else:
            delay_value = int(np.asarray(delay).reshape(-1)[0])
        if not 0 <= delay_value <= self.rtc_max_delay:
            raise ValueError(f"delay {delay_value} outside the trained range [0, {self.rtc_max_delay}]")
        horizon = self.spec.action_horizon
        prefix = np.zeros((horizon, self.spec.internal_action_dim), dtype=np.float32)
        if delay_value > 0:
            if action_prefix is None:
                raise ValueError("pi06_rtc requires action_prefix when delay > 0")
            raw = np.asarray(action_prefix, dtype=np.float32)
            if raw.shape != (delay_value, self.spec.raw_action_dim):
                raise ValueError(
                    f"action_prefix must have shape [{delay_value}, {self.spec.raw_action_dim}], got {raw.shape}"
                )
            mapped = self._map_action_prefix(raw, embodiment_id)
            stats = self._norm_stats["actions"]
            prefix[:delay_value] = (mapped - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
        return (
            np.ascontiguousarray(np.asarray([delay_value], dtype=np.int64)),
            np.ascontiguousarray(prefix[None], dtype=np.float32),
        )

    def run_abi(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        """pi05_offline_v1 ABI + delay/action_prefix (engine-space values)."""
        required = (
            "image",
            "image_mask",
            "tokenized_prompt",
            "tokenized_prompt_mask",
            "x_t",
            "state",
            "embodiment_id",
        )
        missing = [name for name in required if name not in tensors]
        if missing:
            raise ValueError(f"PI0.6 RTC ABI replay is missing tensors: {missing}")
        resolved = {}
        for name in required:
            array = np.asarray(tensors[name])
            if array.dtype == np.float64:
                array = array.astype(np.float32)
            resolved[name] = np.ascontiguousarray(array)
        expected = {
            "image": [1, len(self.spec.camera_order), 3, *self.spec.image_size],
            "image_mask": [1, len(self.spec.camera_order)],
            "tokenized_prompt": [1, self.spec.token_length],
            "tokenized_prompt_mask": [1, self.spec.token_length],
            "x_t": [1, self.spec.action_horizon, self.spec.internal_action_dim],
            "state": [1, self.spec.raw_state_dim],
            "embodiment_id": [1],
        }
        for name, shape in expected.items():
            if list(resolved[name].shape) != shape:
                raise ValueError(
                    f"PI0.6 RTC ABI tensor {name} has shape {resolved[name].shape}, expected {shape}"
                )
        if "delay" in tensors and "action_prefix" in tensors:
            resolved["delay"] = np.ascontiguousarray(np.asarray(tensors["delay"], dtype=np.int64))
            resolved["action_prefix"] = np.ascontiguousarray(
                np.asarray(tensors["action_prefix"], dtype=np.float32)
            )
            if list(resolved["delay"].shape) != [1]:
                raise ValueError(f"delay has shape {resolved['delay'].shape}, expected [1]")
            if list(resolved["action_prefix"].shape) != [
                1,
                self.spec.action_horizon,
                self.spec.internal_action_dim,
            ]:
                raise ValueError(
                    f"action_prefix has shape {resolved['action_prefix'].shape}, expected [1,{self.spec.action_horizon},{self.spec.internal_action_dim}]"
                )
        else:
            resolved["delay"] = np.ascontiguousarray(np.zeros((1,), dtype=np.int64))
            resolved["action_prefix"] = np.ascontiguousarray(
                np.zeros((1, self.spec.action_horizon, self.spec.internal_action_dim), dtype=np.float32)
            )
        result = self._runner.run_once(resolved)
        actions = self._postprocess_actions(result.action)
        self.metadata = {
            "model": "pi06_rtc",
            "manifest_schema": self.manifest_schema,
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "camera_order": list(self.spec.camera_order),
            "internal_action_dim": self.spec.internal_action_dim,
            "token_length": self.spec.token_length,
            "denoise_steps": self.spec.denoise_steps,
            "dt": self.spec.dt,
            "action_semantics": self.spec.action_semantics,
            "state_input": "continuous",
            "embodiment_id": int(resolved["embodiment_id"][0]),
            "noise_mode": "explicit",
            "action_horizon": self.spec.action_horizon,
            "action_dim": self.spec.raw_action_dim,
            "delay": int(resolved["delay"][0]),
            "input_shapes": {name: list(array.shape) for name, array in resolved.items()},
            **result.to_dict(),
        }
        return actions

    def run_once(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        state: Any,
        embodiment_id: int | None = None,
        noise: Any | None = None,
        delay: Any | None = None,
        action_prefix: Any | None = None,
    ) -> np.ndarray:
        preprocess_start = __import__("time").perf_counter()
        resolved_embodiment = self._prepare_embodiment_id(embodiment_id)
        normalized_state = self._normalize_state(state, int(resolved_embodiment[0]))
        tokenized_prompt, tokenized_prompt_mask = self._prepare_tokenized_prompt(str(prompt))
        delay_tensor, prefix_tensor = self._prepare_rtc_inputs(
            delay, action_prefix, int(resolved_embodiment[0])
        )
        tensors = {
            "image": self._prepare_images(images),
            "image_mask": np.ones((1, self.num_cameras), dtype=np.bool_),
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            "x_t": self._prepare_noise(noise),
            "state": np.ascontiguousarray(normalized_state[None], dtype=np.float32),
            "embodiment_id": resolved_embodiment,
            "delay": delay_tensor,
            "action_prefix": prefix_tensor,
        }
        preprocess_ms = (__import__("time").perf_counter() - preprocess_start) * 1000.0
        result = self._runner.run_once(tensors)
        actions = self._postprocess_actions(result.action)
        self.metadata = {
            "model": "pi06_rtc",
            "manifest_schema": self.manifest_schema,
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "camera_order": list(self.spec.camera_order),
            "internal_action_dim": self.spec.internal_action_dim,
            "token_length": self.spec.token_length,
            "denoise_steps": self.spec.denoise_steps,
            "dt": self.spec.dt,
            "action_semantics": self.spec.action_semantics,
            "state_input": "continuous",
            "embodiment_id": int(tensors["embodiment_id"][0]),
            "noise_mode": "explicit" if noise is not None else "seeded_stream",
            "action_horizon": self.spec.action_horizon,
            "action_dim": self.spec.raw_action_dim,
            "delay": int(tensors["delay"][0]),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            **result.to_dict(),
        }
        return actions


def build_pi06_rtc_runner(
    *, model_dir: str | Path | None = None
) -> Pi06RtcRunnerWrapper:
    return Pi06RtcRunnerWrapper(
        model_dir=DEFAULT_PI06_RTC_MODEL_DIR if model_dir is None else Path(model_dir)
    )
