"""PI0.6 Airbot deployment contract over the existing split TensorRT runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import sentencepiece
from PIL import Image

import pi_cpp as picpp

DEFAULT_PI06_AIRBOT_MODEL_DIR = Path("assets/pi06_airbot")
PI06_AIRBOT_ENGINE_DIR = "engines"
PI06_AIRBOT_MANIFEST_PATH = "export_manifest.json"
PI06_AIRBOT_MANIFEST_SCHEMA = "pi_cpp.pi06_airbot.v1"
PI06_AIRBOT_CAMERA_ORDER = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass(frozen=True)
class Pi06AirbotSpec:
    camera_order: tuple[str, ...]
    image_size: tuple[int, int]
    raw_state_dim: int
    raw_action_dim: int
    internal_action_dim: int
    token_length: int
    action_horizon: int
    denoise_steps: int
    dt: float
    action_semantics: str
    advantage_conditioning: bool
    default_advantage: bool
    noise_seed: int
    tokenizer_path: str
    norm_stats_path: str


@dataclass(frozen=True)
class _QuantileStats:
    q01: np.ndarray
    q99: np.ndarray


def _as_uint8_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError(
            f"Airbot images must be HWC uint8 RGB arrays, got {array.shape} {array.dtype}"
        )
    return np.ascontiguousarray(array)


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    ratio = max(source_width / width, source_height / height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    padded[top : top + resized_height, left : left + resized_width] = np.asarray(
        resized, dtype=np.uint8
    )
    return padded


class Pi06AirbotRunnerWrapper:
    def __init__(self, model_dir: str | Path = DEFAULT_PI06_AIRBOT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.manifest_schema = PI06_AIRBOT_MANIFEST_SCHEMA
        manifest = json.loads((self.model_dir / PI06_AIRBOT_MANIFEST_PATH).read_text())
        self.spec = self._load_spec(manifest)
        self.engine_dir = self.model_dir / PI06_AIRBOT_ENGINE_DIR
        self._norm_stats = self._load_norm_stats()
        self._tokenizer: sentencepiece.SentencePieceProcessor | None = None
        self._rng = np.random.default_rng(self.spec.noise_seed)
        self._runner = picpp.Pi05OfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self._validate_input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @staticmethod
    def _load_spec(manifest: dict[str, Any]) -> Pi06AirbotSpec:
        runtime = manifest["runtime"]
        contract = manifest["contract"]
        camera_order = tuple(contract["camera_order"])
        image_size = tuple(int(value) for value in contract["image_size"])
        normalization = manifest["normalization"]
        action = manifest["action"]
        sampling = manifest["sampling"]
        raw_state_dim = int(contract["raw_state_dim"])
        raw_action_dim = int(contract["raw_action_dim"])
        internal_action_dim = int(action["dim"])
        token_length = int(contract["token_length"])
        action_horizon = int(action["horizon"])
        fixed_contract = (
            manifest["schema"],
            manifest["model_family"],
            runtime["runner"],
            runtime["abi"],
            camera_order,
            image_size,
            normalization["method"],
            normalization["state_key"],
            normalization["action_key"],
            raw_state_dim,
            raw_action_dim,
            internal_action_dim,
            token_length,
            action_horizon,
            contract["advantage_conditioning"],
            contract["default_advantage"],
            contract["action_semantics"],
            int(sampling["steps"]),
            float(sampling["dt"]),
        )
        expected_contract = (
            PI06_AIRBOT_MANIFEST_SCHEMA,
            "pi06",
            "Pi05OfflineRunner",
            "pi05_offline_v1",
            PI06_AIRBOT_CAMERA_ORDER,
            (224, 224),
            "quantile",
            "state",
            "actions",
            14,
            14,
            32,
            200,
            50,
            True,
            True,
            "absolute",
            10,
            -0.1,
        )
        if fixed_contract != expected_contract:
            raise ValueError(f"unsupported PI0.6 Airbot contract: {fixed_contract}")

        assets = manifest["assets"]
        return Pi06AirbotSpec(
            camera_order=camera_order,
            image_size=(image_size[0], image_size[1]),
            raw_state_dim=raw_state_dim,
            raw_action_dim=raw_action_dim,
            internal_action_dim=internal_action_dim,
            token_length=token_length,
            action_horizon=action_horizon,
            denoise_steps=int(sampling["steps"]),
            dt=float(sampling["dt"]),
            action_semantics=str(contract["action_semantics"]),
            advantage_conditioning=bool(contract["advantage_conditioning"]),
            default_advantage=bool(contract["default_advantage"]),
            noise_seed=int(sampling["noise_seed"]),
            tokenizer_path=str(assets["tokenizer"]),
            norm_stats_path=str(assets["norm_stats"]),
        )

    def _load_norm_stats(self) -> dict[str, _QuantileStats]:
        payload = json.loads((self.model_dir / self.spec.norm_stats_path).read_text())["norm_stats"]
        stats = {
            key: _QuantileStats(
                q01=np.asarray(payload[key]["q01"], dtype=np.float32),
                q99=np.asarray(payload[key]["q99"], dtype=np.float32),
            )
            for key in ("state", "actions")
        }
        for key, value in stats.items():
            raw_dim = self.spec.raw_state_dim if key == "state" else self.spec.raw_action_dim
            if (
                value.q01.shape != (self.spec.internal_action_dim,)
                or value.q99.shape != value.q01.shape
                or not np.isfinite(value.q01).all()
                or not np.isfinite(value.q99).all()
                or np.any(value.q99 < value.q01)
                or not np.allclose(value.q01[raw_dim:], 0.0)
                or not np.allclose(value.q99[raw_dim:], 0.0)
            ):
                raise ValueError(f"invalid {key} quantile stats")
        return stats

    def _validate_input_shapes(self) -> None:
        height, width = self.spec.image_size
        expected = {
            "image": [1, len(self.spec.camera_order), 3, height, width],
            "image_mask": [1, len(self.spec.camera_order)],
            "tokenized_prompt": [1, self.spec.token_length],
            "tokenized_prompt_mask": [1, self.spec.token_length],
            "x_t": [1, self.spec.action_horizon, self.spec.internal_action_dim],
        }
        actual = {name: list(self._input_shapes[name]) for name in expected}
        if actual != expected:
            raise ValueError(f"PI0.6 Airbot input shapes do not match the fixed ABI: {actual}")

    @property
    def num_cameras(self) -> int:
        return len(self.spec.camera_order)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.raw_state_dim

    @property
    def action_dim(self) -> int:
        return self.spec.raw_action_dim

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    def _load_tokenizer(self) -> sentencepiece.SentencePieceProcessor:
        if self._tokenizer is None:
            self._tokenizer = sentencepiece.SentencePieceProcessor()
            self._tokenizer.Load(str(self.model_dir / self.spec.tokenizer_path))
        return self._tokenizer

    def _prepare_images(self, images: Sequence[Any]) -> np.ndarray:
        if len(images) != self.num_cameras:
            raise ValueError(
                f"PI0.6 Airbot expects {self.num_cameras} images in {self.spec.camera_order}"
            )
        height, width = self.image_size
        prepared = []
        for image in images:
            resized = _resize_with_pad(_as_uint8_rgb(image), height, width)
            prepared.append(np.moveaxis(resized.astype(np.float32) / 255.0 * 2.0 - 1.0, -1, 0))
        return np.ascontiguousarray(np.stack(prepared, axis=0)[None], dtype=np.float32)

    def _normalize_state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(self.spec.raw_state_dim)
        if not np.isfinite(values).all():
            raise ValueError("PI0.6 Airbot state contains non-finite values")
        padded = np.zeros(self.spec.internal_action_dim, dtype=np.float32)
        padded[: self.spec.raw_state_dim] = values
        stats = self._norm_stats["state"]
        return (padded - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0

    def _prepare_tokenized_prompt(
        self, prompt: str, state: np.ndarray, advantage: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1.0, 1.0, 257)[:-1]) - 1
        state_text = " ".join(map(str, discretized_state.astype(np.int64).tolist()))
        advantage_text = "Positive" if advantage else "Negative"
        full_prompt = (
            f"Task: {cleaned_text}, State: {state_text}, Advantage: {advantage_text};\nAction: "
        )
        tokenizer = self._load_tokenizer()
        tokens = [tokenizer.bos_id(), *tokenizer.EncodeAsIds(full_prompt)]
        tokens = tokens[: self.spec.token_length]
        mask = [True] * len(tokens)
        padding = self.spec.token_length - len(tokens)
        tokens.extend([0] * padding)
        mask.extend([False] * padding)
        return (
            np.ascontiguousarray(np.asarray(tokens, dtype=np.int64)[None]),
            np.ascontiguousarray(np.asarray(mask, dtype=np.bool_)[None]),
        )

    def _prepare_noise(self, noise: Any | None) -> np.ndarray:
        if noise is None:
            return np.ascontiguousarray(
                self._rng.standard_normal(
                    (1, self.spec.action_horizon, self.spec.internal_action_dim), dtype=np.float32
                )
            )
        values = np.asarray(noise, dtype=np.float32)
        expected = (1, self.spec.action_horizon, self.spec.internal_action_dim)
        if values.shape != expected:
            raise ValueError(f"PI0.6 Airbot noise has shape {values.shape}, expected {expected}")
        if not np.isfinite(values).all():
            raise ValueError("PI0.6 Airbot noise contains non-finite values")
        return np.ascontiguousarray(values)

    def _postprocess_actions(self, action: Any) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        expected = (self.spec.action_horizon, self.spec.internal_action_dim)
        if normalized.size != int(np.prod(expected)):
            raise ValueError(
                f"PI0.6 Airbot output has {normalized.size} values, expected {np.prod(expected)}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError("PI0.6 Airbot output contains non-finite values")
        normalized = normalized.reshape(expected)
        stats = self._norm_stats["actions"]
        actions = (normalized + 1.0) * 0.5 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
        return np.ascontiguousarray(actions[:, : self.spec.raw_action_dim], dtype=np.float32)

    def run_once(
        self,
        *,
        images: Sequence[Any],
        prompt: str,
        state: Any,
        advantage: bool | None = None,
        noise: Any | None = None,
    ) -> np.ndarray:
        preprocess_start = perf_counter()
        normalized_state = self._normalize_state(state)
        tokenized_prompt, tokenized_prompt_mask = self._prepare_tokenized_prompt(
            str(prompt),
            normalized_state,
            self.spec.default_advantage if advantage is None else bool(advantage),
        )
        tensors = {
            "image": self._prepare_images(images),
            "image_mask": np.ones((1, self.num_cameras), dtype=np.bool_),
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            "x_t": self._prepare_noise(noise),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0
        result = self._runner.run_once(tensors)
        postprocess_start = perf_counter()
        actions = self._postprocess_actions(result.action)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "pi06_airbot",
            "manifest_schema": self.manifest_schema,
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "camera_order": list(self.spec.camera_order),
            "internal_action_dim": self.spec.internal_action_dim,
            "token_length": self.spec.token_length,
            "denoise_steps": self.spec.denoise_steps,
            "dt": self.spec.dt,
            "action_semantics": self.spec.action_semantics,
            "advantage_conditioning": self.spec.advantage_conditioning,
            "default_advantage": self.spec.default_advantage,
            "noise_mode": "explicit" if noise is not None else "seeded_stream",
            "action_horizon": self.spec.action_horizon,
            "action_dim": self.spec.raw_action_dim,
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


def build_pi06_airbot_runner(*, model_dir: str | Path | None = None) -> Pi06AirbotRunnerWrapper:
    return Pi06AirbotRunnerWrapper(
        model_dir=DEFAULT_PI06_AIRBOT_MODEL_DIR if model_dir is None else Path(model_dir)
    )
