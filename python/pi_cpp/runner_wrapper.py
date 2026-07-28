"""Small Python runner wrappers over the native pi.cpp runners."""

from __future__ import annotations
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pi_cpp as picpp
from PIL import Image
import sentencepiece

DEFAULT_FASTWAM_MODEL_DIR = Path("assets/fastwam_libero")
DEFAULT_DIT4DIT_MODEL_DIR = Path("assets/dit4dit_libero")
DEFAULT_PI05_MODEL_DIR = Path("assets/pi05_libero")
DEFAULT_SEMANTICVLA_MODEL_DIR = Path("assets/semanticvla_libero")
DEFAULT_EVO1_MODEL_DIR = Path("assets/evo1_libero")
DEFAULT_SMOLVLA_MODEL_DIR = Path("assets/smolvla_libero")
DEFAULT_GROOT_MODEL_DIR = Path("assets/groot_n1d6_libero")
DEFAULT_STARVLA_MODEL_DIR = Path("assets/starvla_libero")
FASTWAM_ENGINE_DIR = "engines"
FASTWAM_MANIFEST_PATH = "export_manifest.json"
FASTWAM_DATASET_STATS = "dataset_stats.json"
FASTWAM_PROPRIO_ENCODER = "proprio_encoder.pt"
FASTWAM_TEXT_CACHE_DIR = "text_embeds_cache/libero"
FASTWAM_TEXT_CACHE_ENCODER_ID = "wan22ti2v5b"
FASTWAM_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
FASTWAM_SEED = 1234
DIT4DIT_ENGINE_DIR = "engines"
DIT4DIT_MANIFEST_PATH = "export_manifest.json"
DIT4DIT_DATASET_STATS_PATH = "dataset_statistics.json"
DIT4DIT_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
DIT4DIT_INITIAL_ACTIONS_PATH = "initial_actions.npz"
DIT4DIT_CONSTANTS_PATH = "cosmos_constants.npz"
DIT4DIT_ACTION_STATS_KEY = "franka"
PI05_ENGINE_DIR = "engines"
PI05_TOKENIZER_PATH = "tokenizer.model"
PI05_NORM_STATS_PATH = "norm_stats.json"
PI05_STATE_STATS_KEY = "state"
PI05_ACTION_STATS_KEY = "actions"
PI05_MANIFEST_PATH = "export_manifest.json"
PI05_SUFFIX_IO_PATH = "io/suffix_step_io.json"
SEMANTICVLA_ENGINE_DIR = "engines"
SEMANTICVLA_MANIFEST_PATH = "export_manifest.json"
SEMANTICVLA_DATASET_STATS_PATH = "dataset_statistics.json"
SEMANTICVLA_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
SEMANTICVLA_ACTION_STATS_KEY = "franka"
SEMANTICVLA_INITIAL_ACTIONS_PATH = "initial_actions.npz"
EVO1_ENGINE_DIR = "engines"
EVO1_MANIFEST_PATH = "export_manifest.json"
EVO1_NORM_STATS_PATH = "norm_stats.json"
EVO1_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
EVO1_STATS_KEY = "libero_robot"
SMOLVLA_ENGINE_DIR = "engines"
SMOLVLA_MANIFEST_PATH = "export_manifest.json"
SMOLVLA_DATASET_STATS_PATH = "dataset_stats.json"
SMOLVLA_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
SMOLVLA_STATE_STATS_KEY = "observation.state"
SMOLVLA_ACTION_STATS_KEY = "action"
SMOLVLA_SEED = 1234
GROOT_ENGINE_DIR = "engines"
GROOT_MANIFEST_PATH = "export_manifest.json"
GROOT_BACKBONE_INPUTS_PATH = "io/backbone_inputs.npz"
GROOT_ACTION_LOOP_INPUTS_PATH = "io/action_loop_inputs.npz"
GROOT_DATASET_STATS_PATH = "dataset_statistics.json"
GROOT_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
STARVLA_ENGINE_DIR = "engines"
STARVLA_MANIFEST_PATH = "export_manifest.json"
STARVLA_DATASET_STATS_PATH = "dataset_statistics.json"
STARVLA_PROMPT_CACHE_INDEX_PATH = "prompt_cache_index.json"
STARVLA_ACTION_STATS_KEY = "franka"


@dataclass(frozen=True)
class GrootModelSpec:
    num_cameras: int
    image_size: tuple[int, int]
    state_dim: int
    max_state_dim: int
    token_length: int
    action_horizon: int
    action_dim: int
    action_latent_horizon: int
    action_latent_dim: int


@dataclass
class GrootRunnerWrapper:
    model_dir: Path = DEFAULT_GROOT_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / GROOT_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / GROOT_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads((self.model_dir / GROOT_PROMPT_CACHE_INDEX_PATH).read_text())
        self.spec = self._load_spec()
        self._stats = json.loads((self.model_dir / GROOT_DATASET_STATS_PATH).read_text())
        with np.load(self.model_dir / GROOT_BACKBONE_INPUTS_PATH) as inputs:
            self.backbone_inputs = {name: np.ascontiguousarray(inputs[name]) for name in inputs.files}
        with np.load(self.model_dir / GROOT_ACTION_LOOP_INPUTS_PATH) as inputs:
            self.action_loop_inputs = {name: np.ascontiguousarray(inputs[name]) for name in inputs.files}
        self._runner = picpp.GrootOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> GrootModelSpec:
        inputs = self.manifest["inputs"]
        action = self.manifest["action"]
        image_size = tuple(int(value) for value in inputs["image_size"])
        return GrootModelSpec(
            num_cameras=int(inputs["num_cameras"]),
            image_size=(image_size[0], image_size[1]),
            state_dim=int(inputs["state_dim"]),
            max_state_dim=int(inputs["max_state_dim"]),
            token_length=int(inputs["token_length"]),
            action_horizon=int(action["horizon"]),
            action_dim=int(action["dim"]),
            action_latent_horizon=int(action["latent_horizon"]),
            action_latent_dim=int(action["latent_dim"]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else " ".join(prompt.strip().lower().split())
        task = task.replace("akita black bowl", "black bowl")
        task = task.replace("cookies box", "cookie box")
        task = task.replace("top layer of the wooden cabinet", "top drawer of the wooden cabinet")
        task = task.replace("middle layer of the drawer", "middle drawer of the cabinet")
        task = task.replace("top layer of the drawer", "top drawer")
        task = task.replace("on the top of the drawer", "on top of the cabinet")
        task = task.replace("cream cheese on the bowl", "cream cheese in the bowl")
        if task.startswith("pick the "):
            task = "pick up the " + task.removeprefix("pick the ")
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(self.model_dir / self.manifest["assets"]["prompt_cache"] / f"{digest}.npz")
        self.metadata.update({"prompt": task, "prompt_cache_sha256": digest})
        tensors = {name: np.ascontiguousarray(cache[name]) for name in cache.files}
        for name in ("input_ids", "attention_mask"):
            values = tensors[name]
            padded = np.zeros((values.shape[0], self.spec.token_length), dtype=values.dtype)
            padded[:, : values.shape[1]] = values
            tensors[name] = np.ascontiguousarray(padded)
        return tensors

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        shortest = int(self.manifest["inputs"]["shortest_image_edge"])
        crop_fraction = float(self.manifest["inputs"]["crop_fraction"])
        source_height, source_width = image.shape[:2]
        square = max(source_height, source_width)
        padded = np.zeros((square, square, 3), dtype=np.uint8)
        top = (square - source_height) // 2
        left = (square - source_width) // 2
        padded[top : top + source_height, left : left + source_width] = image
        pil_image = Image.fromarray(padded, mode="RGB").resize((shortest, shortest), Image.Resampling.BOX)
        crop = int(shortest * crop_fraction)
        crop_left = (shortest - crop) // 2
        crop_top = (shortest - crop) // 2
        pil_image = pil_image.crop((crop_left, crop_top, crop_left + crop, crop_top + crop))
        pil_image = pil_image.resize((shortest, shortest), Image.Resampling.BOX)
        height, width = self.spec.image_size
        pil_image = pil_image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(pil_image, dtype=np.float32) * (2.0 / 255.0) - 1.0
        return np.ascontiguousarray(np.moveaxis(array, -1, 0)[None].astype(np.float16))

    def _normalize_state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        output = np.zeros((1, 1, self.spec.max_state_dim), dtype=np.float32)
        offset = 0
        for key in self.manifest["features"]["states"]:
            stats = self._stats["state"][key]
            state_min = np.asarray(stats["min"], dtype=np.float32)
            state_max = np.asarray(stats["max"], dtype=np.float32)
            dim = state_min.shape[0]
            raw = values[offset : offset + dim]
            offset += dim
            normalized = (raw - state_min) * (2.0 / (state_max - state_min + 1e-8)) - 1.0
            output[0, 0, offset - dim : offset] = np.clip(normalized, -1.0, 1.0)
        return np.ascontiguousarray(output.astype(np.float16))

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        out = np.zeros((self.spec.action_horizon, self.spec.action_dim), dtype=np.float32)
        offset = 0
        for key in self.manifest["features"]["actions"]:
            stats = self._stats["action"][key]
            action_min = np.asarray(stats["min"], dtype=np.float32)
            action_max = np.asarray(stats["max"], dtype=np.float32)
            dim = action_min.shape[0]
            normalized = np.clip(
                actions[: self.spec.action_horizon, offset : offset + dim],
                -1.0,
                1.0,
            )
            out[:, offset : offset + dim] = 0.5 * (normalized + 1.0) * (action_max - action_min) + action_min
            offset += dim
        return np.ascontiguousarray(out)

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...] | None = None,
        prompt: str | None = None,
        state: Any | None = None,
    ):
        preprocess_start = perf_counter()
        if images is None:
            tensors = {
                "input_ids": self.backbone_inputs["input_ids"].astype(np.int64),
                "attention_mask": self.backbone_inputs["attention_mask"].astype(np.int64),
                "pixel_values_0": self.backbone_inputs["pixel_values_0"].astype(np.float16),
                "pixel_values_1": self.backbone_inputs["pixel_values_1"].astype(np.float16),
                "state": self.action_loop_inputs["state"].astype(np.float16),
                "embodiment_id": self.action_loop_inputs["embodiment_id"].astype(np.int64),
                "initial_actions": self.action_loop_inputs["initial_actions"].astype(np.float16),
            }
        else:
            prompt_cache = self._prompt_cache(prompt)
            tensors = {
                "input_ids": prompt_cache["input_ids"].astype(np.int64),
                "attention_mask": prompt_cache["attention_mask"].astype(np.int64),
                "pixel_values_0": self._prepare_image(images[0]),
                "pixel_values_1": self._prepare_image(images[1]),
                "state": self._normalize_state(state),
                "embodiment_id": self.action_loop_inputs["embodiment_id"].astype(np.int64),
                "initial_actions": self.action_loop_inputs["initial_actions"].astype(np.float16),
            }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(tensors)
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        actions = self._unnormalize_actions(actions)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "groot",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "num_cameras": self.spec.num_cameras,
            "prompt": self.metadata.get("prompt"),
            "prompt_cache_sha256": self.metadata.get("prompt_cache_sha256"),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass(frozen=True)
class Dit4DitModelSpec:
    num_cameras: int
    view_size: tuple[int, int]
    concat_image_size: tuple[int, int]
    state_dim: int
    action_horizon: int
    action_dim: int
    action_latent_dim: int


@dataclass
class Dit4DitRunnerWrapper:
    model_dir: Path = DEFAULT_DIT4DIT_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / DIT4DIT_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / DIT4DIT_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads((self.model_dir / DIT4DIT_PROMPT_CACHE_INDEX_PATH).read_text())
        self._stats = json.loads((self.model_dir / DIT4DIT_DATASET_STATS_PATH).read_text())
        self.spec = self._load_spec()
        constants = np.load(self.model_dir / DIT4DIT_CONSTANTS_PATH)
        self.constants = {name: np.ascontiguousarray(constants[name]) for name in constants.files}
        self.initial_actions = np.ascontiguousarray(
            np.load(self.model_dir / DIT4DIT_INITIAL_ACTIONS_PATH)["actions"].astype(np.float16)
        )
        self._runner = picpp.Dit4DitOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.view_size

    @property
    def state_dim(self) -> int:
        return 8

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> Dit4DitModelSpec:
        inputs = self.manifest["inputs"]
        action = self.manifest["action"]
        action_stats = self._stats[DIT4DIT_ACTION_STATS_KEY]["action"]
        view_size = tuple(int(value) for value in inputs["view_size"])
        concat_image_size = tuple(int(value) for value in inputs["concat_image_size"])
        return Dit4DitModelSpec(
            num_cameras=int(inputs["views"]),
            view_size=(view_size[0], view_size[1]),
            concat_image_size=(concat_image_size[0], concat_image_size[1]),
            state_dim=int(inputs["state_dim"]),
            action_horizon=int(action["horizon"]),
            action_dim=len(action_stats["min"]),
            action_latent_dim=int(action["dim"]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else prompt.strip()
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(self.model_dir / "prompt_embeds" / f"{digest}.npz")
        self.metadata.update({"prompt": task, "prompt_cache_sha256": digest})
        return {name: np.ascontiguousarray(cache[name]) for name in cache.files}

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = self.spec.view_size
        import cv2

        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0)

    def _prepare_video(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        left = self._prepare_image(images[0])
        right = self._prepare_image(images[1])
        frame = np.concatenate([left, right], axis=2) * 2.0 - 1.0
        video = np.zeros((1, 3, 5, *self.spec.concat_image_size), dtype=np.float16)
        video[:, :, 0] = frame.astype(np.float16)
        return np.ascontiguousarray(video)

    def _prepare_state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1)[:8]
        encoded = np.stack([np.sin(values), np.cos(values)], axis=1).reshape(1, 1, self.spec.state_dim)
        return np.ascontiguousarray(encoded.astype(np.float16))

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        stats = self._stats[DIT4DIT_ACTION_STATS_KEY]["action"]
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=np.bool_)), dtype=np.bool_)
        action_low = np.asarray(stats["min"], dtype=np.float32)
        action_high = np.asarray(stats["max"], dtype=np.float32)
        normalized = np.clip(actions[:, : self.spec.action_dim], -1.0, 1.0)
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)
        return np.ascontiguousarray(
            np.where(
                mask,
                0.5 * (normalized + 1.0) * (action_high - action_low) + action_low,
                normalized,
            ).astype(np.float32)
        )

    def _runtime_rng(self, prompt_digest: str, state: Any) -> np.random.Generator:
        values = np.asarray(state, dtype=np.float32).reshape(-1)[:8]
        digest = hashlib.sha256()
        digest.update(prompt_digest.encode("utf-8"))
        digest.update(values.tobytes())
        seed = int.from_bytes(digest.digest()[:8], "little")
        return np.random.default_rng(seed)

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str | None = None,
        state: Any,
    ):
        preprocess_start = perf_counter()
        prompt_cache = self._prompt_cache(prompt)
        prompt_digest = str(self.metadata["prompt_cache_sha256"])
        rng = self._runtime_rng(prompt_digest, state)
        sample_noise = rng.standard_normal(self.constants["sample_noise"].shape).astype(np.float16)
        latents = rng.standard_normal(self.constants["latents"].shape).astype(np.float32)
        initial_actions = rng.standard_normal(self.initial_actions.shape).astype(np.float16)
        tensors = {
            "video_bcthw": self._prepare_video(images),
            "sample_noise": np.ascontiguousarray(sample_noise),
            "latents": np.ascontiguousarray(latents),
            "cond_mask": self.constants["cond_mask"].astype(np.float32),
            "cond_indicator": self.constants["cond_indicator"].astype(np.float32),
            "prompt_embeds": prompt_cache["prompt_embeds"].astype(np.float16),
            "padding_mask": self.constants["padding_mask"].astype(np.float32),
            "sigma_t": self.constants["sigma_t"].astype(np.float32),
            "state": self._prepare_state(state),
            "initial_actions": np.ascontiguousarray(initial_actions),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(tensors)
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        actions = self._unnormalize_actions(actions)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "dit4dit",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "prompt": self.metadata.get("prompt"),
            "prompt_cache_sha256": self.metadata.get("prompt_cache_sha256"),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass(frozen=True)
class Evo1ModelSpec:
    num_cameras: int
    num_image_slots: int
    image_size: tuple[int, int]
    state_input_dim: int
    state_dim: int
    action_horizon: int
    action_width: int
    action_dim: int
    inference_steps: int


@dataclass
class Evo1RunnerWrapper:
    model_dir: Path = DEFAULT_EVO1_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / EVO1_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / EVO1_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads(
            (self.model_dir / EVO1_PROMPT_CACHE_INDEX_PATH).read_text()
        )
        self.spec = self._load_spec()
        self._stats = json.loads((self.model_dir / EVO1_NORM_STATS_PATH).read_text())
        self._robot_stats = self._stats[EVO1_STATS_KEY]
        self._rng = np.random.default_rng()
        self._runner = picpp.Evo1OfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.state_input_dim

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> Evo1ModelSpec:
        inputs = self.manifest["inputs"]
        action = self.manifest["action"]
        image_size = tuple(int(value) for value in inputs["image_size"])
        return Evo1ModelSpec(
            num_cameras=int(inputs["num_cameras"]),
            num_image_slots=int(inputs["num_image_slots"]),
            image_size=(image_size[0], image_size[1]),
            state_input_dim=int(inputs["state_input_dim"]),
            state_dim=int(inputs["state_dim"]),
            action_horizon=int(action["horizon"]),
            action_width=int(action["width"]),
            action_dim=int(action["dim"]),
            inference_steps=int(action["num_inference_timesteps"]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else prompt.strip()
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(
            self.model_dir / self.manifest["internvl"]["prompt_cache"] / f"{digest}.npz"
        )
        return {name: np.ascontiguousarray(cache[name]) for name in cache.files}

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(f"expected one Evo-1 image, got shape {image.shape}")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"expected an HWC or CHW RGB image, got shape {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            raise ValueError(f"expected an HWC or CHW RGB image, got shape {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = self.spec.image_size
        if image.shape[:2] != (height, width):
            import cv2

            image = cv2.resize(image, (width, height))
        image = image[..., ::-1]
        mean = np.asarray(self.manifest["internvl"]["image_mean"], dtype=np.float32)
        std = np.asarray(self.manifest["internvl"]["image_std"], dtype=np.float32)
        image = (image.astype(np.float32) / 255.0 - mean) / std
        return np.moveaxis(image, -1, 0)

    def _prepare_pixel_values(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        if len(images) != self.spec.num_cameras:
            raise ValueError(
                f"expected {self.spec.num_cameras} Evo-1 camera images, got {len(images)}"
            )
        prepared = [self._prepare_image(image) for image in images[: self.spec.num_cameras]]
        height, width = self.spec.image_size
        prepared.extend(
            self._prepare_image(np.zeros((height, width, 3), dtype=np.uint8))
            for _ in range(self.spec.num_image_slots - len(prepared))
        )
        return np.ascontiguousarray(np.stack(prepared).astype(np.float32))

    def _prepare_state(self, state: Any) -> np.ndarray:
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.shape[0] != self.spec.state_input_dim:
            raise ValueError(
                f"expected Evo-1 state width {self.spec.state_input_dim}, got {values.shape[0]}"
            )
        values = np.pad(values, (0, self.spec.state_dim - values.shape[0]))
        state_min = np.pad(
            np.asarray(self._robot_stats["observation.state"]["min"], dtype=np.float32),
            (0, self.spec.state_dim - len(self._robot_stats["observation.state"]["min"])),
        )
        state_max = np.pad(
            np.asarray(self._robot_stats["observation.state"]["max"], dtype=np.float32),
            (0, self.spec.state_dim - len(self._robot_stats["observation.state"]["max"])),
        )
        normalized = np.clip(
            2.0 * (values - state_min) / (state_max - state_min + 1e-8) - 1.0, -1.0, 1.0
        )
        return np.ascontiguousarray(normalized[None].astype(np.float32))

    def _initial_actions(self) -> np.ndarray:
        return np.ascontiguousarray(
            self._rng.uniform(
                -1.0,
                1.0,
                (1, self.spec.action_horizon, self.spec.action_width),
            ).astype(np.float32)
        )

    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        action_min_values = self._robot_stats["action"]["min"]
        action_max_values = self._robot_stats["action"]["max"]
        action_min = np.pad(
            np.asarray(action_min_values, dtype=np.float32),
            (0, self.spec.action_width - len(action_min_values)),
        )
        action_max = np.pad(
            np.asarray(action_max_values, dtype=np.float32),
            (0, self.spec.action_width - len(action_max_values)),
        )
        actions = (actions + 1.0) * 0.5 * (action_max - action_min + 1e-8) + action_min
        return np.ascontiguousarray(actions[:, : self.spec.action_dim].astype(np.float32))

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...] | None = None,
        prompt: str | None = None,
        state: Any | None = None,
    ):
        preprocess_start = perf_counter()
        if images is None:
            raise ValueError("Evo-1 images are required")
        if state is None:
            raise ValueError("Evo-1 state is required")
        prompt_cache = self._prompt_cache(prompt)
        tensors = {
            "input_ids": prompt_cache["input_ids"].astype(np.int64),
            "attention_mask": prompt_cache["attention_mask"].astype(np.float16),
            "attention_mask_2d": prompt_cache["attention_mask_2d"].astype(np.int64),
            "position_ids": prompt_cache["position_ids"].astype(np.int64),
            "pixel_values": self._prepare_pixel_values(images),
            "state": self._prepare_state(state),
            "actions": self._initial_actions(),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(tensors, self.spec.inference_steps)
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        actions = self._denormalize_actions(actions)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "evo1",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass(frozen=True)
class SemanticVlaModelSpec:
    num_cameras: int
    image_size: tuple[int, int]
    state_dim: int
    action_horizon: int
    action_dim: int
    inference_steps: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    processor_size: tuple[int, int]
    pixel_values_shape: tuple[int, int]


@dataclass
class SemanticVlaRunnerWrapper:
    model_dir: Path = DEFAULT_SEMANTICVLA_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / SEMANTICVLA_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / SEMANTICVLA_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads((self.model_dir / SEMANTICVLA_PROMPT_CACHE_INDEX_PATH).read_text())
        self.spec = self._load_spec()
        self._stats = json.loads((self.model_dir / SEMANTICVLA_DATASET_STATS_PATH).read_text())
        self.initial_actions = np.ascontiguousarray(
            np.load(self.model_dir / SEMANTICVLA_INITIAL_ACTIONS_PATH)["actions"].astype(np.float32)
        )
        self._runner = picpp.SemanticVlaOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> SemanticVlaModelSpec:
        inputs = self.manifest["inputs"]
        qwen = self.manifest["qwen"]
        action = self.manifest["action"]
        image_size = tuple(int(value) for value in inputs["image_size"])
        processor_size = tuple(int(value) for value in qwen["processor_resize"])
        pixel_values_shape = tuple(int(value) for value in qwen["pixel_values_shape"])
        return SemanticVlaModelSpec(
            num_cameras=int(inputs["num_cameras"]),
            image_size=(image_size[0], image_size[1]),
            state_dim=int(inputs["state_dim"]),
            action_horizon=int(action["horizon"]),
            action_dim=int(action["dim"]),
            inference_steps=int(action["num_inference_timesteps"]),
            patch_size=int(qwen["patch_size"]),
            temporal_patch_size=int(qwen["temporal_patch_size"]),
            merge_size=int(qwen["merge_size"]),
            processor_size=(processor_size[0], processor_size[1]),
            pixel_values_shape=(pixel_values_shape[0], pixel_values_shape[1]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else prompt.strip()
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(self.model_dir / self.manifest["qwen"]["prompt_cache"] / f"{digest}.npz")
        return {name: np.ascontiguousarray(cache[name]) for name in cache.files}

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = self.spec.image_size
        processor_height, processor_width = self.spec.processor_size
        resized = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BOX)
        qwen_image = resized.resize((processor_width, processor_height), Image.Resampling.BICUBIC)
        array = np.asarray(qwen_image, dtype=np.float32) * (1.0 / 127.5) - 1.0
        return np.moveaxis(array, -1, 0)

    def _patchify_image(self, image: np.ndarray) -> np.ndarray:
        patch = self.spec.patch_size
        temporal_patch = self.spec.temporal_patch_size
        merge = self.spec.merge_size
        channel, height, width = image.shape
        grid_t = 1
        grid_h = height // patch
        grid_w = width // patch
        patches = np.repeat(image[None, None], temporal_patch, axis=1)
        patches = patches.reshape(
            1,
            grid_t,
            temporal_patch,
            channel,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(1, grid_t * grid_h * grid_w, channel * temporal_patch * patch * patch)[0]

    def _prepare_pixel_values(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        prepared = [self._patchify_image(self._prepare_image(image)) for image in images[: self.spec.num_cameras]]
        return np.ascontiguousarray(np.concatenate(prepared, axis=0).astype(np.float16))

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        stats = self._stats[SEMANTICVLA_ACTION_STATS_KEY]["action"]
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=np.bool_)), dtype=np.bool_)
        action_min = np.asarray(stats["min"], dtype=np.float32)
        action_max = np.asarray(stats["max"], dtype=np.float32)
        normalized = np.clip(actions[:, : self.spec.action_dim], -1.0, 1.0)
        normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0.0, 1.0)
        return np.ascontiguousarray(
            np.where(
                mask,
                0.5 * (normalized + 1.0) * (action_max - action_min) + action_min,
                normalized,
            ).astype(np.float32)
        )

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...] | None = None,
        prompt: str | None = None,
        state: Any | None = None,
    ):
        preprocess_start = perf_counter()
        prompt_cache = self._prompt_cache(prompt)
        tensors = {
            "input_ids": prompt_cache["input_ids"].astype(np.int64),
            "attention_mask": prompt_cache["attention_mask_4d"].astype(np.float16),
            "position_ids": prompt_cache["position_ids"].astype(np.int64),
            "visual_select": prompt_cache["visual_select"].astype(np.float16),
            "pixel_values": self._prepare_pixel_values(images),
            "actions": self.initial_actions.copy(),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(tensors, self.spec.inference_steps)
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        actions = self._unnormalize_actions(actions)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "semanticvla",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass(frozen=True)
class StarVlaModelSpec:
    num_cameras: int
    image_size: tuple[int, int]
    action_horizon: int
    action_dim: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    processor_size: tuple[int, int]


@dataclass
class StarVlaRunnerWrapper:
    model_dir: Path = DEFAULT_STARVLA_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / STARVLA_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / STARVLA_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads((self.model_dir / STARVLA_PROMPT_CACHE_INDEX_PATH).read_text())
        self.spec = self._load_spec()
        self._stats = json.loads((self.model_dir / STARVLA_DATASET_STATS_PATH).read_text())
        self._runner = picpp.StarVlaOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return 0

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> StarVlaModelSpec:
        inputs = self.manifest["inputs"]
        qwen = self.manifest["qwen"]
        action = self.manifest["action"]
        image_size = tuple(int(value) for value in inputs["image_size"])
        processor_size = tuple(int(value) for value in qwen["processor_resize"])
        return StarVlaModelSpec(
            num_cameras=int(inputs["num_cameras"]),
            image_size=(image_size[0], image_size[1]),
            action_horizon=int(action["horizon"]),
            action_dim=int(action["dim"]),
            patch_size=int(qwen["patch_size"]),
            temporal_patch_size=int(qwen["temporal_patch_size"]),
            merge_size=int(qwen["merge_size"]),
            processor_size=(processor_size[0], processor_size[1]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else prompt.strip()
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(self.model_dir / self.manifest["qwen"]["prompt_cache"] / f"{digest}.npz")
        self.metadata.update({"prompt": task, "prompt_cache_sha256": digest})
        return {name: np.ascontiguousarray(cache[name]) for name in cache.files}

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = self.spec.image_size
        processor_height, processor_width = self.spec.processor_size
        resized = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        qwen_image = resized.resize((processor_width, processor_height), Image.Resampling.BICUBIC)
        array = np.asarray(qwen_image, dtype=np.float32) * (1.0 / 127.5) - 1.0
        return np.moveaxis(array, -1, 0)

    def _patchify_image(self, image: np.ndarray) -> np.ndarray:
        patch = self.spec.patch_size
        temporal_patch = self.spec.temporal_patch_size
        merge = self.spec.merge_size
        channel, height, width = image.shape
        grid_t = 1
        grid_h = height // patch
        grid_w = width // patch
        patches = np.repeat(image[None, None], temporal_patch, axis=1)
        patches = patches.reshape(
            1,
            grid_t,
            temporal_patch,
            channel,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(1, grid_t * grid_h * grid_w, channel * temporal_patch * patch * patch)[0]

    def _prepare_pixel_values(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        prepared = [self._patchify_image(self._prepare_image(image)) for image in images[: self.spec.num_cameras]]
        return np.ascontiguousarray(np.concatenate(prepared, axis=0).astype(np.float16))

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        stats = self._stats[STARVLA_ACTION_STATS_KEY]["action"]
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=np.bool_)), dtype=np.bool_)
        action_min = np.asarray(stats["min"], dtype=np.float32)
        action_max = np.asarray(stats["max"], dtype=np.float32)
        normalized = actions[:, : self.spec.action_dim].astype(np.float32)
        output = normalized.copy()
        output[:, mask] = (
            0.5
            * (np.clip(normalized[:, mask], -1.0, 1.0) + 1.0)
            * (action_max[mask] - action_min[mask])
            + action_min[mask]
        )
        return np.ascontiguousarray(output.astype(np.float32))

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str | None = None,
        state: Any | None = None,
    ):
        _ = state
        preprocess_start = perf_counter()
        prompt_cache = self._prompt_cache(prompt)
        tensors = {
            "input_ids": prompt_cache["input_ids"].astype(np.int64),
            "attention_mask": prompt_cache["attention_mask"].astype(np.float16),
            "position_ids": prompt_cache["position_ids"].astype(np.int64),
            "visual_select": prompt_cache["visual_select"].astype(np.float16),
            "action_select": prompt_cache["action_select"].astype(np.float16),
            "pixel_values": self._prepare_pixel_values(images),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(tensors)
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        actions = self._unnormalize_actions(actions)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "starvla",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "prompt": self.metadata.get("prompt"),
            "prompt_cache_sha256": self.metadata.get("prompt_cache_sha256"),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass
class FastWamRunnerWrapper:
    model_dir: Path = DEFAULT_FASTWAM_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / FASTWAM_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / FASTWAM_MANIFEST_PATH).read_text())
        self._stats: dict[str, Any] | None = None
        self.proprio_encoder_path = self.model_dir / FASTWAM_PROPRIO_ENCODER
        self._runner = picpp.FastWamOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        input_image_shape = self._input_shapes["input_image"]
        latents_shape = self._input_shapes["latents_action"]
        context_shape = self._input_shapes["context"]
        self.context_dim = int(context_shape[2])
        self.image_height = int(input_image_shape[2])
        self.image_width = int(input_image_shape[3])
        if self.image_width == self.image_height * 2:
            self.image_layout = "horizontal_2cam"
            self.camera_count = 2
        else:
            self.image_layout = "robotwin_3cam_mosaic"
            self.camera_count = 3
        self.context_len = int(context_shape[1]) - 1
        self.text_context_mode = str(self.manifest["metadata"].get("text_context", ""))
        self.text_cache_dir = self.model_dir / FASTWAM_TEXT_CACHE_DIR
        self.proprio_encoder_weight, self.proprio_encoder_bias = self._load_proprio_encoder()
        self.inference_steps = int(self.manifest["loop"]["steps"])
        self.scheduler_shift = float(self.manifest["metadata"]["scheduler"]["action"]["infer_shift"])
        self.num_train_timesteps = int(self.manifest["metadata"]["scheduler"]["action"]["num_train_timesteps"])
        self._action_horizon = int(latents_shape[1])
        self._action_dim = int(latents_shape[2])
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.camera_count

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def state_dim(self) -> int:
        return len(self._stats_data()["state"]["default"]["global_min"])

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _stats_data(self) -> dict[str, Any]:
        if self._stats is None:
            self._stats = json.loads((self.model_dir / FASTWAM_DATASET_STATS).read_text())
        return self._stats

    def _prepare_images(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        if self.image_layout == "horizontal_2cam":
            left = self._resize_rgb_to(images[0], width=self.image_width // 2, height=self.image_height)
            right = self._resize_rgb_to(
                images[1],
                width=self.image_width - self.image_width // 2,
                height=self.image_height,
            )
            image = np.concatenate([left, right], axis=2)
            return np.ascontiguousarray((image[None] * 2.0 - 1.0).astype(np.float16))

        head_height = self.image_height * 2 // 3
        wrist_height = self.image_height - head_height
        wrist_width = self.image_width // 2
        head = self._resize_rgb_to(images[0], width=self.image_width, height=head_height)
        left = self._resize_rgb_to(images[1], width=wrist_width, height=wrist_height)
        right = self._resize_rgb_to(images[2], width=self.image_width - wrist_width, height=wrist_height)
        bottom = np.concatenate([left, right], axis=2)
        mosaic = np.concatenate([head, bottom], axis=1)
        return np.ascontiguousarray((mosaic[None] * 2.0 - 1.0).astype(np.float16))

    def _resize_rgb_to(self, value: Any, *, width: int, height: int) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        resized = Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        return np.moveaxis(np.asarray(resized, dtype=np.float32) / 255.0, -1, 0)

    def _format_prompt(self, prompt: str | None) -> str:
        text = prompt.strip()
        if text.startswith("A video recorded from a robot's point of view"):
            return text
        return FASTWAM_DEFAULT_PROMPT.format(task=text)

    def _text_cache_path(self, prompt: str | None) -> tuple[Path, str]:
        formatted_prompt = self._format_prompt(prompt)
        digest = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
        filename = f"{digest}.t5_len{self.context_len}.{FASTWAM_TEXT_CACHE_ENCODER_ID}.pt"
        return self.text_cache_dir / filename, formatted_prompt

    def _read_text_cache_pt(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            root = names[0].split("/", 1)[0]
            pickle_bytes = archive.read(f"{root}/data.pkl")
            context_bytes = archive.read(f"{root}/data/0")
            if b"BFloat16Storage" in pickle_bytes:
                context_u16 = np.frombuffer(context_bytes, dtype=np.dtype("<u2")).astype(np.uint32)
                context = (context_u16 << 16).view(np.float32).astype(np.float16)
            elif b"HalfStorage" in pickle_bytes:
                context = np.frombuffer(context_bytes, dtype=np.dtype("<f2"))
            elif b"FloatStorage" in pickle_bytes:
                context = np.frombuffer(context_bytes, dtype=np.dtype("<f4"))
            else:
                context = np.frombuffer(context_bytes, dtype=np.dtype("<f4"))
            mask_bytes = archive.read(f"{root}/data/1")
            mask_size = len(mask_bytes)
            if mask_size == self.context_len:
                mask = np.frombuffer(mask_bytes, dtype=np.bool_)
            elif mask_size == self.context_len * 2:
                mask = np.frombuffer(mask_bytes, dtype=np.dtype("<i2")).astype(np.bool_)
            else:
                mask = np.frombuffer(mask_bytes, dtype=np.dtype("<i4")).astype(np.bool_)
        context = context.reshape(1, self.context_len, self.context_dim).astype(np.float16)
        mask = mask.reshape(1, self.context_len).astype(np.bool_)
        context[:, ~mask[0], :] = 0.0
        return np.ascontiguousarray(context), np.ascontiguousarray(np.ones_like(mask, dtype=np.bool_))

    def _load_context(self, prompt: str | None) -> tuple[np.ndarray, np.ndarray]:
        cache_path, formatted_prompt = self._text_cache_path(prompt)
        context, context_mask = self._read_text_cache_pt(cache_path)
        self.metadata.update(
            {
                "text_context": "precomputed_t5_cache",
                "text_cache_path": str(cache_path),
                "prompt": formatted_prompt,
            }
        )
        return context, context_mask

    def _normalize_state(self, state: Any) -> np.ndarray:
        stats = self._stats_data()["state"]["default"]
        state_min = np.asarray(stats["global_min"], dtype=np.float32).reshape(1, -1)
        state_max = np.asarray(stats["global_max"], dtype=np.float32).reshape(1, -1)
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state_range = state_max - state_min
        ignored = state_range < 1e-4
        state_range[ignored] = 2.0
        scale = 2.0 / state_range
        offset = -1.0 - scale * state_min
        offset[ignored] = -state_min[ignored]
        return np.ascontiguousarray(np.clip(state_array * scale + offset, -5.0, 5.0))

    def _append_proprio(self, context: np.ndarray, context_mask: np.ndarray, state: Any) -> tuple[np.ndarray, np.ndarray]:
        proprio = self._normalize_state(state)
        proprio_token = (
            proprio.astype(np.float32) @ self.proprio_encoder_weight.T + self.proprio_encoder_bias
        ).astype(np.float16)[:, None, :]
        proprio_mask = np.ones((1, 1), dtype=np.bool_)
        return (
            np.ascontiguousarray(np.concatenate([context, proprio_token], axis=1)),
            np.ascontiguousarray(np.concatenate([context_mask, proprio_mask], axis=1)),
        )

    @staticmethod
    def _read_torch_storage_tensor(archive: zipfile.ZipFile, name: str, *, expected_values: int) -> np.ndarray:
        data = archive.read(name)
        data_len = len(data)
        dtype = "<f2" if data_len == expected_values * 2 else "<f4"
        return np.frombuffer(data, dtype=np.dtype(dtype)).astype(np.float32)

    def _load_proprio_encoder(self) -> tuple[np.ndarray, np.ndarray]:
        state_dim = self.state_dim
        weight_values = self.context_dim * state_dim
        bias_values = self.context_dim
        with zipfile.ZipFile(self.proprio_encoder_path.expanduser()) as archive:
            root = archive.namelist()[0].split("/", 1)[0]
            weight = self._read_torch_storage_tensor(
                archive,
                f"{root}/data/0",
                expected_values=weight_values,
            ).reshape(self.context_dim, state_dim)
            bias = self._read_torch_storage_tensor(
                archive,
                f"{root}/data/1",
                expected_values=bias_values,
            ).reshape(self.context_dim)
        return (
            np.ascontiguousarray(weight),
            np.ascontiguousarray(bias),
        )

    def _action_denorm(self) -> tuple[np.ndarray, np.ndarray]:
        stats = self._stats_data()["action"]["default"]
        action_min = np.asarray(stats["global_min"], dtype=np.float32).reshape(-1)
        action_max = np.asarray(stats["global_max"], dtype=np.float32).reshape(-1)
        mean = (action_max + action_min) * 0.5
        std = (action_max - action_min) * 0.5
        return np.ascontiguousarray(mean), np.ascontiguousarray(std)

    def _scheduler_deltas(self) -> np.ndarray:
        u_steps = np.linspace(1.0, 0.0, self.inference_steps + 1, dtype=np.float32)
        sigma_steps = self.scheduler_shift * u_steps / (
            1.0 + (self.scheduler_shift - 1.0) * u_steps
        )
        return np.ascontiguousarray((sigma_steps[1:] - sigma_steps[:-1]).astype(np.float32))

    def _timestep_action(self) -> np.ndarray:
        u_steps = np.linspace(1.0, 0.0, self.inference_steps + 1, dtype=np.float32)
        sigma_steps = self.scheduler_shift * u_steps / (
            1.0 + (self.scheduler_shift - 1.0) * u_steps
        )
        timesteps = (sigma_steps[:-1] * float(self.num_train_timesteps)).astype(np.float16)
        return np.ascontiguousarray(timesteps.reshape(self.inference_steps, 1))

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...] | None = None,
        prompt: str | None = None,
        state: Any | None = None,
    ):
        preprocess_start = perf_counter()
        context, context_mask = self._load_context(prompt)
        context, context_mask = self._append_proprio(context, context_mask, state)
        rng = np.random.default_rng(FASTWAM_SEED)
        latents_action = np.ascontiguousarray(
            rng.standard_normal(
                (1, self.action_horizon, self.action_dim),
                dtype=np.float32,
            ).astype(np.float16)
        )
        tensors = {
            "input_image": self._prepare_images(images),
            "context": context,
            "context_mask": context_mask,
            "latents_action": latents_action,
            "timestep_action": self._timestep_action(),
        }
        scheduler_deltas = self._scheduler_deltas()
        action_mean, action_std = self._action_denorm()
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(
            tensors,
            scheduler_deltas,
            action_mean,
            action_std,
            self.inference_steps,
            self.action_dim,
        )
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "fastwam",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "steps": self.inference_steps,
            "action_dim": self.action_dim,
            "text_context": self.metadata.get("text_context", self.text_context_mode),
            "text_cache_path": self.metadata.get("text_cache_path"),
            "prompt": self.metadata.get("prompt"),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions[0]


@dataclass(frozen=True)
class SmolVlaModelSpec:
    num_cameras: int
    image_size: tuple[int, int]
    state_dim: int
    max_state_dim: int
    token_length: int
    action_horizon: int
    action_dim: int
    action_latent_dim: int
    inference_steps: int


@dataclass
class SmolVlaRunnerWrapper:
    model_dir: Path = DEFAULT_SMOLVLA_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / SMOLVLA_ENGINE_DIR
        self.manifest = json.loads((self.model_dir / SMOLVLA_MANIFEST_PATH).read_text())
        self.prompt_cache_index = json.loads(
            (self.model_dir / SMOLVLA_PROMPT_CACHE_INDEX_PATH).read_text()
        )["prompts"]
        self.spec = self._load_spec()
        self._stats = json.loads((self.model_dir / SMOLVLA_DATASET_STATS_PATH).read_text())
        self._runner = picpp.SmolVlaOfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def input_shapes(self) -> dict[str, list[int]]:
        return self._input_shapes

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    @property
    def action_horizon(self) -> int:
        return self.spec.action_horizon

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def default_prompt(self) -> str:
        return str(self.prompt_cache_index[0]["task"])

    def _load_spec(self) -> SmolVlaModelSpec:
        inputs = self.manifest["inputs"]
        action = self.manifest["action"]
        loop = self.manifest["loop"]
        image_size = tuple(int(value) for value in inputs["image_size"])
        return SmolVlaModelSpec(
            num_cameras=int(inputs["num_cameras"]),
            image_size=(image_size[0], image_size[1]),
            state_dim=int(inputs["state_dim"]),
            max_state_dim=int(inputs["max_state_dim"]),
            token_length=int(inputs["token_length"]),
            action_horizon=int(action["horizon"]),
            action_dim=int(action["dim"]),
            action_latent_dim=int(action["latent_dim"]),
            inference_steps=int(loop["steps"]),
        )

    def _prompt_cache(self, prompt: str | None) -> dict[str, np.ndarray]:
        task = self.default_prompt if prompt is None else " ".join(prompt.strip().split())
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()
        cache = np.load(self.model_dir / self.manifest["assets"]["prompt_cache"] / f"{digest}.npz")
        self.metadata.update({"prompt": task, "prompt_cache_sha256": digest})
        return {name: np.ascontiguousarray(cache[name]) for name in cache.files}

    def _prepare_image(self, value: Any) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = self.spec.image_size
        source_height, source_width = image.shape[:2]
        ratio = max(source_width / width, source_height / height)
        resized_height = int(source_height / ratio)
        resized_width = int(source_width / ratio)
        resized = Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        padded = np.zeros((height, width, 3), dtype=np.float32)
        top = height - resized_height
        left = width - resized_width
        padded[top:, left:] = np.asarray(resized, dtype=np.float32) / 255.0
        return np.moveaxis(padded * 2.0 - 1.0, -1, 0)

    def _prepare_images(self, images: list[Any] | tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
        prepared = [self._prepare_image(image) for image in images[: self.spec.num_cameras]]
        image_mask = np.zeros((1, self.spec.num_cameras), dtype=np.bool_)
        image_mask[:, : len(prepared)] = True
        pad_image = np.full((3, *self.spec.image_size), -1.0, dtype=np.float32)
        while len(prepared) < self.spec.num_cameras:
            prepared.append(pad_image.copy())
        image = np.stack(prepared, axis=0)[None].astype(np.float32)
        return np.ascontiguousarray(image), np.ascontiguousarray(image_mask)

    def _normalize_state(self, state: Any) -> np.ndarray:
        stats = self._stats[SMOLVLA_STATE_STATS_KEY]
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1)
        values = np.asarray(state, dtype=np.float32).reshape(1, -1)
        normalized = (values - mean) / (std + 1e-8)
        out = np.zeros((1, self.spec.max_state_dim), dtype=np.float32)
        out[:, : normalized.shape[1]] = normalized[:, : self.spec.max_state_dim]
        return np.ascontiguousarray(out.astype(np.float32))

    def _action_denorm(self) -> tuple[np.ndarray, np.ndarray]:
        stats = self._stats[SMOLVLA_ACTION_STATS_KEY]
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(mean), np.ascontiguousarray(std)

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str | None = None,
        state: Any,
    ):
        preprocess_start = perf_counter()
        prompt_cache = self._prompt_cache(prompt)
        image, image_mask = self._prepare_images(images)
        x_t_seed = int(hashlib.sha256(prompt_cache["tokenized_prompt"].tobytes()).hexdigest()[:16], 16)
        rng = np.random.default_rng(x_t_seed ^ SMOLVLA_SEED)
        x_t = rng.standard_normal(
            (1, self.spec.action_horizon, self.spec.action_latent_dim),
            dtype=np.float32,
        )
        tensors = {
            "image": image,
            "image_mask": image_mask,
            "tokenized_prompt": prompt_cache["tokenized_prompt"].astype(np.int64),
            "tokenized_prompt_mask": prompt_cache["tokenized_prompt_mask"].astype(np.bool_),
            "state": self._normalize_state(state),
            "x_t": np.ascontiguousarray(x_t.astype(np.float32)),
        }
        action_mean, action_std = self._action_denorm()
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        result = self._runner.run_once(
            tensors,
            action_mean,
            action_std,
            self.spec.action_dim,
        )
        postprocess_start = perf_counter()
        actions = np.asarray(result.action, dtype=np.float32).reshape(result.action_shape)[0]
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "smolvla",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "steps": self.spec.inference_steps,
            "prompt": self.metadata.get("prompt"),
            "prompt_cache_sha256": self.metadata.get("prompt_cache_sha256"),
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


@dataclass(frozen=True)
class Pi05ModelSpec:
    num_cameras: int
    image_size: tuple[int, int]
    state_dim: int
    token_length: int
    action_horizon: int
    action_steps: int
    action_dim: int
    action_latent_dim: int


@dataclass
class Pi05RunnerWrapper:
    model_dir: Path = DEFAULT_PI05_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.engine_dir = self.model_dir / PI05_ENGINE_DIR
        self.tokenizer_path = self.model_dir / PI05_TOKENIZER_PATH
        self.norm_stats_path = self.model_dir / PI05_NORM_STATS_PATH
        self._norm_stats = self._load_norm_stats()
        self.spec = self._load_spec()
        self._tokenizer = None
        self._runner = picpp.Pi05OfflineRunner(self.engine_dir)
        self._runner.load()
        self._input_shapes = self._runner.input_shapes()
        self.camera_count = self.spec.num_cameras
        self.metadata: dict[str, Any] = {"input_shapes": self._input_shapes}

    @property
    def num_cameras(self) -> int:
        return self.spec.num_cameras

    @property
    def image_size(self) -> tuple[int, int]:
        return self.spec.image_size

    @property
    def state_dim(self) -> int:
        return self.spec.state_dim

    def _load_norm_stats(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.norm_stats_path.read_text())["norm_stats"]

    def _load_spec(self) -> Pi05ModelSpec:
        manifest = json.loads((self.model_dir / PI05_MANIFEST_PATH).read_text())
        metadata = manifest["metadata"]
        image_size = tuple(int(value) for value in metadata["image_size"])

        suffix_io = json.loads((self.model_dir / PI05_SUFFIX_IO_PATH).read_text())
        x_t_shape = suffix_io["inputs"]["x_t"]["shape"]
        action_horizon = int(x_t_shape[1])
        action_steps = min(int(metadata.get("n_action_steps", action_horizon)), action_horizon)

        return Pi05ModelSpec(
            num_cameras=int(metadata["num_cameras"]),
            image_size=(image_size[0], image_size[1]),
            state_dim=len(self._norm_stats[PI05_STATE_STATS_KEY]["q01"]),
            token_length=int(metadata["max_token_len"]),
            action_horizon=action_horizon,
            action_steps=action_steps,
            action_dim=len(self._norm_stats[PI05_ACTION_STATS_KEY]["q01"]),
            action_latent_dim=int(x_t_shape[2]),
        )

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            tokenizer = sentencepiece.SentencePieceProcessor()
            tokenizer.Load(str(self.tokenizer_path))
            self._tokenizer = tokenizer
        return self._tokenizer

    def _prepare_images(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        height, width = self.spec.image_size
        prepared = []
        for image in images:
            array = np.asarray(image)
            if array.shape[0] == 3:
                array = np.moveaxis(array, 0, -1)
            if np.issubdtype(array.dtype, np.floating):
                array = (255.0 * array).astype(np.uint8)

            source_height, source_width = array.shape[:2]
            ratio = max(source_width / width, source_height / height)
            resized_height = int(source_height / ratio)
            resized_width = int(source_width / ratio)
            resized = Image.fromarray(array).resize((resized_width, resized_height), Image.Resampling.BILINEAR)
            padded = np.zeros((height, width, 3), dtype=np.uint8)
            top = (height - resized_height) // 2
            left = (width - resized_width) // 2
            padded[top : top + resized_height, left : left + resized_width] = np.asarray(resized, dtype=np.uint8)
            prepared.append(np.moveaxis(padded.astype(np.float32) * (2.0 / 255.0) - 1.0, -1, 0))

        pad_image = np.full((3, height, width), -1.0, dtype=np.float32)
        while len(prepared) < self.spec.num_cameras:
            prepared.append(pad_image.copy())

        return np.ascontiguousarray(np.stack(prepared, axis=0)[None], dtype=np.float32)

    def _prepare_image_mask(self, real_image_count: int) -> np.ndarray:
        mask = np.zeros((1, self.spec.num_cameras), dtype=np.bool_)
        mask[:, :real_image_count] = True
        return mask

    def _normalize_state(self, state: Any) -> np.ndarray:
        stats = self._norm_stats[PI05_STATE_STATS_KEY]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        return (values[: mean.shape[0]] - mean) / (std + 1e-8)

    def _prepare_tokenized_prompt(self, prompt: str, state: Any) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(self._normalize_state(state), bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_text = " ".join(map(str, discretized_state.astype(np.int64).tolist()))
        full_prompt = f"Task: {cleaned_text}, State: {state_text};\nAction: "

        tokens = list(self._load_tokenizer().EncodeAsIds(full_prompt))
        tokens = [self._load_tokenizer().bos_id(), *tokens]
        tokens_len = len(tokens)
        if tokens_len < self.spec.token_length:
            mask = [True] * tokens_len + [False] * (self.spec.token_length - tokens_len)
            tokens = tokens + [0] * (self.spec.token_length - tokens_len)
        else:
            tokens = tokens[: self.spec.token_length]
            mask = [True] * self.spec.token_length

        return (
            np.ascontiguousarray(np.asarray(tokens, dtype=np.int64)[None]),
            np.ascontiguousarray(np.asarray(mask, dtype=np.bool_)[None]),
        )

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        stats = self._norm_stats[PI05_ACTION_STATS_KEY]
        action_min = np.asarray(stats["q01"], dtype=np.float32)
        action_max = np.asarray(stats["q99"], dtype=np.float32)
        return (actions[:, : self.spec.action_dim] + 1.0) * 0.5 * (action_max - action_min + 1e-6) + action_min

    def _postprocess_actions(self, action: Any) -> np.ndarray:
        actions = np.asarray(action, dtype=np.float32).reshape(
            self.spec.action_horizon,
            self.spec.action_latent_dim,
        )
        return np.ascontiguousarray(self._unnormalize_actions(actions))

    def run_once(
        self,
        *,
        images: list[Any] | tuple[Any, ...] | None = None,
        prompt: str | None = None,
        state: Any | None = None,
    ):
        preprocess_start = perf_counter()
        images_array = self._prepare_images(images)
        image_mask = self._prepare_image_mask(len(images))
        tokenized_prompt, tokenized_prompt_mask = self._prepare_tokenized_prompt(prompt, state)

        x_t_seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(x_t_seed)
        x_t = rng.standard_normal(
            (1, self.spec.action_horizon, self.spec.action_latent_dim),
            dtype=np.float32,
        )
        tensors = {
            "image": images_array,
            "image_mask": image_mask,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
            "x_t": np.ascontiguousarray(x_t),
        }
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0
        result = self._runner.run_once(tensors)
        postprocess_start = perf_counter()
        actions = self._postprocess_actions(result.action)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        self.metadata = {
            "model": "pi05",
            "model_dir": str(self.model_dir),
            "engine_dir": str(self.engine_dir),
            "num_cameras": self.spec.num_cameras,
            "action_horizon": self.spec.action_horizon,
            "action_steps": self.spec.action_steps,
            "input_shapes": {name: list(array.shape) for name, array in tensors.items()},
            "preprocess_ms": preprocess_ms,
            "postprocess_ms": postprocess_ms,
            **result.to_dict(),
        }
        return actions


def build_pi05_runner(
    *,
    model_dir: str | Path | None = None,
) -> Pi05RunnerWrapper:
    return Pi05RunnerWrapper(
        model_dir=DEFAULT_PI05_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_semanticvla_runner(
    *,
    model_dir: str | Path | None = None,
) -> SemanticVlaRunnerWrapper:
    return SemanticVlaRunnerWrapper(
        model_dir=DEFAULT_SEMANTICVLA_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_evo1_runner(
    *,
    model_dir: str | Path | None = None,
) -> Evo1RunnerWrapper:
    return Evo1RunnerWrapper(
        model_dir=DEFAULT_EVO1_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_fastwam_runner(
    *,
    model_dir: str | Path | None = None,
) -> FastWamRunnerWrapper:
    return FastWamRunnerWrapper(
        model_dir=DEFAULT_FASTWAM_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_dit4dit_runner(
    *,
    model_dir: str | Path | None = None,
) -> Dit4DitRunnerWrapper:
    return Dit4DitRunnerWrapper(
        model_dir=DEFAULT_DIT4DIT_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_smolvla_runner(
    *,
    model_dir: str | Path | None = None,
) -> SmolVlaRunnerWrapper:
    return SmolVlaRunnerWrapper(
        model_dir=DEFAULT_SMOLVLA_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_groot_runner(
    *,
    model_dir: str | Path | None = None,
) -> GrootRunnerWrapper:
    return GrootRunnerWrapper(
        model_dir=DEFAULT_GROOT_MODEL_DIR if model_dir is None else Path(model_dir),
    )


def build_starvla_runner(
    *,
    model_dir: str | Path | None = None,
) -> StarVlaRunnerWrapper:
    return StarVlaRunnerWrapper(
        model_dir=DEFAULT_STARVLA_MODEL_DIR if model_dir is None else Path(model_dir),
    )
