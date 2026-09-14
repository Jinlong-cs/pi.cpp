"""WebSocket policy adapters."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from policy_websocket import BasePolicy
import pi_cpp as picpp

LIBERO_TASK_ALIASES = {
    "pick up the black bowl in the top layer of the wooden cabinet and place it on the plate": (
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
    ),
    "pick up the black bowl next to the cookies box and place it on the plate": (
        "pick up the black bowl next to the cookie box and place it on the plate"
    ),
    "pick up the black bowl on the cookies box and place it on the plate": (
        "pick up the black bowl on the cookie box and place it on the plate"
    ),
    "pick the alphabet soup and place it in the basket": (
        "pick up the alphabet soup and place it in the basket"
    ),
    "pick the bbq sauce and place it in the basket": (
        "pick up the bbq sauce and place it in the basket"
    ),
    "pick the butter and place it in the basket": (
        "pick up the butter and place it in the basket"
    ),
    "pick the chocolate pudding and place it in the basket": (
        "pick up the chocolate pudding and place it in the basket"
    ),
    "pick the cream cheese and place it in the basket": (
        "pick up the cream cheese and place it in the basket"
    ),
    "pick the ketchup and place it in the basket": (
        "pick up the ketchup and place it in the basket"
    ),
    "pick the milk and place it in the basket": (
        "pick up the milk and place it in the basket"
    ),
    "pick the orange juice and place it in the basket": (
        "pick up the orange juice and place it in the basket"
    ),
    "pick the salad dressing and place it in the basket": (
        "pick up the salad dressing and place it in the basket"
    ),
    "pick the tomato sauce and place it in the basket": (
        "pick up the tomato sauce and place it in the basket"
    ),
    "open the middle layer of the drawer": "open the middle drawer of the cabinet",
    "open the top layer of the drawer and put the bowl inside": (
        "open the top drawer and put the bowl inside"
    ),
    "put the bowl on the top of the drawer": "put the bowl on top of the cabinet",
    "put the cream cheese on the bowl": "put the cream cheese in the bowl",
    "put the wine bottle on the top of the drawer": "put the wine bottle on top of the cabinet",
}


def _canonical_libero_task(task: str) -> str:
    task = " ".join(task.strip().lower().split())
    if task.startswith("pick the akita black bowl "):
        task = "pick up the black bowl " + task.removeprefix("pick the akita black bowl ")
    return LIBERO_TASK_ALIASES.get(task, task)


@dataclass
class Pi05Policy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_pi05_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.spec.action_horizon
        self.action_steps = self._runner.spec.action_steps
        self.action_dim = self._runner.spec.action_dim
        self.image_size = self._runner.spec.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=_canonical_libero_task(str(obs["task"])),
            state=obs["state"],
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class Pi06AirbotPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_pi06_airbot_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim
        self.camera_order = self._runner.spec.camera_order
        self.internal_action_dim = self._runner.spec.internal_action_dim
        self.token_length = self._runner.spec.token_length
        self.action_semantics = self._runner.spec.action_semantics
        self.advantage_conditioning = self._runner.spec.advantage_conditioning
        self.default_advantage = self._runner.spec.default_advantage
        self.denoise_steps = self._runner.spec.denoise_steps
        self.dt = self._runner.spec.dt
        self.manifest_schema = self._runner.manifest_schema

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=tuple(obs[name] for name in self.camera_order),
            prompt=str(obs["prompt"]),
            state=obs["state"],
            advantage=bool(obs["advantage"]),
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class Pi06HeterogeneousPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_pi06_heterogeneous_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim
        self.camera_order = self._runner.spec.camera_order
        self.internal_action_dim = self._runner.spec.internal_action_dim
        self.token_length = self._runner.spec.token_length
        self.action_semantics = self._runner.spec.action_semantics
        self.num_embodiments = self._runner.spec.num_embodiments
        self.default_embodiment_id = self._runner.spec.default_embodiment_id
        self.denoise_steps = self._runner.spec.denoise_steps
        self.dt = self._runner.spec.dt
        self.manifest_schema = self._runner.manifest_schema

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=tuple(obs[name] for name in self.camera_order),
            prompt=str(obs["prompt"]),
            state=obs["state"],
            embodiment_id=obs.get("embodiment_id"),
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class Pi06RtcPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_pi06_rtc_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim
        self.camera_order = self._runner.spec.camera_order
        self.internal_action_dim = self._runner.spec.internal_action_dim
        self.token_length = self._runner.spec.token_length
        self.action_semantics = self._runner.spec.action_semantics
        self.num_embodiments = self._runner.spec.num_embodiments
        self.default_embodiment_id = self._runner.spec.default_embodiment_id
        self.denoise_steps = self._runner.spec.denoise_steps
        self.dt = self._runner.spec.dt
        self.manifest_schema = self._runner.manifest_schema
        self.rtc_max_delay = self._runner.rtc_max_delay

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=tuple(obs[name] for name in self.camera_order),
            prompt=str(obs["prompt"]),
            state=obs["state"],
            embodiment_id=obs.get("embodiment_id"),
            delay=obs.get("delay"),
            action_prefix=obs.get("action_prefix"),
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class FastWamPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_fastwam_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = np.asarray(
            self._runner.run_once(
                images=obs["images"],
                prompt=_canonical_libero_task(str(obs["task"])),
                state=obs["state"],
            ),
            dtype=np.float32,
        )
        actions[:, -1] = 1.0 - 2.0 * actions[:, -1]
        actions[:, -1] = np.sign(actions[:, -1])
        return {"actions": actions}


@dataclass
class SemanticVlaPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_semanticvla_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=_canonical_libero_task(str(obs["task"])),
            state=obs["state"],
        )
        actions[:, -1] = 1.0 - 2.0 * (actions[:, -1] > 0.5)
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class Evo1Policy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_evo1_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=str(obs["task"]),
            state=obs["state"],
        )
        actions[:, -1] = np.where(actions[:, -1] > 0.5, -1.0, 1.0)
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class SmolVlaPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_smolvla_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=_canonical_libero_task(str(obs["task"])),
            state=obs["state"],
        )
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class Dit4DitPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_dit4dit_runner(model_dir=self.model_dir)
        self._prompt_by_task = {
            " ".join(str(row["task"]).strip().lower().split()): str(row["task"])
            for row in self._runner.prompt_cache_index
        }
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        task = _canonical_libero_task(str(obs["task"]))
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=self._prompt_by_task[task],
            state=obs["state"],
        )
        actions[:, -1] = 1.0 - 2.0 * (actions[:, -1] > 0.5)
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class GrootPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_groot_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=str(obs["task"]),
            state=obs["state"],
        )
        actions[:, -1] = 1.0 - 2.0 * (actions[:, -1] > 0.5)
        return {"actions": np.asarray(actions, dtype=np.float32)}


@dataclass
class StarVlaPolicy(BasePolicy):
    model_dir: Path

    def __post_init__(self) -> None:
        self._runner = picpp.build_starvla_runner(model_dir=self.model_dir)
        self.action_horizon = self._runner.action_horizon
        self.action_dim = self._runner.action_dim
        self.image_size = self._runner.image_size
        self.num_cameras = self._runner.num_cameras
        self.state_dim = self._runner.state_dim

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self._runner.run_once(
            images=obs["images"],
            prompt=_canonical_libero_task(str(obs["task"])),
        )
        actions[:, -1] = 1.0 - 2.0 * (actions[:, -1] > 0.5)
        return {"actions": np.asarray(actions, dtype=np.float32)}
