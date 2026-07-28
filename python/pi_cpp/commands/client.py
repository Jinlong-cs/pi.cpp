"""Closed-loop LIBERO client command."""

from __future__ import annotations
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import json
import os
import pickle
from pathlib import Path
import runpy
import sys
from time import perf_counter
from typing import Any
import zipfile
import numpy as np
from policy_websocket import WebsocketClientPolicy
import policy_websocket.websocket_client as websocket_client_module
from rich.console import Console
from rich.progress import track

console = Console()

TASK_SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_RESOLUTION = 256
ACTION_DIM = 7
TRIALS_PER_TASK = 10
DEFAULT_SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

LIBERO_NUM_STEPS_WAIT = 30
LIBERO_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 800,
}
MODEL_REPLAN_STEPS = {
    "pi05": 5,
    "fastwam": 10,
    "semanticvla": 8,
    "evo1": 14,
    "smolvla": 10,
    "dit4dit": 8,
    "groot": 16,
    "starvla": 8,
}


websocket_client_module.websockets.sync.client.connect = partial(
    websocket_client_module.websockets.sync.client.connect,
    ping_interval=None,
    ping_timeout=None,
)


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    denom = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(denom, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / denom


def _make_env(
    env_cls: Any,
    task_bddl_file: Path,
    resolution: int,
    horizon: int | None,
    task_description: str | None = None,
) -> tuple[Any, str]:
    env_kwargs = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    if horizon is not None:
        env_kwargs["horizon"] = horizon
    env = env_cls(**env_kwargs)
    if task_description is not None:
        return env, task_description
    for line in task_bddl_file.read_text().splitlines():
        text = line.strip()
        if text.startswith("(:language "):
            return env, text.removeprefix("(:language ").removesuffix(")")
    raise RuntimeError(f"task language missing from {task_bddl_file}")


def _evo1_task_language(task_name: str) -> str:
    filename = f"{task_name}.bddl"
    if filename[0].isupper():
        scene_offset = 8 if "SCENE10" in filename else 7
        language = " ".join(filename[filename.find("SCENE") + scene_offset :].split("_"))
    else:
        language = " ".join(filename.split("_"))
    return language[: language.find(".bddl")]


def _evo1_task_names(libero_source_root: Path, suite: str) -> list[str]:
    task_map_path = libero_source_root / "libero/libero/benchmark/libero_suite_task_map.py"
    task_map = runpy.run_path(str(task_map_path))["libero_task_map"]
    task_names = task_map[suite]
    if len(task_names) != 10:
        raise RuntimeError(f"expected 10 Evo-1 tasks in {suite}, found {len(task_names)}")
    return task_names


def _run_episode(
    *,
    client: WebsocketClientPolicy,
    env: Any,
    initial_state: np.ndarray,
    task_description: str,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    dummy_action: list[float],
    async_supervision: bool = False,
) -> dict[str, object]:
    episode_start = perf_counter()
    env.reset()
    obs = env.set_init_state(initial_state)

    action_plan: deque[np.ndarray] = deque()
    episode_reward = 0.0
    episode_len = 0
    episode_roundtrip_ms: list[float] = []
    server_infer_ms: list[float] = []
    success = False
    step_count = 0
    monitor_checks = 0
    monitor_replans = 0
    monitor_stuck_count = 0
    monitor_pool = ThreadPoolExecutor(max_workers=1) if async_supervision else None
    monitor_future: Future[Any] | None = None
    recent_eef_positions: deque[np.ndarray] = deque(maxlen=4)

    for _ in range(max_steps + num_steps_wait):
        if monitor_future is not None and monitor_future.done():
            monitor_checks += 1
            needs_replan = float(monitor_future.result()) < 1e-3
            monitor_future = None
            if needs_replan and action_plan:
                monitor_stuck_count += 1
                if monitor_stuck_count >= 2:
                    action_plan.clear()
                    recent_eef_positions.clear()
                    monitor_replans += 1
                    monitor_stuck_count = 0
            else:
                monitor_stuck_count = 0

        if step_count < num_steps_wait:
            obs, reward, done, _ = env.step(dummy_action)
            episode_reward += float(reward)
            episode_len += 1
            step_count += 1
            if done:
                success = True
                break
            continue

        if not action_plan:
            roundtrip_start = perf_counter()
            result = client.infer(
                {
                    "images": [
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    ],
                    "state": np.ascontiguousarray(
                        np.concatenate(
                            (
                                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                                _quat_to_axis_angle(obs["robot0_eef_quat"]),
                                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
                            ),
                        )
                    ),
                    "task": task_description,
                }
            )
            episode_roundtrip_ms.append((perf_counter() - roundtrip_start) * 1000.0)
            server_infer_ms.append(float(result["server_timing"]["infer_ms"]))
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            action_plan.extend(action_chunk[:replan_steps])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)[:ACTION_DIM]
        obs, reward, done, _ = env.step(action.tolist())
        episode_reward += float(reward)
        episode_len += 1
        step_count += 1
        if async_supervision:
            recent_eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy())
            if monitor_pool is not None and monitor_future is None and action_plan and len(recent_eef_positions) == 4:
                monitor_future = monitor_pool.submit(
                    np.linalg.norm,
                    recent_eef_positions[-1] - recent_eef_positions[0],
                )
        if done:
            success = True
            break

    if monitor_pool is not None:
        monitor_pool.shutdown(wait=True)

    return {
        "success": success,
        "reward": episode_reward,
        "episode_len": episode_len,
        "wall_time_s": perf_counter() - episode_start,
        "mean_client_roundtrip_ms": float(np.mean(episode_roundtrip_ms)) if episode_roundtrip_ms else 0.0,
        "mean_server_infer_ms": float(np.mean(server_infer_ms)) if server_infer_ms else 0.0,
        "client_roundtrip_ms": episode_roundtrip_ms,
        "server_infer_ms": server_infer_ms,
        "monitor_checks": monitor_checks,
        "monitor_replans": monitor_replans,
    }


def run_client(
    *,
    host: str,
    port: int,
    async_supervision: bool = False,
) -> int:
    data_dir = Path(os.environ.get("PI_CPP_DATA_DIR", Path.cwd() / "data")).expanduser()
    libero_source_root = data_dir / "libero" / "LIBERO"
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_source_root.parent / ".libero")
    sys.path.insert(0, str(libero_source_root))

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    client_start = perf_counter()
    client = WebsocketClientPolicy(host=host, port=port)
    metadata = client.get_server_metadata()
    model = str(metadata["model"])
    evo1_protocol = model == "evo1"
    resolution = 448 if evo1_protocol else LIBERO_RESOLUTION
    seed = 42 if evo1_protocol else DEFAULT_SEED
    num_steps_wait = 10 if evo1_protocol else LIBERO_NUM_STEPS_WAIT
    dummy_action = [0.0] * ACTION_DIM if evo1_protocol else LIBERO_DUMMY_ACTION

    suite_results: list[dict[str, object]] = []
    all_episode_results: list[dict[str, object]] = []
    all_client_roundtrip_ms: list[float] = []
    all_server_infer_ms: list[float] = []
    total_successes = 0
    total_episodes = 0

    for task_suite_name in track(TASK_SUITE_NAMES, description="libero suites"):
        bddl_dir = Path(get_libero_path("bddl_files")) / task_suite_name
        init_states_dir = Path(get_libero_path("init_states")) / task_suite_name
        if evo1_protocol:
            task_names = _evo1_task_names(libero_source_root, task_suite_name)
            task_bddl_files = [bddl_dir / f"{task_name}.bddl" for task_name in task_names]
            task_descriptions: list[str | None] = [
                _evo1_task_language(task_name) for task_name in task_names
            ]
        else:
            task_bddl_files = sorted(bddl_dir.glob("*.bddl"))
            task_descriptions = [None] * len(task_bddl_files)
        max_steps = (
            1330
            if evo1_protocol and task_suite_name == "libero_10"
            else 350
            if evo1_protocol
            else LIBERO_MAX_STEPS[task_suite_name]
        )
        replan_steps = MODEL_REPLAN_STEPS[model]
        suite_successes = 0
        suite_episodes = 0
        task_results: list[dict[str, object]] = []

        task_rows = zip(task_bddl_files, task_descriptions, strict=True)
        for task_id, (task_bddl_file, task_description) in track(
            enumerate(task_rows),
            total=len(task_bddl_files),
            description=task_suite_name,
        ):
            with zipfile.ZipFile(init_states_dir / f"{task_bddl_file.stem}.pruned_init") as archive:
                initial_states = pickle.loads(archive.read("archive/data.pkl"))
            env, task_description = _make_env(
                OffScreenRenderEnv,
                task_bddl_file,
                resolution,
                max_steps + num_steps_wait if evo1_protocol else None,
                task_description,
            )
            env.seed(seed)

            task_successes = 0
            task_episode_results: list[dict[str, object]] = []
            for init_state_id in range(TRIALS_PER_TASK):
                episode_result = _run_episode(
                    client=client,
                    env=env,
                    initial_state=initial_states[init_state_id],
                    task_description=task_description,
                    max_steps=max_steps,
                    num_steps_wait=num_steps_wait,
                    replan_steps=replan_steps,
                    dummy_action=dummy_action,
                    async_supervision=async_supervision,
                )
                task_successes += int(episode_result["success"])
                suite_successes += int(episode_result["success"])
                total_successes += int(episode_result["success"])
                suite_episodes += 1
                total_episodes += 1
                all_client_roundtrip_ms.extend(episode_result["client_roundtrip_ms"])
                all_server_infer_ms.extend(episode_result["server_infer_ms"])

                detail = {
                    "suite": task_suite_name,
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "task": task_description,
                    "success": episode_result["success"],
                    "reward": episode_result["reward"],
                    "episode_len": episode_result["episode_len"],
                    "wall_time_s": episode_result["wall_time_s"],
                    "mean_client_roundtrip_ms": episode_result["mean_client_roundtrip_ms"],
                    "mean_server_infer_ms": episode_result["mean_server_infer_ms"],
                    "monitor_checks": episode_result["monitor_checks"],
                    "monitor_replans": episode_result["monitor_replans"],
                }
                task_episode_results.append(detail)
                all_episode_results.append(detail)
                console.print_json(data={"event": "episode_result", "episode": detail})

            env.close()
            task_results.append(
                {
                    "task_id": task_id,
                    "task": task_description,
                    "episodes": TRIALS_PER_TASK,
                    "successes": task_successes,
                    "success_rate": float(task_successes) / float(TRIALS_PER_TASK),
                    "episodes_detail": task_episode_results,
                }
            )

        suite_results.append(
            {
                "suite": task_suite_name,
                "episodes": suite_episodes,
                "successes": suite_successes,
                "success_rate": float(suite_successes) / float(suite_episodes),
                "tasks": task_results,
            }
        )
        console.print_json(data={"event": "suite_result", "suite": suite_results[-1]})

    episode_wall_time_s = [float(detail["wall_time_s"]) for detail in all_episode_results]
    monitor_checks = [int(detail["monitor_checks"]) for detail in all_episode_results]
    monitor_replans = [int(detail["monitor_replans"]) for detail in all_episode_results]
    client_overhead_ms = [
        roundtrip_ms - infer_ms
        for roundtrip_ms, infer_ms in zip(
            all_client_roundtrip_ms,
            all_server_infer_ms,
            strict=True,
        )
    ]
    summary = {
        "command": "eval",
        "model": model,
        "task_suites": TASK_SUITE_NAMES,
        "trials_per_task": TRIALS_PER_TASK,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes),
        "total_wall_time_s": perf_counter() - client_start,
        "mean_episode_wall_time_s": float(np.mean(episode_wall_time_s)) if episode_wall_time_s else 0.0,
        "mean_client_roundtrip_ms": float(np.mean(all_client_roundtrip_ms)) if all_client_roundtrip_ms else 0.0,
        "mean_server_infer_ms": float(np.mean(all_server_infer_ms)) if all_server_infer_ms else 0.0,
        "mean_client_overhead_ms": float(np.mean(client_overhead_ms)) if client_overhead_ms else 0.0,
        "async_supervision": async_supervision,
        "monitor_checks": int(np.sum(monitor_checks)),
        "monitor_replans": int(np.sum(monitor_replans)),
        "metadata": metadata,
        "suites": suite_results,
        "episodes_detail": all_episode_results,
    }
    eval_path = Path(str(metadata["model_dir"])).expanduser() / "eval.json"
    summary["eval_result_path"] = str(eval_path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(summary, indent=2) + "\n")
    console.print_json(data=summary)
    client.close()
    return 0
