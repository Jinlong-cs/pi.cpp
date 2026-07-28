"""Small first-impression dashboard for picpp runtime tasks."""

from __future__ import annotations

import json
import math
import os
import shutil
import shlex
import signal
import subprocess
import sys
import time
from importlib.metadata import distribution
from pathlib import Path
from typing import NamedTuple

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Header, ListItem, ListView, RichLog, Static

from pi_cpp.commands.client import (
    DEFAULT_SEED,
    LIBERO_DUMMY_ACTION,
    LIBERO_MAX_STEPS,
    LIBERO_NUM_STEPS_WAIT,
    LIBERO_RESOLUTION,
    MODEL_REPLAN_STEPS,
    TASK_SUITE_NAMES,
    TRIALS_PER_TASK,
)

HF_ASSET_REPO = "EddieWu577/picpp_assets"
LIBERO_SOURCE_PACKAGE = "hf-libero==0.1.3"
LIBERO_SOURCE_DISTRIBUTION = "hf-libero"
PI_CPP_HOME = Path(os.environ.get("PI_CPP_HOME", Path.cwd()))
PICPP_ROOT = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share/")).expanduser() / "picpp"
MODEL_ROOT = PICPP_ROOT / "models"
DATA_DIR = PICPP_ROOT / "data"
LIBERO_DATA_ROOT = DATA_DIR / "libero"
LIBERO_SOURCE_ROOT = LIBERO_DATA_ROOT / "LIBERO"
LIBERO_CONFIG_DIR = LIBERO_DATA_ROOT / ".libero"
LIBERO_CONFIG_FILE = LIBERO_CONFIG_DIR / "config.yaml"
IGNORED_LATENCY_KEYS = {"preprocess_ms", "postprocess_ms"}
LATENCY_RESULT_ROWS = 10
EVAL_PROGRESS_WIDTH = 24
TUI_SERVER_HOST = "127.0.0.1"
TUI_SERVER_PORT = 8000
TUI_SERVER_READY_TIMEOUT = 300.0
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
RENDER_PROFILE_STEPS = 20
MODEL_ENV = {
    "pi05": ("PI_ONNX_DIR", "PI_ENGINE_DIR"),
    "pi05_int8": ("PI05_INT8_ONNX_DIR", "PI05_INT8_ENGINE_DIR"),
    "fastwam": ("FASTWAM_ONNX_DIR", "FASTWAM_ENGINE_DIR"),
    "fastwam_int8": ("FASTWAM_INT8_ONNX_DIR", "FASTWAM_INT8_ENGINE_DIR"),
    "smolvla": ("SMOLVLA_ONNX_DIR", "SMOLVLA_ENGINE_DIR"),
    "semanticvla": ("SEMANTICVLA_ONNX_DIR", "SEMANTICVLA_ENGINE_DIR"),
    "evo1": ("EVO1_ONNX_DIR", "EVO1_ENGINE_DIR"),
    "dit4dit": ("DIT4DIT_ONNX_DIR", "DIT4DIT_ENGINE_DIR"),
    "groot": ("GROOT_ONNX_DIR", "GROOT_ENGINE_DIR"),
    "starvla": ("STARVLA_ONNX_DIR", "STARVLA_ENGINE_DIR"),
}
DEFAULT_PROMPT = "Pick the akita black bowl from table center and place it on the plate"


class Model(NamedTuple):
    name: str
    info: tuple[str, ...]
    onnx: tuple[str, ...]
    engines: tuple[str, ...]


MODELS = (
    Model(
        "PI0.5",
        (
            "VL encoder: PaliGemma (Gemma 2B + SigLIP)",
            "action decoder: Gemma 300M flow matching head",
            "input: 3 camera slots, 200 prompt tokens",
            "action: 50-step normalized chunk, 10 denoise steps, 7-DoF output",
        ),
        ("prefix_embed.onnx", "prefix_lm.onnx", "suffix_step.onnx"),
        ("prefix_embed.engine", "prefix_lm.engine", "suffix_step.engine"),
    ),
    Model(
        "PI0.5-INT8",
        (
            "VL encoder: PaliGemma (Gemma 2B + SigLIP)",
            "action decoder: Gemma 300M flow matching head",
            "INT8: fixed-weight Linear/Gemm QDQ plus suffix broadcast-KV",
            "action: 10-step normalized chunk, 10 denoise steps, 7-DoF output",
        ),
        ("prefix_embed.onnx", "prefix_lm.onnx", "suffix_step.onnx"),
        ("prefix_embed.engine", "prefix_lm.engine", "suffix_step.engine"),
    ),
    Model(
        "FastWAM",
        (
            "VL encoder: T5 text cache + Wan2.2-5B video DiT",
            "action decoder: 1B action expert DiT",
            "input: 2 LIBERO cameras, precomputed T5 context len 128",
            "action: 32-step normalized chunk, 7-DoF output",
        ),
        ("vae_image_encoder.onnx", "video_prefill.onnx", "action_step_dynamic_kv.onnx"),
        ("vae_image_encoder.engine", "video_prefill.engine", "action_step_dynamic_kv.engine"),
    ),
    Model(
        "FastWAM-INT8",
        (
            "VL encoder: T5 text cache + Wan2.2-5B video DiT",
            "action decoder: 1B action expert DiT",
            "INT8: video/action fixed-weight Linear/Gemm QDQ",
            "action: 32-step normalized chunk, 7-DoF output",
        ),
        ("vae_image_encoder.onnx", "video_prefill.onnx", "action_step_dynamic_kv.onnx"),
        ("vae_image_encoder.engine", "video_prefill.engine", "action_step_dynamic_kv.engine"),
    ),
    Model(
        "SmolVLA",
        (
            "VL encoder: SmolVLM2-500M-Video-Instruct",
            "action decoder: SmolVLA flow-matching expert",
            "input: 2 LIBERO cameras, 512px images, 48 prompt tokens",
            "action: 50-step normalized chunk, 7-DoF output",
        ),
        ("prefix_embed.onnx", "prefix_lm.onnx", "suffix_loop.onnx"),
        ("prefix_embed.engine", "prefix_lm.engine", "suffix_loop.engine"),
    ),
    Model(
        "SemanticVLA",
        (
            "VL encoder: Qwen3-VL-4B-Instruct",
            "action decoder: DiT-B flow-matching expert",
            "input: 2 LIBERO cameras, precomputed Qwen prompt cache",
            "action: 10-step normalized chunk, 7-DoF output",
        ),
        ("backbone.onnx", "action_step.onnx"),
        ("backbone.engine", "action_step.engine"),
    ),
    Model(
        "Evo-1",
        (
            "VL encoder: InternVL3-1B truncated to 14 language layers",
            "action decoder: 8-layer flow-matching transformer",
            "input: 2 LIBERO cameras plus 1 masked slot, 448px images",
            "action: 50-step chunk, 32 flow steps, 7-DoF output",
        ),
        ("backbone.onnx", "action_step.onnx"),
        ("backbone.engine", "action_step.engine"),
    ),
    Model(
        "DiT4DiT",
        (
            "VL encoder: Qwen text encoder + Cosmos video DiT",
            "action decoder: DiT4DiT action diffusion head",
            "input: 2 LIBERO cameras, cached prompt embeddings",
            "action: 8-step chunk, 10 denoise steps, 7-DoF output",
        ),
        ("cosmos_vae_encode.onnx", "cosmos_feature.onnx", "action_loop.onnx"),
        ("cosmos_vae_encode.engine", "cosmos_feature.engine", "action_loop.engine"),
    ),
    Model(
        "GR00T N1.6",
        (
            "VL encoder: Eagle2-5B backbone",
            "action decoder: GR00T N1.6 diffusion action loop",
            "input: 2 LIBERO cameras, cached prompt tokens, proprio state",
            "action: 16-step normalized chunk, 7-DoF output",
        ),
        ("backbone.onnx", "action_loop.onnx"),
        ("backbone.engine", "action_loop.engine"),
    ),
    Model(
        "StarVLA",
        (
            "VL encoder: Qwen3-VL-OFT-4B prompt-forward backbone",
            "action decoder: continuous MLP action head",
            "input: 2 LIBERO cameras, right-padded prompt cache",
            "action: 8-step normalized chunk, 7-DoF output",
        ),
        ("policy.onnx",),
        ("policy.engine",),
    ),
)


def libero_400_data_ready() -> bool:
    bddl_root = LIBERO_SOURCE_ROOT / "libero" / "libero" / "bddl_files"
    init_root = LIBERO_SOURCE_ROOT / "libero" / "libero" / "init_files"
    if not LIBERO_CONFIG_FILE.exists():
        return False
    for suite in LIBERO_SUITES:
        task_bddl_files = sorted((bddl_root / suite).glob("*.bddl"))
        if len(task_bddl_files) != 10:
            return False
        if not all(
            (init_root / suite / f"{task_bddl_file.stem}.pruned_init").exists()
            for task_bddl_file in task_bddl_files
        ):
            return False
    return True


def print_download_progress(phase: str, done: int, total: int) -> None:
    print(
        json.dumps(
            {"event": "download_progress", "phase": phase, "done": done, "total": total},
            indent=2,
        ),
        flush=True,
    )


def run_tui_download(model_dir_name: str, target_root: str) -> None:
    total = 5
    print_download_progress("model assets", 0, total)
    command = [
        "hf",
        "download",
        HF_ASSET_REPO,
        "--repo-type",
        "dataset",
        "--include",
        f"{model_dir_name}/**",
        "--local-dir",
        target_root,
    ]
    for attempt in range(3):
        result = subprocess.run(command, check=attempt == 2)
        if result.returncode == 0:
            break
        print(f"hf download failed, retrying in {10 * (attempt + 1)}s", flush=True)
        time.sleep(10 * (attempt + 1))
    print_download_progress("model assets", 1, total)

    prepare_tui_libero(start_step=1, total_steps=total)


def prepare_tui_libero(start_step: int = 0, total_steps: int = 4) -> None:
    print_download_progress("libero deps", start_step, total_steps)
    subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--extra", "libero"], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", LIBERO_SOURCE_PACKAGE],
        check=True,
    )
    print_download_progress("libero deps", start_step + 1, total_steps)

    print_download_progress("libero env/data", start_step + 1, total_steps)
    libero_source = Path(distribution(LIBERO_SOURCE_DISTRIBUTION).locate_file("libero"))
    LIBERO_SOURCE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if LIBERO_SOURCE_ROOT.is_symlink() or LIBERO_SOURCE_ROOT.is_file():
        LIBERO_SOURCE_ROOT.unlink()
    elif LIBERO_SOURCE_ROOT.exists():
        shutil.rmtree(LIBERO_SOURCE_ROOT)
    shutil.copytree(libero_source, LIBERO_SOURCE_ROOT / "libero", symlinks=True)
    print_download_progress("libero env/data", start_step + 2, total_steps)

    benchmark_root = LIBERO_SOURCE_ROOT / "libero" / "libero"
    LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LIBERO_CONFIG_FILE.write_text(
        "\n".join(
            (
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {LIBERO_SOURCE_ROOT / 'libero' / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            )
        )
    )

    print_download_progress("libero assets", start_step + 2, total_steps)
    assets_dir = benchmark_root / "assets"
    if all(
        (assets_dir / path).is_file()
        for path in (
            Path("scenes/libero_tabletop_base_style.xml"),
            Path("stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"),
            Path("stable_scanned_objects/plate/plate.xml"),
        )
    ):
        print("libero assets are ready; skip asset download", flush=True)
    else:
        for attempt in range(3):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from libero.libero.utils.download_utils import "
                    "download_assets_from_huggingface; "
                    "download_assets_from_huggingface(sys.argv[1])",
                    str(assets_dir),
                ],
                check=attempt == 2,
            )
            if result.returncode == 0:
                break
            print(
                f"libero asset download failed, retrying in {10 * (attempt + 1)}s",
                flush=True,
            )
            time.sleep(10 * (attempt + 1))
    print_download_progress("libero assets", start_step + 3, total_steps)

    print_download_progress("libero config", start_step + 3, total_steps)
    print_download_progress("libero config", start_step + 4, total_steps)


def print_libero_eval_estimate(model: str, model_dir: str) -> None:
    sys.path.insert(0, str(LIBERO_SOURCE_ROOT))
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    latency = json.loads((Path(model_dir) / "latency.json").read_text())
    infer_ms = float(latency["infer_ms"])
    evo1_protocol = model == "evo1"
    resolution = 448 if evo1_protocol else LIBERO_RESOLUTION
    seed = 42 if evo1_protocol else DEFAULT_SEED
    dummy_action = [0.0] * 7 if evo1_protocol else LIBERO_DUMMY_ACTION
    step_values = []
    for suite in TASK_SUITE_NAMES:
        task_bddl_file = sorted((Path(get_libero_path("bddl_files")) / suite).glob("*.bddl"))[0]
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(seed)
        env.reset()
        for _ in range(RENDER_PROFILE_STEPS):
            start = time.perf_counter()
            _obs, _reward, done, _info = env.step(dummy_action)
            step_values.append((time.perf_counter() - start) * 1000.0)
            if done:
                break
        env.close()

    env_step_ms = sum(step_values) / len(step_values)
    suites = {}
    total_seconds = 0.0
    total_episodes = 0
    for suite in TASK_SUITE_NAMES:
        task_count = len(list((Path(get_libero_path("bddl_files")) / suite).glob("*.bddl")))
        max_steps = (
            1330
            if evo1_protocol and suite == "libero_10"
            else 350
            if evo1_protocol
            else LIBERO_MAX_STEPS[suite]
        )
        wait_steps = 10 if evo1_protocol else LIBERO_NUM_STEPS_WAIT
        replan_steps = MODEL_REPLAN_STEPS[model]
        episodes = task_count * TRIALS_PER_TASK
        calls = math.ceil(max_steps / replan_steps)
        seconds = episodes * ((wait_steps + max_steps) * env_step_ms + calls * infer_ms) / 1000.0
        suites[suite] = {"episodes": episodes, "seconds": seconds}
        total_seconds += seconds
        total_episodes += episodes

    minutes, sec = divmod(max(0.0, total_seconds), 60.0)
    hours, minute = divmod(int(minutes), 60)
    estimated_time = f"{hours}h{minute:02d}m{sec:04.1f}s" if hours else f"{minute}m{sec:04.1f}s"
    print(
        json.dumps(
            {
                "event": "eval_estimate",
                "model": model,
                "episodes": total_episodes,
                "infer_ms": infer_ms,
                "env_step_ms": env_step_ms,
                "estimated_seconds": total_seconds,
                "estimated_time": estimated_time,
                "suites": suites,
            },
            indent=2,
        )
    )


class PicppTui(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #left { width: 1fr; }
    #right { width: 3fr; }
    #models { height: 10; border: round $primary; }
    #results { height: 1fr; border: round $secondary; padding: 0 1; }
    #info { height: 1fr; border: round $accent; padding: 0 1; }
    #assets { height: 1fr; border: round $secondary; }
    #log { height: 1fr; border: round $secondary; }
    #keys { height: 1; padding: 0 1; background: $panel; }
    """
    BINDINGS = [
        Binding("up", "previous_model", show=False, priority=True),
        Binding("down", "next_model", show=False, priority=True),
        Binding("q", "quit", show=False),
        Binding("d", "download", show=False),
        Binding("b", "build", show=False),
        Binding("l", "latency", show=False),
        Binding("e", "eval", show=False),
        Binding("s", "stop", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.model_index = 0
        self.model = MODELS[0]
        self.runtime = ""
        self.asset_runtime = ""
        self.model_dir = MODEL_ROOT
        self.make = ""
        self.prompt = ""
        self.process: subprocess.Popen[str] | None = None
        self.results: dict[str, dict[str, object]] = {}
        self.json_lines: list[str] = []
        self.select_model(0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield ListView(
                    *[ListItem(Static(model.name)) for model in MODELS],
                    id="models",
                )
                yield Static(id="results")
            with Vertical(id="right"):
                yield Static(id="info")
                yield DataTable(id="assets")
                yield RichLog(
                    id="log", max_lines=1000, wrap=False, highlight=False, auto_scroll=True
                )
        yield Static(
            Text.assemble(
                ("up/down", "bold cyan"),
                " model | ",
                ("d", "bold green"),
                " Download ONNX | ",
                ("b", "bold yellow"),
                " Build engine | ",
                ("l", "bold magenta"),
                " Latency | ",
                ("e", "bold blue"),
                " Eval | ",
                ("s", "bold red"),
                " Stop | ",
                ("q", "bold red"),
                " Quit",
            ),
            id="keys",
        )

    def on_mount(self) -> None:
        self.title = "picpp TUI"
        self.query_one("#models", ListView).border_title = "Models"
        self.query_one("#results", Static).border_title = "Results"
        self.query_one("#info", Static).border_title = "Info"
        self.query_one("#assets", DataTable).border_title = "Assets"
        self.query_one("#log", RichLog).border_title = "run log"
        self.query_one("#models", ListView).index = 0
        self.query_one("#models", ListView).focus()
        self.query_one("#assets", DataTable).cursor_type = "row"
        self.write_log(f"picpp TUI ready. data root: {PICPP_ROOT}")
        self.refresh_screen()

    def on_unmount(self) -> None:
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)

    def action_stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.write_log("no process running")
            return
        self.results[self.model.name]["status"] = "stopping"
        self.refresh_screen()
        self.write_log(f"stopping process: pid {self.process.pid}")
        os.killpg(self.process.pid, signal.SIGTERM)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self.select_model(event.list_view.index or 0)
        self.refresh_screen()

    def action_previous_model(self) -> None:
        self.select_model((self.model_index - 1) % len(MODELS))
        self.query_one("#models", ListView).index = self.model_index
        self.refresh_screen()

    def action_next_model(self) -> None:
        self.select_model((self.model_index + 1) % len(MODELS))
        self.query_one("#models", ListView).index = self.model_index
        self.refresh_screen()

    def select_model(self, index: int) -> None:
        self.model_index = index
        self.model = MODELS[index]
        if self.model.name == "PI0.5":
            self.runtime = "pi05"
            self.model_dir = MODEL_ROOT / "pi05_libero"
            self.asset_runtime = "pi05"
            self.make = "engines_pi05"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "PI0.5-INT8":
            self.runtime = "pi05"
            self.asset_runtime = "pi05_int8"
            self.model_dir = MODEL_ROOT / "pi05_libero_int8"
            self.make = "engines_pi05_int8"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "FastWAM":
            self.runtime = "fastwam"
            self.asset_runtime = "fastwam"
            self.model_dir = MODEL_ROOT / "fastwam_libero"
            self.make = "engines_fastwam"
            self.prompt = "put the bowl on the plate"
        elif self.model.name == "FastWAM-INT8":
            self.runtime = "fastwam"
            self.asset_runtime = "fastwam_int8"
            self.model_dir = MODEL_ROOT / "fastwam_libero_int8"
            self.make = "engines_fastwam_int8"
            self.prompt = "put the bowl on the plate"
        elif self.model.name == "SmolVLA":
            self.runtime = "smolvla"
            self.asset_runtime = "smolvla"
            self.model_dir = MODEL_ROOT / "smolvla_libero"
            self.make = "engines_smolvla"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "SemanticVLA":
            self.runtime = "semanticvla"
            self.asset_runtime = "semanticvla"
            self.model_dir = MODEL_ROOT / "semanticvla_libero"
            self.make = "engines_semanticvla"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "Evo-1":
            self.runtime = "evo1"
            self.asset_runtime = "evo1"
            self.model_dir = MODEL_ROOT / "evo1_libero"
            self.make = "engines_evo1"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "DiT4DiT":
            self.runtime = "dit4dit"
            self.asset_runtime = "dit4dit"
            self.model_dir = MODEL_ROOT / "dit4dit_libero"
            self.make = "engines_dit4dit"
            self.prompt = DEFAULT_PROMPT
        elif self.model.name == "GR00T N1.6":
            self.runtime = "groot"
            self.asset_runtime = "groot"
            self.model_dir = MODEL_ROOT / "groot_n1d6_libero"
            self.make = "engines_groot"
            self.prompt = DEFAULT_PROMPT
        else:
            self.runtime = "starvla"
            self.asset_runtime = "starvla"
            self.model_dir = MODEL_ROOT / "starvla_libero"
            self.make = "engines_starvla"
            self.prompt = DEFAULT_PROMPT
        result_data = self.results.setdefault(self.model.name, {})
        if str(result_data.get("status", "")).startswith("running "):
            return
        latency_path = self.model_dir / "latency.json"
        if latency_path.exists():
            result_data.update(json.loads(latency_path.read_text()))
        eval_path = self.model_dir / "eval.json"
        if eval_path.exists():
            data = json.loads(eval_path.read_text())
            result_data["eval_done"] = int(data["episodes"])
            result_data["eval_total"] = int(data["episodes"])
            result_data["eval_successes"] = int(data["successes"])
            result_data["eval_remaining_time"] = "0m00.0s"
            result_data["eval_suites"] = {
                str(suite["suite"]): {
                    "done": int(suite["episodes"]),
                    "successes": int(suite["successes"]),
                    "total": int(suite["episodes"]),
                }
                for suite in data["suites"]
            }

    def refresh_screen(self) -> None:
        body = Text()
        body.append(f"{self.model.name}\n", style="bold cyan")
        body.append(f"assets: {self.model_dir}\n")
        body.append(f"HF: {HF_ASSET_REPO}\n\n", style="dim")
        for line in self.model.info:
            body.append(f"{line}\n")
        self.query_one("#info", Static).update(body)

        assets = self.query_one("#assets", DataTable)
        assets.clear(columns=True)
        assets.add_columns("group", "item", "status")
        for name in self.model.onnx:
            path = self.model_dir / "onnx" / str(name)
            assets.add_row(
                "onnx", str(name), "[green]present[/]" if path.exists() else "[red]missing[/]"
            )
        for name in self.model.engines:
            path = self.model_dir / "engines" / str(name)
            assets.add_row(
                "engine", str(name), "[green]present[/]" if path.exists() else "[red]missing[/]"
            )
        for name in ("latency.json", "eval.json"):
            path = self.model_dir / name
            assets.add_row(
                "result", name, "[green]present[/]" if path.exists() else "[red]missing[/]"
            )
        env_ready = (LIBERO_SOURCE_ROOT / "libero" / "libero" / "envs").is_dir()
        data_ready = libero_400_data_ready()
        assets.add_row("env", "libero", "[green]present[/]" if env_ready else "[red]missing[/]")
        assets.add_row(
            "data",
            "libero 400 episodes",
            "[green]present[/]" if data_ready else "[red]missing[/]",
        )

        result_data = self.results[self.model.name]
        result = Text()
        result.append(f"model: {self.model.name}\n")
        result.append(f"status: {result_data.get('status', 'idle')}\n\n")
        result.append("Latency\n", style="bold cyan")
        latency_keys = [
            key for key in result_data if key.endswith("_ms") and key not in IGNORED_LATENCY_KEYS
        ]
        for index in range(LATENCY_RESULT_ROWS):
            if index < len(latency_keys):
                key = latency_keys[index]
                result.append(f"{key}: {float(result_data[key]):.3f} ms\n")
            else:
                result.append("\n")

        result.append("\nLIBERO 400 episodes\n", style="bold cyan")
        eval_total = int(result_data.get("eval_total", 0))
        eval_done = int(result_data.get("eval_done", 0))
        successes = int(result_data.get("eval_successes", 0))
        if eval_total or eval_done:
            total = eval_total or 400
            result.append(
                f"completed: {eval_done}/{total} | success: {successes}/{eval_done} "
                f"({100.0 * float(successes) / float(eval_done) if eval_done else 0.0:.1f}%)\n"
            )
        else:
            result.append("completed: 0/400 | success: 0/0 (0.0%)\n")
        suites = result_data.get("eval_suites", {})
        for suite in LIBERO_SUITES:
            suite_data = suites.get(suite, {}) if isinstance(suites, dict) else {}
            suite_done = int(suite_data.get("done", 0))
            suite_successes = int(suite_data.get("successes", 0))
            suite_total = int(suite_data.get("total", 100))
            result.append(
                f"{suite}: {suite_done}/{suite_total} | success: "
                f"{suite_successes}/{suite_done} "
                f"({100.0 * float(suite_successes) / float(suite_done) if suite_done else 0.0:.1f}%)\n"
            )
        if "eval_current_suite" in result_data:
            current_success = "yes" if result_data.get("eval_current_success") else "no"
            result.append(
                f"current: {result_data['eval_current_suite']} "
                f"task {int(result_data.get('eval_current_task_id', 0))} "
                f"init {int(result_data.get('eval_current_init_state_id', 0))} "
                f"success {current_success}\n"
            )
        self.query_one("#results", Static).update(result)

    def progress_bar(self, done: int, total: int) -> str:
        filled = int(EVAL_PROGRESS_WIDTH * min(done, total) / total) if total else 0
        return "[" + "#" * filled + "-" * (EVAL_PROGRESS_WIDTH - filled) + "]"

    def action_download(self) -> None:
        env = os.environ.copy()
        env["PI_CPP_DATA_DIR"] = str(DATA_DIR)
        env["PYTHONPATH"] = f"{PI_CPP_HOME / 'python'}" + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        self.run_command(
            [
                sys.executable,
                "-c",
                "import sys; from pi_cpp.tui import run_tui_download; "
                "run_tui_download(sys.argv[1], sys.argv[2])",
                self.model_dir.name,
                str(self.model_dir.parent),
            ],
            "download",
            env=env,
        )

    def action_build(self) -> None:
        env = os.environ.copy()
        env["PI_CPP_DATA_DIR"] = str(DATA_DIR)
        env["PYTHONPATH"] = f"{PI_CPP_HOME / 'python'}" + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        onnx_key, engine_key = MODEL_ENV[self.asset_runtime]
        env[onnx_key] = str(self.model_dir / "onnx")
        env[engine_key] = str(self.model_dir / "engines")
        self.run_command(["make", "-B", self.make], "build", env=env)

    def action_latency(self) -> None:
        env = os.environ.copy()
        env["PI_CPP_DATA_DIR"] = str(DATA_DIR)
        env["PICPP_LATENCY_PROGRESS"] = "1"
        self.run_command(
            [
                "picpp",
                "latency",
                "--model",
                self.runtime,
                "--model-dir",
                str(self.model_dir),
                "--prompt",
                self.prompt,
            ],
            "latency",
            env=env,
        )

    def action_eval(self) -> None:
        env = os.environ.copy()
        env["PI_CPP_DATA_DIR"] = str(DATA_DIR)
        env["PICPP_LATENCY_PROGRESS"] = "1"
        env["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
        python_paths = [str(PI_CPP_HOME / "python"), str(LIBERO_SOURCE_ROOT)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_paths)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        server_command = [
            "picpp",
            "server",
            "--model",
            self.runtime,
            "--model-dir",
            str(self.model_dir),
            "--host",
            TUI_SERVER_HOST,
            "--port",
            str(TUI_SERVER_PORT),
        ]
        wait_command = [
            sys.executable,
            "-c",
            (
                "import http.client, sys, time\n"
                f"host = {TUI_SERVER_HOST!r}\n"
                f"port = {TUI_SERVER_PORT}\n"
                f"deadline = time.monotonic() + {TUI_SERVER_READY_TIMEOUT!r}\n"
                "last_error = None\n"
                "while True:\n"
                "    connection = http.client.HTTPConnection(host, port, timeout=1.0)\n"
                "    try:\n"
                "        connection.request('GET', '/healthz')\n"
                "        response = connection.getresponse()\n"
                "        response.read()\n"
                "        if response.status == 200:\n"
                "            break\n"
                "        last_error = RuntimeError(f'healthz status {response.status}')\n"
                "    except (OSError, http.client.HTTPException) as exc:\n"
                "        last_error = exc\n"
                "    finally:\n"
                "        connection.close()\n"
                "    if time.monotonic() > deadline:\n"
                "        print(f'picpp server did not become ready: {last_error}', file=sys.stderr)\n"
                "        sys.exit(1)\n"
                "    time.sleep(1.0)\n"
            ),
        ]
        client_command = [
            "picpp",
            "client",
            "--host",
            TUI_SERVER_HOST,
            "--port",
            str(TUI_SERVER_PORT),
        ]
        latency_command = [
            "picpp",
            "latency",
            "--model",
            self.runtime,
            "--model-dir",
            str(self.model_dir),
            "--prompt",
            self.prompt,
        ]
        estimate_command = [
            sys.executable,
            "-c",
            "import sys; from pi_cpp.tui import print_libero_eval_estimate; "
            "print_libero_eval_estimate(sys.argv[1], sys.argv[2])",
            self.runtime,
            str(self.model_dir),
        ]
        prepare_command = [
            sys.executable,
            "-c",
            "from pi_cpp.tui import prepare_tui_libero; prepare_tui_libero()",
        ]
        latency_profile = self.model_dir / "latency.json"
        latency_prefix = "" if latency_profile.exists() else shlex.join(latency_command) + "; "
        self.run_command(
            [
                "sh",
                "-lc",
                "set -e; "
                + latency_prefix
                + shlex.join(prepare_command)
                + "; "
                + shlex.join(estimate_command)
                + "; "
                + shlex.join(server_command)
                + " & server_pid=$!; "
                + 'trap \'kill -TERM "$server_pid" 2>/dev/null || true; '
                + 'wait "$server_pid" 2>/dev/null || true\' EXIT; '
                + shlex.join(wait_command)
                + " && "
                + shlex.join(client_command),
            ],
            "eval",
            env=env,
        )

    def run_command(self, command: list[str], kind: str, env: dict[str, str] | None = None) -> None:
        if self.process and self.process.poll() is None:
            self.write_log(f"process still running: pid {self.process.pid}")
            return
        runtime = self.model.name
        result_data = self.results[runtime]
        result_data["status"] = f"running {kind}"
        if kind == "latency":
            for key in list(result_data):
                if key.endswith("_ms") or key.startswith("latency_"):
                    result_data.pop(key)
        elif kind == "eval":
            for key in list(result_data):
                if key.startswith(("eval_", "libero_")):
                    result_data.pop(key, None)
            result_data["eval_done"] = 0
            result_data["eval_total"] = 400
            result_data["eval_successes"] = 0
            result_data["eval_suites"] = {
                suite: {"done": 0, "successes": 0, "total": 100} for suite in LIBERO_SUITES
            }
        self.json_lines = []
        self.refresh_screen()
        self.run_worker(
            lambda: self.run_subprocess(command, env or os.environ.copy(), kind, runtime),
            thread=True,
            exclusive=True,
        )

    def run_subprocess(
        self, command: list[str], env: dict[str, str], kind: str, runtime: str
    ) -> None:
        self.call_from_thread(
            self.write_log, "$ " + " ".join(shlex.quote(part) for part in command)
        )
        self.process = subprocess.Popen(
            command,
            cwd=PI_CPP_HOME,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        for line in self.process.stdout:
            stripped = line.rstrip()
            if kind in {"download", "latency", "eval"} and (
                self.json_lines or stripped.startswith("{")
            ):
                self.json_lines.append(stripped)
                try:
                    data = json.loads("\n".join(self.json_lines))
                except json.JSONDecodeError:
                    continue
                self.json_lines = []
                self.call_from_thread(lambda data=data: self.update_result(runtime, data))
                self.call_from_thread(self.refresh_screen)
                continue
            self.call_from_thread(self.write_log, stripped)
        return_code = self.process.wait()
        self.call_from_thread(self.write_log, f"[exit {return_code}]")
        self.results[runtime]["status"] = (
            "completed" if return_code == 0 else f"failed ({return_code})"
        )
        self.process = None
        self.call_from_thread(self.refresh_screen)

    def write_log(self, line: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.query_one("#log", RichLog).write(f"[{timestamp}] {line}", scroll_end=True)

    def update_result(self, runtime: str, data: dict[str, object]) -> None:
        result_data = self.results[runtime]
        event = data.get("event")
        if event == "download_progress":
            result_data["download_phase"] = str(data["phase"])
            result_data["download_done"] = int(data["done"])
            result_data["download_total"] = int(data["total"])
            self.write_log(
                "download progress: "
                f"{data['phase']} "
                f"{self.progress_bar(int(data['done']), int(data['total']))} "
                f"{int(data['done'])}/{int(data['total'])}"
            )
            return

        if event == "latency_progress":
            result_data["latency_phase"] = str(data["phase"])
            result_data["latency_done"] = int(data["done"])
            result_data["latency_total"] = int(data["total"])
            self.write_log(
                "latency progress: "
                f"{data['phase']} "
                f"{self.progress_bar(int(data['done']), int(data['total']))} "
                f"{int(data['done'])}/{int(data['total'])}"
            )
            return

        if event == "eval_estimate":
            result_data["eval_total"] = int(data["episodes"])
            result_data["eval_estimated_seconds"] = float(data["estimated_seconds"])
            result_data["eval_estimated_time"] = str(data["estimated_time"])
            self.write_log(f"estimated eval time: {data['estimated_time']}")
            return

        if event == "episode_result":
            episode = data["episode"]
            if not isinstance(episode, dict):
                return
            result_data["eval_current_suite"] = str(episode["suite"])
            result_data["eval_current_task_id"] = int(episode["task_id"])
            result_data["eval_current_init_state_id"] = int(episode["init_state_id"])
            result_data["eval_current_success"] = bool(episode["success"])
            result_data["eval_done"] = int(result_data.get("eval_done", 0)) + 1
            result_data["eval_successes"] = int(result_data.get("eval_successes", 0)) + int(
                bool(episode["success"])
            )
            suite_name = str(episode["suite"])
            suites = result_data.setdefault(
                "eval_suites",
                {suite: {"done": 0, "successes": 0, "total": 100} for suite in LIBERO_SUITES},
            )
            if isinstance(suites, dict):
                suite_data = suites.setdefault(
                    suite_name, {"done": 0, "successes": 0, "total": 100}
                )
                suite_data["done"] = int(suite_data.get("done", 0)) + 1
                suite_data["successes"] = int(suite_data.get("successes", 0)) + int(
                    bool(episode["success"])
                )
            eval_total = int(result_data.get("eval_total", 400))
            estimated_s = float(result_data.get("eval_estimated_seconds", 0.0))
            if estimated_s:
                remaining_s = estimated_s * (1.0 - int(result_data["eval_done"]) / eval_total)
                result_data["eval_remaining_time"] = self.hms(remaining_s)
            self.write_log(
                "eval progress: "
                f"{self.progress_bar(int(result_data['eval_done']), eval_total)} "
                f"{result_data['eval_done']}/{eval_total} "
                f"success {result_data['eval_successes']}/{result_data['eval_done']} "
                f"remaining {result_data.get('eval_remaining_time', '-')}"
            )
            return

        if event == "suite_result":
            suite = data["suite"]
            if not isinstance(suite, dict):
                return
            suites = result_data.setdefault(
                "eval_suites",
                {name: {"done": 0, "successes": 0, "total": 100} for name in LIBERO_SUITES},
            )
            if isinstance(suites, dict):
                suites[str(suite["suite"])] = {
                    "done": int(suite["episodes"]),
                    "successes": int(suite["successes"]),
                    "total": int(suite["episodes"]),
                }
            return

        if {"episodes", "successes", "success_rate", "suites"}.issubset(data):
            result_data["eval_done"] = int(data["episodes"])
            result_data["eval_total"] = int(data["episodes"])
            result_data["eval_successes"] = int(data["successes"])
            result_data["eval_remaining_time"] = "0m00.0s"
            result_data["eval_suites"] = {
                str(suite["suite"]): {
                    "done": int(suite["episodes"]),
                    "successes": int(suite["successes"]),
                    "total": int(suite["episodes"]),
                }
                for suite in data["suites"]
            }
            return

        result_data.update(data)

    def hms(self, seconds: float) -> str:
        minutes, sec = divmod(max(0.0, seconds), 60.0)
        hours, minute = divmod(int(minutes), 60)
        return f"{hours}h{minute:02d}m{sec:04.1f}s" if hours else f"{minute}m{sec:04.1f}s"


def main() -> int:
    PicppTui().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
