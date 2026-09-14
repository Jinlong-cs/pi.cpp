"""Unified pi.cpp command-line entrypoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import tyro
from rich.traceback import install as install_rich_traceback

from pi_cpp.commands.latency import run_latency
from pi_cpp.commands.package import PackageConfig
from pi_cpp.commands.server import run_server
from pi_cpp.commands.toolchain import BuildConfig, CalibrateConfig, GraphConfig

EvalModel = Literal["pi05", "pi06_heterogeneous", "pi06_rtc", "fastwam", "semanticvla", "smolvla", "dit4dit", "groot", "starvla"]
LatencyModel = Literal["pi05", "pi06_airbot", "pi06_heterogeneous", "pi06_rtc", "fastwam", "semanticvla", "evo1", "smolvla", "dit4dit", "groot", "starvla"]
ServerModel = Literal["pi05", "pi06_airbot", "pi06_heterogeneous", "pi06_rtc", "fastwam", "semanticvla", "evo1", "smolvla", "dit4dit", "groot", "starvla"]
InferModel = Literal["pi06_heterogeneous", "pi06_rtc"]


@dataclass(kw_only=True)
class EvalConfig:
    model: EvalModel
    dataset: str
    model_dir: Path | None = None


@dataclass(kw_only=True)
class LatencyConfig:
    model: LatencyModel
    model_dir: Path | None = None
    prompt: str = "Pick the akita black bowl from table center and place it on the plate"


@dataclass(kw_only=True)
class InferConfig:
    model: InferModel
    model_dir: Path | None = None
    case_dir: Path | None = None
    prompt: str | None = None
    state: str | None = None
    images: list[Path] | None = None
    embodiment_id: int | None = None
    noise_file: Path | None = None
    output: Path | None = None
    delay: int | None = None
    action_prefix_file: Path | None = None


@dataclass(kw_only=True)
class ServerConfig:
    model: ServerModel
    model_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(kw_only=True)
class ClientConfig:
    host: str
    port: int = 8000
    async_supervision: bool = False


Command = (
    Annotated[EvalConfig, tyro.conf.subcommand(name="eval")]
    | Annotated[InferConfig, tyro.conf.subcommand(name="infer")]
    | Annotated[LatencyConfig, tyro.conf.subcommand(name="latency")]
    | Annotated[PackageConfig, tyro.conf.subcommand(name="package")]
    | Annotated[GraphConfig, tyro.conf.subcommand(name="graph")]
    | Annotated[CalibrateConfig, tyro.conf.subcommand(name="calibrate")]
    | Annotated[BuildConfig, tyro.conf.subcommand(name="build")]
    | Annotated[ServerConfig, tyro.conf.subcommand(name="server")]
    | Annotated[ClientConfig, tyro.conf.subcommand(name="client")]
)


def main(argv: list[str] | None = None) -> int:
    install_rich_traceback(show_locals=False)
    command = tyro.cli(Command, args=list(sys.argv[1:] if argv is None else argv), prog="picpp")

    if isinstance(command, EvalConfig):
        from pi_cpp.commands.eval import run_eval

        return run_eval(model=command.model, dataset=command.dataset, model_dir=command.model_dir)

    if isinstance(command, InferConfig):
        from pi_cpp.commands.infer import run_infer

        return run_infer(
            model=command.model,
            model_dir=command.model_dir,
            case_dir=command.case_dir,
            prompt=command.prompt,
            state=command.state,
            images=command.images or [],
            embodiment_id=command.embodiment_id,
            noise_file=command.noise_file,
            output=command.output,
            delay=command.delay,
            action_prefix_file=command.action_prefix_file,
        )

    if isinstance(command, LatencyConfig):
        return run_latency(model=command.model, model_dir=command.model_dir, prompt=command.prompt)

    if isinstance(command, PackageConfig):
        from pi_cpp.commands.package import run_package

        return run_package(command)

    if isinstance(command, (GraphConfig, CalibrateConfig, BuildConfig)):
        from pi_cpp.commands.toolchain import run_toolchain

        return run_toolchain(command)

    if isinstance(command, ServerConfig):
        return run_server(
            model=command.model,
            model_dir=command.model_dir,
            host=command.host,
            port=command.port,
        )

    if isinstance(command, ClientConfig):
        from pi_cpp.commands.client import run_client

        return run_client(
            host=command.host,
            port=command.port,
            async_supervision=command.async_supervision,
        )


if __name__ == "__main__":
    sys.exit(main())
