"""Python entrypoints for pi.cpp."""

__all__ = [
    "Dit4DitOfflineRunner",
    "Dit4DitResult",
    "Dit4DitRunnerWrapper",
    "Evo1OfflineRunner",
    "Evo1Result",
    "Evo1RunnerWrapper",
    "FastWamOfflineRunner",
    "FastWamResult",
    "FastWamRunnerWrapper",
    "GrootOfflineRunner",
    "GrootResult",
    "GrootRunnerWrapper",
    "Pi05OfflineRunner",
    "Pi05Result",
    "Pi05RunnerWrapper",
    "Pi06AirbotRunnerWrapper",
    "SemanticVlaOfflineRunner",
    "SemanticVlaResult",
    "SemanticVlaRunnerWrapper",
    "SmolVlaOfflineRunner",
    "SmolVlaResult",
    "SmolVlaRunnerWrapper",
    "StarVlaOfflineRunner",
    "StarVlaResult",
    "StarVlaRunnerWrapper",
    "build_fastwam_runner",
    "build_dit4dit_runner",
    "build_evo1_runner",
    "build_groot_runner",
    "build_pi05_runner",
    "build_pi06_airbot_runner",
    "build_pi06_heterogeneous_runner",
    "build_pi06_rtc_runner",
    "build_semanticvla_runner",
    "build_smolvla_runner",
    "build_starvla_runner",
]


def __getattr__(name: str):
    if name in {
        "FastWamRunnerWrapper",
        "Dit4DitRunnerWrapper",
        "Evo1RunnerWrapper",
        "GrootRunnerWrapper",
        "Pi05RunnerWrapper",
        "Pi06AirbotRunnerWrapper",
        "Pi06HeterogeneousRunnerWrapper",
        "Pi06RtcRunnerWrapper",
        "SemanticVlaRunnerWrapper",
        "SmolVlaRunnerWrapper",
        "StarVlaRunnerWrapper",
        "build_fastwam_runner",
        "build_dit4dit_runner",
        "build_evo1_runner",
        "build_groot_runner",
        "build_pi05_runner",
        "build_pi06_airbot_runner",
        "build_pi06_heterogeneous_runner",
    "build_pi06_heterogeneous_runner",
        "build_pi06_rtc_runner",
        "build_semanticvla_runner",
        "build_smolvla_runner",
        "build_starvla_runner",
    }:
        from pi_cpp.runner_wrapper import (
            FastWamRunnerWrapper,
            Dit4DitRunnerWrapper,
            Evo1RunnerWrapper,
            GrootRunnerWrapper,
            Pi05RunnerWrapper,
            SemanticVlaRunnerWrapper,
            SmolVlaRunnerWrapper,
            StarVlaRunnerWrapper,
            build_fastwam_runner,
            build_dit4dit_runner,
            build_evo1_runner,
            build_groot_runner,
            build_pi05_runner,
            build_semanticvla_runner,
            build_smolvla_runner,
            build_starvla_runner,
        )
        from pi_cpp.pi06_airbot import Pi06AirbotRunnerWrapper, build_pi06_airbot_runner
        from pi_cpp.pi06_heterogeneous import (
            Pi06HeterogeneousRunnerWrapper,
            build_pi06_heterogeneous_runner,
        )
        from pi_cpp.pi06_rtc import Pi06RtcRunnerWrapper, build_pi06_rtc_runner

        globals()["FastWamRunnerWrapper"] = FastWamRunnerWrapper
        globals()["Dit4DitRunnerWrapper"] = Dit4DitRunnerWrapper
        globals()["Evo1RunnerWrapper"] = Evo1RunnerWrapper
        globals()["GrootRunnerWrapper"] = GrootRunnerWrapper
        globals()["Pi05RunnerWrapper"] = Pi05RunnerWrapper
        globals()["Pi06AirbotRunnerWrapper"] = Pi06AirbotRunnerWrapper
        globals()["Pi06HeterogeneousRunnerWrapper"] = Pi06HeterogeneousRunnerWrapper
        globals()["Pi06RtcRunnerWrapper"] = Pi06RtcRunnerWrapper
        globals()["SemanticVlaRunnerWrapper"] = SemanticVlaRunnerWrapper
        globals()["SmolVlaRunnerWrapper"] = SmolVlaRunnerWrapper
        globals()["StarVlaRunnerWrapper"] = StarVlaRunnerWrapper
        globals()["build_fastwam_runner"] = build_fastwam_runner
        globals()["build_dit4dit_runner"] = build_dit4dit_runner
        globals()["build_evo1_runner"] = build_evo1_runner
        globals()["build_groot_runner"] = build_groot_runner
        globals()["build_pi05_runner"] = build_pi05_runner
        globals()["build_pi06_airbot_runner"] = build_pi06_airbot_runner
        globals()["build_pi06_heterogeneous_runner"] = build_pi06_heterogeneous_runner
        globals()["build_pi06_rtc_runner"] = build_pi06_rtc_runner
        globals()["build_semanticvla_runner"] = build_semanticvla_runner
        globals()["build_smolvla_runner"] = build_smolvla_runner
        globals()["build_starvla_runner"] = build_starvla_runner
        return globals()[name]

    if name not in __all__:
        raise AttributeError(name)
    import importlib

    _native = importlib.import_module("._native", __name__)

    value = getattr(_native, name)
    globals()[name] = value
    return value
