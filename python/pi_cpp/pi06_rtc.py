"""Back-compat shim for the PI0.6 RTC deployment contract.

The RTC ABI (``delay`` + ``action_prefix`` suffix inputs, per-step pin,
delay=0 reducing bit-for-bit to the plain heterogeneous semantics) now
lives in ``pi06_heterogeneous`` and is driven by the export manifest's
``rtc`` section — not by a separate class. This module preserves the
historical import surface: ``Pi06RtcRunnerWrapper`` (a fail-closed
subclass that refuses non-RTC manifests) and ``build_pi06_rtc_runner``.
"""

from __future__ import annotations

from pathlib import Path

from pi_cpp.pi06_heterogeneous import (
    ACTION_INDEX_MAPS,
    DEFAULT_PI06_HETEROGENEOUS_MODEL_DIR,
    Pi06HeterogeneousRunnerWrapper,
    Pi06HeterogeneousSpec,
)

DEFAULT_PI06_RTC_MODEL_DIR = Path("assets/pi06_rtc")


class Pi06RtcRunnerWrapper(Pi06HeterogeneousRunnerWrapper):
    """The RTC variant, selected by the manifest; fails closed on non-RTC manifests."""

    def __init__(self, model_dir: str | Path = DEFAULT_PI06_RTC_MODEL_DIR) -> None:
        super().__init__(model_dir=model_dir)
        if not self.rtc_enabled:
            raise ValueError("pi06_rtc requires a manifest with rtc.training_time_rtc=True")


def build_pi06_rtc_runner(*, model_dir: str | Path | None = None) -> Pi06RtcRunnerWrapper:
    return Pi06RtcRunnerWrapper(model_dir=DEFAULT_PI06_RTC_MODEL_DIR if model_dir is None else Path(model_dir))


__all__ = [
    "ACTION_INDEX_MAPS",
    "DEFAULT_PI06_HETEROGENEOUS_MODEL_DIR",
    "DEFAULT_PI06_RTC_MODEL_DIR",
    "Pi06HeterogeneousRunnerWrapper",
    "Pi06HeterogeneousSpec",
    "Pi06RtcRunnerWrapper",
    "build_pi06_rtc_runner",
]
