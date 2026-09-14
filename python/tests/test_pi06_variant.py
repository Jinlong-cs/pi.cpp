from __future__ import annotations

import json

import numpy as np
import pytest
from pi_cpp.pi06 import build_pi06_runner, detect_pi06_variant, detect_variant
from pi_cpp.pi06_heterogeneous import (
    _detect_rtc,
    _map_action_prefix,
    _QuantileStats,
    prepare_rtc_inputs,
)


def test_detect_pi06_variant():
    assert detect_pi06_variant({"schema": "pi_cpp.pi06_airbot.v1"}) == "airbot"
    assert detect_pi06_variant({"schema": "pi_cpp.pi06_heterogeneous.v1"}) == "heterogeneous"
    assert (
        detect_pi06_variant({"schema": "pi_cpp.pi06_heterogeneous.v1", "rtc": {"training_time_rtc": True}})
        == "rtc"
    )


def test_detect_pi06_variant_fail_closed():
    with pytest.raises(ValueError, match="training_time_rtc"):
        detect_pi06_variant({"schema": "pi_cpp.pi06_heterogeneous.v1", "rtc": {"training_time_rtc": False}})
    with pytest.raises(ValueError, match="unrecognized"):
        detect_pi06_variant({"schema": "something.else.v1"})


def test_detect_rtc_flag():
    assert _detect_rtc({}) is False
    assert _detect_rtc({"rtc": {"training_time_rtc": True}}) is True
    with pytest.raises(ValueError, match="training_time_rtc"):
        _detect_rtc({"rtc": {"training_time_rtc": False}})


def test_detect_variant_from_deployment_manifest(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(json.dumps({"model": {"variant": "rtc"}}))
    assert detect_variant(model_dir) == "rtc"


def test_detect_variant_deployment_contradiction(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "deployment_manifest.json").write_text(
        json.dumps({"model": {"variant": "airbot"}, "runtime": {"rtc": {"enabled": True}}})
    )
    with pytest.raises(ValueError, match="contradicts"):
        detect_variant(model_dir)


def test_detect_variant_from_export_manifest(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "export_manifest.json").write_text(
        json.dumps({"schema": "pi_cpp.pi06_heterogeneous.v1", "rtc": {"training_time_rtc": True}})
    )
    assert detect_variant(model_dir) == "rtc"


def test_detect_variant_missing_manifests(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    with pytest.raises(ValueError, match="neither"):
        detect_variant(model_dir)


def test_map_action_prefix_embodiment_zero():
    mapped = _map_action_prefix(np.arange(16, dtype=np.float32).reshape(1, 16), 0, 16)
    assert mapped.shape == (1, 16)
    assert mapped[0, 0] == 0 and mapped[0, 5] == 5 and mapped[0, 7] == 6 and mapped[0, 15] == 13
    assert mapped[0, 6] == 0 and mapped[0, 14] == 0


def test_map_action_prefix_embodiment_one_identity():
    mapped = _map_action_prefix(np.arange(16, dtype=np.float32).reshape(1, 16), 1, 16)
    assert np.array_equal(mapped, np.arange(16, dtype=np.float32).reshape(1, 16))


def _stats(q01: np.ndarray, q99: np.ndarray) -> _QuantileStats:
    return _QuantileStats(q01=np.asarray(q01, dtype=np.float32), q99=np.asarray(q99, dtype=np.float32))


def test_prepare_rtc_inputs_defaults():
    stats = _stats(np.zeros(16), np.ones(16))
    delay, prefix = prepare_rtc_inputs(
        None, None, 1, horizon=50, raw_action_dim=16, internal_action_dim=16, norm_stats={"actions": stats}, rtc_max_delay=4
    )
    assert delay.tolist() == [0]
    assert prefix.shape == (1, 50, 16)
    assert not prefix.any()


def test_prepare_rtc_inputs_normalization():
    stats = _stats(np.zeros(16), np.ones(16))
    delay, prefix = prepare_rtc_inputs(
        2,
        np.ones((2, 16), dtype=np.float32) * 0.25,
        1,
        horizon=50,
        raw_action_dim=16,
        internal_action_dim=16,
        norm_stats={"actions": stats},
        rtc_max_delay=4,
    )
    assert delay.tolist() == [2]
    assert prefix.shape == (1, 50, 16)
    # (0.25 - 0) / (1 - 0) * 2 - 1 = -0.5
    assert np.allclose(prefix[0, :2], -0.5)
    assert not prefix[0, 2:].any()


def test_prepare_rtc_inputs_fail_fast():
    stats = _stats(np.zeros(16), np.ones(16))
    kwargs = {"horizon": 50, "raw_action_dim": 16, "internal_action_dim": 16, "norm_stats": {"actions": stats}, "rtc_max_delay": 4}
    with pytest.raises(ValueError, match="outside the trained range"):
        prepare_rtc_inputs(5, None, 1, **kwargs)
    with pytest.raises(ValueError, match="requires action_prefix"):
        prepare_rtc_inputs(2, None, 1, **kwargs)
    with pytest.raises(ValueError, match="must have shape"):
        prepare_rtc_inputs(2, np.ones((3, 16)), 1, **kwargs)


def test_build_pi06_runner_routes_to_variant_class(tmp_path):
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "export_manifest.json").write_text(json.dumps({"schema": "pi_cpp.pi06_airbot.v1"}))
    # The override routes to the heterogeneous wrapper, whose spec loader
    # fails fast on the airbot-shaped manifest (no "runtime" section).
    with pytest.raises(KeyError, match="runtime"):
        build_pi06_runner(model_dir, variant="heterogeneous")
