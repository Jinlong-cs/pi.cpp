#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


Shape = tuple[int, ...]
TensorSpec = tuple[torch.dtype, Shape]

ACTION_SHAPE = (1, 32, 26)
KV_SHAPE = (1, 2, 736, 128)

PREFIX_INPUT_SPECS: dict[str, TensorSpec] = {
    "input_ids": (torch.int64, (1, 768)),
    "attention_mask": (torch.bool, (1, 768)),
    "pixel_values": (torch.float32, (2048, 1176)),
    "image_grid_thw": (torch.int64, (2, 3)),
    "moe_token_types": (torch.bool, (1, 768)),
    "x0": (torch.float32, ACTION_SHAPE),
    "t0": (torch.float32, (1,)),
    "dof_mask": (torch.float32, ACTION_SHAPE),
}

PREFILL_INPUT_SPECS: dict[str, TensorSpec] = {
    "prefill_inputs_embeds": (torch.bfloat16, (1, 768, 2048)),
    "prefill_attention_mask": (torch.bool, (1, 768)),
    "prefill_position_ids": (torch.int64, (3, 1, 768)),
    "prefill_moe_token_types": (torch.bool, (1, 768)),
    "prefill_action_mask": (torch.bool, (1, 768)),
    "prefill_start_indices": (torch.int64, (2,)),
    "prefill_end_indices": (torch.int64, (2,)),
    "x0": (torch.float32, ACTION_SHAPE),
    "dt0": (torch.float32, (1,)),
    "dof_mask": (torch.float32, ACTION_SHAPE),
    "v_padding": (torch.float32, ACTION_SHAPE),
}

PREFIX_POSTFIX_METADATA_SPECS: dict[str, TensorSpec] = {
    "postfix_position_ids": (torch.int64, (3, 1, 32)),
    "postfix_attention_mask": (torch.bool, (1, 32, 768)),
}

PREFIX_OUTPUT_SPECS = {
    **PREFILL_INPUT_SPECS,
    **PREFIX_POSTFIX_METADATA_SPECS,
}

KV_SPECS: dict[str, TensorSpec] = {
    f"prefix_kv.layer_{layer:02d}.{kind}": (torch.bfloat16, KV_SHAPE)
    for layer in range(36)
    for kind in ("key", "value")
}

PREFILL_OUTPUT_SPECS: dict[str, TensorSpec] = {
    "x1": (torch.float32, ACTION_SHAPE),
    **KV_SPECS,
}

POSTFIX_CONTROL_SPECS: dict[str, TensorSpec] = {
    "x_i": (torch.float32, ACTION_SHAPE),
    "t_i": (torch.float32, (1,)),
    "dt_i": (torch.float32, (1,)),
    "dof_mask": (torch.float32, ACTION_SHAPE),
    "v_padding": (torch.float32, ACTION_SHAPE),
    **PREFIX_POSTFIX_METADATA_SPECS,
}

POSTFIX_INPUT_SPECS = {
    **POSTFIX_CONTROL_SPECS,
    **KV_SPECS,
}

ACTION_SPECS: dict[str, TensorSpec] = {
    "internal_action": (torch.float32, ACTION_SHAPE),
    "external_action": (torch.float32, ACTION_SHAPE),
    "wrapper_action": (torch.float32, ACTION_SHAPE),
}

REQUIRED_SECTIONS = {
    "prefix_inputs",
    "prefix_outputs",
    "prefill_inputs",
    "prefill_outputs",
    "postfix_inputs",
    "postfix_output",
    *ACTION_SPECS,
}

TRAJECTORY_NAMES = tuple(f"x{index}" for index in range(1, 11))
POSTFIX_STEP_NAMES = tuple(f"step{index}" for index in range(1, 10))
POSTFIX_INVARIANT_NAMES = (
    "dof_mask",
    "v_padding",
    "postfix_position_ids",
    "postfix_attention_mask",
)
PAIR_IDENTITY_INPUTS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "moe_token_types",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def describe(value: torch.Tensor) -> dict[str, Any]:
    floating = value.dtype.is_floating_point
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()) if floating else True,
        "nonzero": int(torch.count_nonzero(value)),
        "sha256": tensor_sha256(value),
    }


def tensor_mapping(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        return {}
    return {
        name: tensor
        for name, tensor in value.items()
        if isinstance(name, str) and torch.is_tensor(tensor)
    }


def validate_mapping(value: object, specs: dict[str, TensorSpec]) -> dict[str, Any]:
    is_mapping = isinstance(value, dict)
    raw_names = set(value) if is_mapping else set()
    tensors = tensor_mapping(value)
    expected_names = set(specs)
    missing = sorted(expected_names - raw_names)
    unexpected = sorted(raw_names - expected_names, key=str)
    non_tensor = sorted(
        name for name in expected_names & raw_names if name not in tensors
    )
    rows: dict[str, Any] = {}
    for name in sorted(expected_names & set(tensors)):
        expected_dtype, expected_shape = specs[name]
        observed = describe(tensors[name])
        observed["expected_dtype"] = str(expected_dtype)
        observed["expected_shape"] = list(expected_shape)
        observed["dtype_match"] = tensors[name].dtype == expected_dtype
        observed["shape_match"] = tuple(tensors[name].shape) == expected_shape
        observed["passed"] = bool(
            observed["dtype_match"] and observed["shape_match"] and observed["finite"]
        )
        rows[name] = observed
    return {
        "is_mapping": is_mapping,
        "missing": missing,
        "unexpected": unexpected,
        "non_tensor": non_tensor,
        "tensors": rows,
        "passed": bool(
            is_mapping
            and not missing
            and not unexpected
            and not non_tensor
            and len(rows) == len(specs)
            and all(row["passed"] for row in rows.values())
        ),
    }


def validate_tensor(value: object, dtype: torch.dtype, shape: Shape) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {
            "is_tensor": False,
            "expected_dtype": str(dtype),
            "expected_shape": list(shape),
            "passed": False,
        }
    row = describe(value)
    row.update(
        {
            "is_tensor": True,
            "expected_dtype": str(dtype),
            "expected_shape": list(shape),
            "dtype_match": value.dtype == dtype,
            "shape_match": tuple(value.shape) == shape,
        }
    )
    row["passed"] = bool(row["dtype_match"] and row["shape_match"] and row["finite"])
    return row


def compare_named(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    names: set[str] | tuple[str, ...],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    missing_from_left = sorted(set(names) - set(left))
    missing_from_right = sorted(set(names) - set(right))
    for name in sorted(set(names) & set(left) & set(right)):
        left_hash = tensor_sha256(left[name])
        right_hash = tensor_sha256(right[name])
        rows[name] = {
            "dtype_equal": left[name].dtype == right[name].dtype,
            "shape_equal": tuple(left[name].shape) == tuple(right[name].shape),
            "bit_exact": left_hash == right_hash,
            "left_sha256": left_hash,
            "right_sha256": right_hash,
        }
    return {
        "missing_from_left": missing_from_left,
        "missing_from_right": missing_from_right,
        "tensors": rows,
        "passed": bool(
            not missing_from_left
            and not missing_from_right
            and len(rows) == len(names)
            and all(
                row["dtype_equal"] and row["shape_equal"] and row["bit_exact"]
                for row in rows.values()
            )
        ),
    }


def compare_float32_exact(
    expected: dict[str, torch.Tensor],
    observed: dict[str, torch.Tensor],
    names: set[str],
) -> dict[str, Any]:
    report = compare_named(expected, observed, names)
    for name, row in report["tensors"].items():
        left = expected[name]
        right = observed[name]
        row["float32_exact"] = bool(
            left.dtype == torch.float32
            and right.dtype == torch.float32
            and tuple(left.shape) == tuple(right.shape)
            and torch.isfinite(left).all()
            and torch.isfinite(right).all()
            and torch.equal(left, right)
        )
        row["passed"] = bool(
            row["dtype_equal"]
            and row["shape_equal"]
            and (row["bit_exact"] or row["float32_exact"])
        )
    report["passed"] = bool(
        not report["missing_from_left"]
        and not report["missing_from_right"]
        and len(report["tensors"]) == len(names)
        and all(row["passed"] for row in report["tensors"].values())
    )
    return report


def trajectory_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"x([1-9]|10)", name)
    return (int(match.group(1)), name) if match else (11, name)


def mapping_sha256(
    values: dict[str, torch.Tensor], names: tuple[str, ...] | None = None
) -> str:
    selected = tuple(sorted(values)) if names is None else names
    digest = hashlib.sha256()
    for name in selected:
        if name not in values:
            raise KeyError(name)
        value = values[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


def bfloat16_roundtrip(value: torch.Tensor) -> dict[str, Any]:
    source = value.detach().cpu().contiguous()
    bits = source.view(torch.uint16).clone()
    restored = bits.view(torch.bfloat16).reshape(source.shape)
    source_hash = tensor_sha256(source)
    restored_hash = tensor_sha256(restored)
    return {
        "shape_equal": tuple(source.shape) == tuple(restored.shape),
        "dtype_equal": restored.dtype == torch.bfloat16,
        "bit_exact": source_hash == restored_hash,
        "source_sha256": source_hash,
        "roundtrip_sha256": restored_hash,
        "passed": bool(
            tuple(source.shape) == tuple(restored.shape)
            and restored.dtype == torch.bfloat16
            and source_hash == restored_hash
        ),
    }


def validate_bfloat16_roundtrips(fixture: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, torch.Tensor] = {}
    for section_name, specs in (
        ("prefix_outputs", PREFIX_OUTPUT_SPECS),
        ("prefill_inputs", PREFILL_INPUT_SPECS),
        ("prefill_outputs", PREFILL_OUTPUT_SPECS),
        ("postfix_inputs", POSTFIX_INPUT_SPECS),
    ):
        section = tensor_mapping(fixture.get(section_name))
        for name, (dtype, _) in specs.items():
            if dtype == torch.bfloat16 and name in section:
                expected[f"{section_name}.{name}"] = section[name]
    rows = {name: bfloat16_roundtrip(value) for name, value in sorted(expected.items())}
    expected_count = sum(
        dtype == torch.bfloat16
        for specs in (
            PREFIX_OUTPUT_SPECS,
            PREFILL_INPUT_SPECS,
            PREFILL_OUTPUT_SPECS,
            POSTFIX_INPUT_SPECS,
        )
        for dtype, _ in specs.values()
    )
    return {
        "expected_tensor_count": expected_count,
        "observed_tensor_count": len(rows),
        "tensors": rows,
        "passed": bool(
            len(rows) == expected_count and all(row["passed"] for row in rows.values())
        ),
    }


def validate_postfix_inputs_by_step(
    value: object,
    trajectory: dict[str, torch.Tensor],
    prefill_outputs: dict[str, torch.Tensor],
    legacy_postfix_inputs: dict[str, torch.Tensor],
    required: bool,
) -> dict[str, Any]:
    frozen_times = torch.linspace(0.0, 1.0, 11, dtype=torch.float32) * torch.tensor(
        0.999, dtype=torch.float32
    )
    frozen_deltas = frozen_times[1:] - frozen_times[:-1]
    schedule = {
        "definition": "float32 linspace(0,1,11) * 0.999",
        "times": frozen_times.tolist(),
        "deltas": frozen_deltas.tolist(),
        "times_sha256": tensor_sha256(frozen_times),
        "deltas_sha256": tensor_sha256(frozen_deltas),
    }
    present = value is not None
    if not present and not required:
        return {
            "required": False,
            "present": False,
            "is_mapping": False,
            "missing_steps": [],
            "unexpected_steps": [],
            "frozen_d10": schedule,
            "steps": {},
            "invariant_controls": {},
            "legacy_final_only_fixture": False,
            "failure_reason": None,
            "legacy_postfix_inputs_matches_step9": {
                "checked": False,
                "passed": True,
            },
            "passed": True,
        }

    is_mapping = isinstance(value, dict)
    raw_names = set(value) if is_mapping else set()
    expected_names = set(POSTFIX_STEP_NAMES)
    missing_steps = sorted(expected_names - raw_names)
    unexpected_steps = sorted(raw_names - expected_names, key=str)
    step_inputs: dict[str, dict[str, torch.Tensor]] = {}
    steps: dict[str, Any] = {}
    for index, step_name in enumerate(POSTFIX_STEP_NAMES, start=1):
        step_value = value.get(step_name) if is_mapping else None
        inputs = tensor_mapping(step_value)
        step_inputs[step_name] = inputs
        abi = validate_mapping(step_value, POSTFIX_INPUT_SPECS)
        trajectory_name = f"x{index}"
        x_reference = (
            {"x_i": trajectory[trajectory_name]}
            if trajectory_name in trajectory
            else {}
        )
        x_link = compare_named(x_reference, inputs, {"x_i"})
        kv_bridge = compare_named(prefill_outputs, inputs, set(KV_SPECS))
        expected_time_inputs = {
            "t_i": frozen_times[index : index + 1],
            "dt_i": frozen_deltas[index : index + 1],
        }
        time_exact = compare_float32_exact(
            expected_time_inputs, inputs, {"t_i", "dt_i"}
        )
        steps[step_name] = {
            "trajectory_input": trajectory_name,
            "abi": abi,
            "x_i_matches_trajectory": x_link,
            "kv_matches_prefill_outputs": kv_bridge,
            "time_source": "frozen_expected_to_fixture",
            "time_float32_exact": time_exact,
            "passed": bool(
                abi["passed"]
                and x_link["passed"]
                and kv_bridge["passed"]
                and time_exact["passed"]
            ),
        }

    reference_inputs = step_inputs.get("step1", {})
    invariant_controls: dict[str, Any] = {}
    for step_name in POSTFIX_STEP_NAMES:
        comparison = compare_named(
            reference_inputs,
            step_inputs.get(step_name, {}),
            set(POSTFIX_INVARIANT_NAMES),
        )
        invariant_controls[step_name] = {
            "reference_step": "step1",
            **comparison,
        }

    step9_inputs = step_inputs.get("step9", {})
    legacy_step9 = compare_named(
        legacy_postfix_inputs,
        step9_inputs,
        set(POSTFIX_INPUT_SPECS),
    )
    legacy_step9["checked"] = True
    passed = bool(
        is_mapping
        and not missing_steps
        and not unexpected_steps
        and all(row["passed"] for row in steps.values())
        and all(row["passed"] for row in invariant_controls.values())
        and legacy_step9["passed"]
    )
    return {
        "required": required,
        "present": present,
        "is_mapping": is_mapping,
        "missing_steps": missing_steps,
        "unexpected_steps": unexpected_steps,
        "frozen_d10": schedule,
        "steps": steps,
        "invariant_control_names": list(POSTFIX_INVARIANT_NAMES),
        "invariant_controls": invariant_controls,
        "legacy_final_only_fixture": bool(
            required and not present and legacy_postfix_inputs
        ),
        "failure_reason": (
            "legacy fixture has only final postfix_inputs; postfix_inputs_by_step is required"
            if required and not present and legacy_postfix_inputs
            else (
                "postfix_inputs_by_step must be a mapping"
                if present and not is_mapping
                else None
            )
        ),
        "legacy_postfix_inputs_matches_step9": legacy_step9,
        "passed": passed,
    }


def validate_pair(
    fixture_path: Path,
    fixture: dict[str, Any],
    paired_path: Path | None,
) -> dict[str, Any]:
    if paired_path is None:
        return {"requested": False, "passed": True}
    paired = torch.load(paired_path, map_location="cpu", weights_only=True)
    if not isinstance(paired, dict):
        return {
            "requested": True,
            "path": str(paired_path),
            "sha256": sha256_file(paired_path),
            "error": "paired fixture root is not a dictionary",
            "passed": False,
        }
    current_inputs = tensor_mapping(fixture.get("prefix_inputs"))
    paired_inputs = tensor_mapping(paired.get("prefix_inputs"))
    paired_abi = validate_mapping(paired.get("prefix_inputs"), PREFIX_INPUT_SPECS)
    can_hash = bool(
        set(PREFIX_INPUT_SPECS) <= set(current_inputs)
        and set(PREFIX_INPUT_SPECS) <= set(paired_inputs)
    )
    current_full_hash = mapping_sha256(current_inputs) if can_hash else None
    paired_full_hash = mapping_sha256(paired_inputs) if can_hash else None
    current_identity_hash = (
        mapping_sha256(current_inputs, PAIR_IDENTITY_INPUTS) if can_hash else None
    )
    paired_identity_hash = (
        mapping_sha256(paired_inputs, PAIR_IDENTITY_INPUTS) if can_hash else None
    )
    identity_comparison = compare_named(
        current_inputs, paired_inputs, set(PAIR_IDENTITY_INPUTS)
    )
    identity_comparison["all_bit_exact"] = identity_comparison.pop("passed")
    fixture_hash = sha256_file(fixture_path)
    paired_fixture_hash = sha256_file(paired_path)
    return {
        "requested": True,
        "path": str(paired_path),
        "sha256": paired_fixture_hash,
        "paired_prefix_inputs_abi": paired_abi,
        "fixture_files_differ": fixture_hash != paired_fixture_hash,
        "prefix_inputs_sha256": {
            "fixture": current_full_hash,
            "paired_fixture": paired_full_hash,
            "different": bool(
                current_full_hash is not None and current_full_hash != paired_full_hash
            ),
        },
        "sample_identity_sha256": {
            "fields": list(PAIR_IDENTITY_INPUTS),
            "fixture": current_identity_hash,
            "paired_fixture": paired_identity_hash,
            "different": bool(
                current_identity_hash is not None
                and current_identity_hash != paired_identity_hash
            ),
        },
        "sample_identity_tensor_comparison": identity_comparison,
        "passed": bool(
            paired_abi["passed"]
            and fixture_hash != paired_fixture_hash
            and current_full_hash is not None
            and current_full_hash != paired_full_hash
            and current_identity_hash is not None
            and current_identity_hash != paired_identity_hash
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--paired-fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-trajectory", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if not isinstance(fixture, dict):
        raise TypeError("fixture root must be a dictionary")

    required_sections = set(REQUIRED_SECTIONS)
    if args.require_trajectory:
        required_sections.update({"trajectory", "postfix_inputs_by_step"})
    missing_sections = sorted(required_sections - set(fixture))
    section_validation = {
        "prefix_inputs": validate_mapping(
            fixture.get("prefix_inputs"), PREFIX_INPUT_SPECS
        ),
        "prefix_outputs": validate_mapping(
            fixture.get("prefix_outputs"), PREFIX_OUTPUT_SPECS
        ),
        "prefill_inputs": validate_mapping(
            fixture.get("prefill_inputs"), PREFILL_INPUT_SPECS
        ),
        "prefill_outputs": validate_mapping(
            fixture.get("prefill_outputs"), PREFILL_OUTPUT_SPECS
        ),
        "postfix_inputs": validate_mapping(
            fixture.get("postfix_inputs"), POSTFIX_INPUT_SPECS
        ),
    }
    postfix_output = validate_tensor(
        fixture.get("postfix_output"), torch.float32, ACTION_SHAPE
    )
    actions = {
        name: validate_tensor(fixture.get(name), dtype, shape)
        for name, (dtype, shape) in ACTION_SPECS.items()
    }

    prefix_outputs = tensor_mapping(fixture.get("prefix_outputs"))
    prefill_inputs = tensor_mapping(fixture.get("prefill_inputs"))
    prefill_outputs = tensor_mapping(fixture.get("prefill_outputs"))
    postfix_inputs = tensor_mapping(fixture.get("postfix_inputs"))

    prefix_shared = compare_named(
        prefix_outputs, prefill_inputs, set(PREFILL_INPUT_SPECS)
    )
    prefill_not_from_prefix = sorted(set(prefill_inputs) - set(prefix_outputs))
    prefix_extra_keys = sorted(set(prefix_outputs) - set(prefill_inputs))
    expected_prefix_extras = sorted(PREFIX_POSTFIX_METADATA_SPECS)
    prefix_bridge = {
        "prefill_inputs_are_prefix_outputs_subset": not prefill_not_from_prefix,
        "prefill_inputs_missing_from_prefix_outputs": prefill_not_from_prefix,
        "prefix_output_extra_keys": prefix_extra_keys,
        "expected_postfix_metadata_keys": expected_prefix_extras,
        "prefix_extras_are_legal_postfix_metadata": prefix_extra_keys
        == expected_prefix_extras,
        "shared_tensors": prefix_shared,
    }
    prefix_bridge["passed"] = bool(
        not prefill_not_from_prefix
        and prefix_extra_keys == expected_prefix_extras
        and prefix_shared["passed"]
    )

    kv_bridge = compare_named(prefill_outputs, postfix_inputs, set(KV_SPECS))
    prefill_kv_names = sorted(set(prefill_outputs) & set(KV_SPECS))
    postfix_kv_names = sorted(set(postfix_inputs) & set(KV_SPECS))
    kv_layers = {
        int(match.group(1))
        for name in prefill_kv_names
        if (match := re.fullmatch(r"prefix_kv\.layer_(\d{2})\.(?:key|value)", name))
    }
    kv_nonzero = {
        name: int(torch.count_nonzero(prefill_outputs[name]))
        for name in prefill_kv_names
    }
    kv_gate = {
        "prefill_tensor_count": len(prefill_kv_names),
        "postfix_tensor_count": len(postfix_kv_names),
        "layer_count": len(kv_layers),
        "layers": sorted(kv_layers),
        "all_prefill_tensors_nonzero": bool(
            len(kv_nonzero) == 72 and all(count > 0 for count in kv_nonzero.values())
        ),
        "prefill_nonzero_counts": kv_nonzero,
        "bridge": kv_bridge,
    }
    kv_gate["passed"] = bool(
        len(prefill_kv_names) == 72
        and len(postfix_kv_names) == 72
        and kv_layers == set(range(36))
        and kv_gate["all_prefill_tensors_nonzero"]
        and kv_bridge["passed"]
    )

    trajectory_value = fixture.get("trajectory")
    trajectory = tensor_mapping(trajectory_value)
    trajectory_names = sorted(trajectory, key=trajectory_sort_key)
    trajectory_mapping = validate_mapping(
        trajectory_value,
        {name: (torch.float32, ACTION_SHAPE) for name in TRAJECTORY_NAMES},
    )
    trajectory_required_or_present = (
        args.require_trajectory or trajectory_value is not None
    )
    trajectory_links = {
        "prefill_x1": compare_named(prefill_outputs, trajectory, {"x1"}),
        "postfix_input_x9": compare_named(
            {"x9": postfix_inputs["x_i"]} if "x_i" in postfix_inputs else {},
            trajectory,
            {"x9"},
        ),
        "postfix_output_x10": compare_named(
            (
                {"x10": fixture["postfix_output"]}
                if torch.is_tensor(fixture.get("postfix_output"))
                else {}
            ),
            trajectory,
            {"x10"},
        ),
    }
    trajectory_gate = {
        "required": args.require_trajectory,
        "names": trajectory_names,
        "expected_names": list(TRAJECTORY_NAMES),
        "mapping": trajectory_mapping,
        "links": trajectory_links,
        "passed": bool(
            not trajectory_required_or_present
            or (
                trajectory_mapping["passed"]
                and all(link["passed"] for link in trajectory_links.values())
            )
        ),
    }

    postfix_inputs_by_step = validate_postfix_inputs_by_step(
        fixture.get("postfix_inputs_by_step"),
        trajectory,
        prefill_outputs,
        postfix_inputs,
        args.require_trajectory,
    )
    bf16_roundtrip = validate_bfloat16_roundtrips(fixture)
    paired_fixture = validate_pair(args.fixture, fixture, args.paired_fixture)
    inventory = {
        section: {
            name: describe(value) for name, value in tensor_mapping(values).items()
        }
        for section, values in fixture.items()
        if isinstance(values, dict)
    }
    actions_passed = bool(
        len(actions) == len(ACTION_SPECS)
        and all(row["passed"] for row in actions.values())
    )
    passed = bool(
        not missing_sections
        and all(row["passed"] for row in section_validation.values())
        and postfix_output["passed"]
        and actions_passed
        and prefix_bridge["passed"]
        and kv_gate["passed"]
        and trajectory_gate["passed"]
        and postfix_inputs_by_step["passed"]
        and bf16_roundtrip["passed"]
        and paired_fixture["passed"]
    )
    report = {
        "schema_version": "walloss.revision7.fixture-integrity.v3",
        "fixture": {"path": str(args.fixture), "sha256": sha256_file(args.fixture)},
        "paired_fixture": paired_fixture,
        "missing_sections": missing_sections,
        "fixed_abi": {
            "section_validation": section_validation,
            "postfix_output": postfix_output,
            "actions": actions,
            "actions_passed": actions_passed,
        },
        "inventory": inventory,
        "prefix_outputs_to_prefill_inputs": prefix_bridge,
        "prefill_outputs_to_postfix_inputs_kv_only": kv_bridge,
        "kv": kv_gate,
        "trajectory": trajectory_gate,
        "postfix_inputs_by_step": postfix_inputs_by_step,
        "bfloat16_bit_roundtrip": bf16_roundtrip,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "kv_tensor_count": kv_gate["prefill_tensor_count"],
                "kv_layer_count": kv_gate["layer_count"],
                "trajectory_complete": trajectory_mapping["passed"],
                "postfix_steps_complete": postfix_inputs_by_step["passed"],
                "paired_fixture_distinct": paired_fixture["passed"],
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
