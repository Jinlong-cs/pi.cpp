"""Small JSON contracts shared by optimization commands and evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str
    stages: dict[str, str]
    contract: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: object) -> ModelSpec:
        raw = _require_object(value, "model_spec")
        stages = _require_object(raw.get("stages"), "model_spec.stages")
        if any(
            not isinstance(key, str) or not isinstance(path, str) for key, path in stages.items()
        ):
            raise TypeError("model_spec.stages must map strings to strings")
        return cls(
            name=_require_string(raw.get("name"), "model_spec.name"),
            revision=_require_string(raw.get("revision"), "model_spec.revision"),
            stages=dict(stages),
            contract=dict(_require_object(raw.get("contract", {}), "model_spec.contract")),
            schema_version=int(raw.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class TargetFingerprint:
    name: str
    architecture: str
    hardware: dict[str, Any]
    software: dict[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> TargetFingerprint:
        raw = _require_object(value, "target_fingerprint")
        return cls(
            name=_require_string(raw.get("name"), "target_fingerprint.name"),
            architecture=_require_string(
                raw.get("architecture"), "target_fingerprint.architecture"
            ),
            hardware=dict(_require_object(raw.get("hardware", {}), "target_fingerprint.hardware")),
            software=dict(_require_object(raw.get("software", {}), "target_fingerprint.software")),
            schema_version=int(raw.get("schema_version", 1)),
        )


@dataclass(frozen=True)
class OptimizationPlan:
    name: str
    status: str
    model: str
    target: str
    stages: dict[str, dict[str, Any]]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: object) -> OptimizationPlan:
        raw = _require_object(value, "optimization_plan")
        stages = _require_object(raw.get("stages"), "optimization_plan.stages")
        if any(
            not isinstance(key, str) or not isinstance(stage, dict) for key, stage in stages.items()
        ):
            raise TypeError("optimization_plan.stages must map stage names to JSON objects")
        status = _require_string(raw.get("status"), "optimization_plan.status")
        if status not in {"proposed", "measured", "accepted", "rejected"}:
            raise ValueError(f"unsupported optimization status: {status}")
        return cls(
            name=_require_string(raw.get("name"), "optimization_plan.name"),
            status=status,
            model=_require_string(raw.get("model"), "optimization_plan.model"),
            target=_require_string(raw.get("target"), "optimization_plan.target"),
            stages={key: dict(stage) for key, stage in stages.items()},
            schema_version=int(raw.get("schema_version", 1)),
        )

    def stage(self, name: str) -> dict[str, Any]:
        if name not in self.stages:
            raise KeyError(f"optimization plan {self.name!r} has no stage {name!r}")
        return dict(self.stages[name])


@dataclass(frozen=True)
class EvidenceRecord:
    kind: str
    status: str
    command: list[str]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    target: TargetFingerprint | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_bundle(
    path: Path,
) -> tuple[ModelSpec, TargetFingerprint, OptimizationPlan, list[EvidenceRecord]]:
    raw = _require_object(json.loads(path.read_text()), "recipe")
    model = ModelSpec.from_dict(raw.get("model_spec"))
    target = TargetFingerprint.from_dict(raw.get("target_fingerprint"))
    plan = OptimizationPlan.from_dict(raw.get("optimization_plan"))
    evidence: list[EvidenceRecord] = []
    for index, item in enumerate(raw.get("evidence", [])):
        record = _require_object(item, f"evidence[{index}]")
        evidence.append(
            EvidenceRecord(
                kind=_require_string(record.get("kind"), f"evidence[{index}].kind"),
                status=_require_string(record.get("status"), f"evidence[{index}].status"),
                command=list(record.get("command", [])),
                inputs=dict(_require_object(record.get("inputs", {}), f"evidence[{index}].inputs")),
                outputs=dict(
                    _require_object(record.get("outputs", {}), f"evidence[{index}].outputs")
                ),
                metrics=dict(
                    _require_object(record.get("metrics", {}), f"evidence[{index}].metrics")
                ),
                target=target,
                schema_version=int(record.get("schema_version", 1)),
            )
        )
    return model, target, plan, evidence


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
