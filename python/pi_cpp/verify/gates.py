"""Acceptance gate defaults per model family, with per-package overrides.

Families with recorded board evidence ship defaults here; any other family
must declare gates explicitly in its manifest (fail-closed). Per-package
`gates` deep-merge over the family defaults, override wins.
"""

from __future__ import annotations

import json
from typing import Any

FAMILY_GATES: dict[str, dict[str, Any]] = {
    "pi06": {
        "parity": {"rel_l2": 0.02, "cosine": 0.9998, "max_abs_scaled": 0.2, "raw": 0.1, "worst_rel_l2": 0.095},
        "latency": {"protocol": "d10", "warmup": 10, "runs": 100},
    },
    "pi05": {
        "parity": {"rel_l2": 1e-5},
        "latency": {"protocol": "d10", "warmup": 10, "runs": 100},
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def gates_for(family: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    base = FAMILY_GATES.get(family)
    if base is None:
        if not overrides:
            raise ValueError(
                f"no gate defaults recorded for family {family!r}; the package must declare gates explicitly"
            )
        return overrides
    merged = json.loads(json.dumps(base))
    if overrides:
        _deep_merge(merged, overrides)
    return merged
