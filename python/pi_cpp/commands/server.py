"""WebSocket policy server command."""

from __future__ import annotations
from functools import partial
from pathlib import Path
from typing import Literal
from policy_websocket import WebsocketPolicyServer
import policy_websocket.websocket_server as websocket_server_module
from pi_cpp.server.policy import (
    Dit4DitPolicy,
    Evo1Policy,
    FastWamPolicy,
    GrootPolicy,
    Pi05Policy,
    Pi06AirbotPolicy,
    Pi06HeterogeneousPolicy,
    SemanticVlaPolicy,
    SmolVlaPolicy,
    StarVlaPolicy,
)

Model = Literal["pi05", "pi06_airbot", "pi06_heterogeneous", "fastwam", "semanticvla", "evo1", "smolvla", "dit4dit", "groot", "starvla"]
websocket_server_module._server.serve = partial(
    websocket_server_module._server.serve,
    ping_interval=None,
    ping_timeout=None,
)


def run_server(*, model: Model, model_dir: Path, host: str, port: int) -> int:
    if model == "pi05":
        policy = Pi05Policy(model_dir=model_dir)
    elif model == "pi06_airbot":
        policy = Pi06AirbotPolicy(model_dir=model_dir)
    elif model == "pi06_heterogeneous":
        policy = Pi06HeterogeneousPolicy(model_dir=model_dir)
    elif model == "fastwam":
        policy = FastWamPolicy(model_dir=model_dir)
    elif model == "semanticvla":
        policy = SemanticVlaPolicy(model_dir=model_dir)
    elif model == "evo1":
        policy = Evo1Policy(model_dir=model_dir)
    elif model == "smolvla":
        policy = SmolVlaPolicy(model_dir=model_dir)
    elif model == "dit4dit":
        policy = Dit4DitPolicy(model_dir=model_dir)
    elif model == "groot":
        policy = GrootPolicy(model_dir=model_dir)
    elif model == "starvla":
        policy = StarVlaPolicy(model_dir=model_dir)
    else:
        raise ValueError(f"unsupported server model: {model}")

    metadata = {
        "model": model,
        "model_dir": str(model_dir),
        "image_size": list(policy.image_size),
        "num_cameras": policy.num_cameras,
        "state_dim": policy.state_dim,
        "action_horizon": policy.action_horizon,
        "action_dim": policy.action_dim,
    }
    if model == "pi05":
        metadata["action_steps"] = policy.action_steps
    elif model == "pi06_airbot":
        metadata.update(
            {
                "camera_order": list(policy.camera_order),
                "request_keys": [*policy.camera_order, "state", "prompt", "advantage"],
                "image_layout": "HWC",
                "image_dtype": "uint8",
                "image_range": [0, 255],
                "state_layout": "left_arm_6,left_gripper,right_arm_6,right_gripper",
                "action_layout": "left_arm_6,left_gripper,right_arm_6,right_gripper",
                "internal_action_dim": policy.internal_action_dim,
                "token_length": policy.token_length,
                "action_semantics": policy.action_semantics,
                "advantage_conditioning": policy.advantage_conditioning,
                "default_advantage": policy.default_advantage,
                "denoise_steps": policy.denoise_steps,
                "dt": policy.dt,
                "noise_mode": "seeded_stream",
                "manifest_schema": policy.manifest_schema,
            }
        )

    server = WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=metadata,
    )
    server.serve_forever()
    return 0
