#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <torch211|torch28> <attempt-name>" >&2
  exit 64
fi

lane=$1
attempt=$2

case "$lane" in
  torch211)
    expected_torch=2.11
    torch_overlay=
    ;;
  torch28)
    expected_torch=2.8
    torch_overlay=/home/zem/wujinlong/picpp_walloss_20260818/revision7_export/env/torch28_overlay
    ;;
  *)
    echo "unsupported lane: $lane" >&2
    exit 64
    ;;
esac

if [[ ! "$attempt" =~ ^attempt[0-9]+$ ]]; then
  echo "attempt must match attempt<number>: $attempt" >&2
  exit 64
fi

r7_root=/home/zem/wujinlong/picpp_walloss_20260818/revision7_export
r7_scripts="$r7_root/scripts"
r7_input="$r7_root/input"
python_bin=/home/zem/miniconda3/envs/zem-lens/bin/python
checkpoint="$r7_input/checkpoint"
v18_root="$r7_input/v18/onnx_real_legacy_v18"
legacy_fixture="$r7_root/reference/calibration/pytorch_tensors.pt"
placement=/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/results/placement_manifest.json
allowlist=/home/zem/wujinlong/picpp_walloss_20260818/revision6_mixed_ep/results/cpu_allowlist.json
output_root="$r7_root/diagnostic/same_host/$lane/$attempt"

[[ $(hostname) == zem-pc ]]
[[ ! -e "$output_root" ]]

gpu_apps=$(nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits)
if [[ -n "$gpu_apps" ]]; then
  echo "GPU has an existing compute owner:" >&2
  printf '%s\n' "$gpu_apps" >&2
  exit 20
fi

required_files=(
  "$r7_scripts/same_host_prefix_prefill_diagnostics.py"
  "$r7_scripts/agx_export_runner.py"
  "$r7_scripts/fixture_integrity.py"
  "$checkpoint/model.safetensors"
  "$checkpoint/config.yml"
  "$checkpoint/config.json"
  "$checkpoint/normalizer_action.pth"
  "$checkpoint/normalizer_propri.pth"
  "$v18_root/prefix_embed.onnx"
  "$v18_root/prefill.onnx"
  "$legacy_fixture"
  "$placement"
  "$allowlist"
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]]
done

printf '%s  %s\n' \
  44b91f69cd39a302114c39da1f1a3f1e3caad62005932603430ca4e8f6d094cd \
  "$r7_scripts/same_host_prefix_prefill_diagnostics.py" | sha256sum -c -
printf '%s  %s\n' \
  2c8a1d3f8feb32780b407c3497634d0ea7a5d020061314d8c376a55d393399bf \
  "$r7_scripts/fixture_integrity.py" | sha256sum -c -
printf '%s  %s\n' \
  8165beccc6d3c2704a067ba07a4874448fd18ab541c3568029eccb1e71fe79ab \
  "$r7_scripts/agx_export_runner.py" | sha256sum -c -
printf '%s  %s\n' \
  0305f89c90fac681026c187f71cb8487947bd7d220c62c32495b774f367e08a2 \
  "$placement" | sha256sum -c -
printf '%s  %s\n' \
  474fd05e98a9379e6be6a385cb77c0c8339645507ae3ea8bc4bfab413353e341 \
  "$allowlist" | sha256sum -c -

base_pythonpath="$r7_scripts:$r7_root/env/python_overlay_v2:$r7_root/env/ort_py:$r7_input/source/wall-x:$r7_input/source"
if [[ -n "$torch_overlay" ]]; then
  lane_pythonpath="$torch_overlay:$base_pythonpath"
else
  lane_pythonpath=$base_pythonpath
fi

mkdir -p "$output_root"

{
  echo "walloss Revision 7 same-host diagnostic"
  printf 'lane=%s\n' "$lane"
  printf 'attempt=%s\n' "$attempt"
  printf 'started_at=%s\n' "$(date -Is)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'python=%s\n' "$python_bin"
  printf 'expected_torch=%s\n' "$expected_torch"
  printf 'PYTHONPATH=%s\n' "$lane_pythonpath"
  printf 'TMPDIR=%s\n' "$r7_root/env/tmp"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits
  df -B1 /home/zem/wujinlong
  sha256sum \
    "$r7_scripts/run_same_host_lane.sh" \
    "$r7_scripts/same_host_prefix_prefill_diagnostics.py" \
    "$r7_scripts/agx_export_runner.py" \
    "$r7_scripts/fixture_integrity.py" \
    "$placement" \
    "$allowlist"
} >"$output_root/execution_preflight.log"

TMPDIR="$r7_root/env/tmp" \
PYTHONPATH="$lane_pythonpath" \
"$python_bin" - "$expected_torch" >>"$output_root/execution_preflight.log" <<'PY'
import sys

import onnxruntime as ort
import torch

expected = sys.argv[1]
assert torch.__version__.split("+")[0].startswith(expected + ".")
assert torch.cuda.is_available()
assert ort.__version__ == "1.23.2"
assert {"CUDAExecutionProvider", "CPUExecutionProvider"} <= set(ort.get_available_providers())
print(
    {
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "ort": ort.__version__,
        "ort_file": ort.__file__,
        "providers": ort.get_available_providers(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }
)
PY

set +e
PYTHONUNBUFFERED=1 \
TMPDIR="$r7_root/env/tmp" \
PYTHONPATH="$lane_pythonpath" \
"$python_bin" "$r7_scripts/same_host_prefix_prefill_diagnostics.py" \
  --checkpoint "$checkpoint" \
  --v18-root "$v18_root" \
  --legacy-fixture "$legacy_fixture" \
  --output-fixture "$output_root/recovered.pt" \
  --output-json "$output_root/same_host_prefix_prefill_diagnostics.json" \
  --profile-dir "$output_root/profiles" \
  --approved-placement-manifest "$placement" \
  --approved-cpu-allowlist "$allowlist" \
  --fixture-integrity-script "$r7_scripts/fixture_integrity.py" \
  --device cuda:0 \
  2>&1 | tee "$output_root/run.log"
rc=${PIPESTATUS[0]}
set -e

printf '%s\n' "$rc" >"$output_root/run.rc"
{
  printf 'finished_at=%s\n' "$(date -Is)"
  printf 'returncode=%s\n' "$rc"
  nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
} >"$output_root/execution_postflight.log"

exit "$rc"
