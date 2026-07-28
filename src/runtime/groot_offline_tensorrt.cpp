#include "pi_cpp/runtime/groot_offline.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/groot_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

float HalfBitsToFloat(std::uint16_t half) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(half & 0x8000U)) << 16;
  std::uint32_t exponent = (half >> 10) & 0x1FU;
  std::uint32_t mantissa = half & 0x03FFU;

  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Status CopyDeviceTensorToFloatVector(const runtime::DeviceTensor& tensor, cudaStream_t stream, std::vector<float>* values) {
  const auto elements = static_cast<std::size_t>(tensor.spec.shape.NumElements());
  values->assign(elements, 0.0F);
  if (tensor.spec.dtype == DType::kFloat32) {
    RETURN_IF_ERROR(runtime::CopyDeviceToHost(tensor, values->data(), tensor.spec.NumBytes(), stream));
    return runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed for " + tensor.spec.name);
  }
  if (tensor.spec.dtype == DType::kFloat16) {
    std::vector<std::uint16_t> half_values(elements);
    RETURN_IF_ERROR(runtime::CopyDeviceToHost(tensor, half_values.data(), tensor.spec.NumBytes(), stream));
    RETURN_IF_ERROR(
        runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed for " + tensor.spec.name));
    for (std::size_t index = 0; index < elements; ++index) (*values)[index] = HalfBitsToFloat(half_values[index]);
    return Status::Ok();
  }
  return Status::InvalidArgument("unsupported output dtype for float decode: " + tensor.spec.name);
}

}  // namespace

Status GrootStageTimers::Init() {
  RETURN_IF_ERROR(backbone.Init(std::string(groot::kBackboneStage)));
  return action_loop.Init(std::string(groot::kActionLoopStage));
}

GrootOfflineRunner::GrootOfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      backbone_(engine_dir_ / std::string(groot::kBackboneEngine)),
      action_loop_(engine_dir_ / std::string(groot::kActionLoopEngine)) {}

GrootOfflineRunner::~GrootOfflineRunner() = default;

Status GrootOfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&backbone_));
  RETURN_IF_ERROR(runtime::LoadEngine(&action_loop_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(backbone_, &backbone_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(action_loop_, &action_workspace_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* input_ids_spec = FindTensorSpec(backbone_.inputs(), groot::kInputIds);
  const auto* attention_mask_spec = FindTensorSpec(backbone_.inputs(), groot::kAttentionMask);
  const auto* pixel_values_0_spec = FindTensorSpec(backbone_.inputs(), groot::kPixelValues0);
  const auto* pixel_values_1_spec = FindTensorSpec(backbone_.inputs(), groot::kPixelValues1);
  const auto* state_spec = FindTensorSpec(action_loop_.inputs(), groot::kState);
  const auto* embodiment_id_spec = FindTensorSpec(action_loop_.inputs(), groot::kEmbodimentId);
  const auto* initial_actions_spec = FindTensorSpec(action_loop_.inputs(), groot::kInitialActions);
  if (input_ids_spec == nullptr || attention_mask_spec == nullptr || pixel_values_0_spec == nullptr ||
      pixel_values_1_spec == nullptr || state_spec == nullptr || embodiment_id_spec == nullptr ||
      initial_actions_spec == nullptr) {
    return Status::InvalidArgument("GR00T engine tensor contract is missing a required input tensor");
  }

  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*input_ids_spec, &input_ids_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*attention_mask_spec, &attention_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*pixel_values_0_spec, &pixel_values_0_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*pixel_values_1_spec, &pixel_values_1_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*state_spec, &state_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*embodiment_id_spec, &embodiment_id_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*initial_actions_spec, &initial_actions_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(backbone_, &backbone_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, groot::kInputIds, input_ids_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, groot::kAttentionMask, attention_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, groot::kPixelValues0, pixel_values_0_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, groot::kPixelValues1, pixel_values_1_));

  auto backbone_features = backbone_workspace_.named_outputs.find(std::string(groot::kBackboneFeatures));
  auto backbone_attention_mask = backbone_workspace_.named_outputs.find(std::string(groot::kBackboneAttentionMask));
  auto image_mask = backbone_workspace_.named_outputs.find(std::string(groot::kImageMask));
  auto normalized_actions = action_workspace_.named_outputs.find(std::string(groot::kNormalizedActions));
  if (backbone_features == backbone_workspace_.named_outputs.end() ||
      backbone_attention_mask == backbone_workspace_.named_outputs.end() ||
      image_mask == backbone_workspace_.named_outputs.end() ||
      normalized_actions == action_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("GR00T engine tensor contract is missing a required output tensor");
  }

  RETURN_IF_ERROR(runtime::PrepareStagePlan(action_loop_, &action_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kBackboneFeatures, *backbone_features->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kBackboneAttentionMask, *backbone_attention_mask->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kImageMask, *image_mask->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kState, state_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kEmbodimentId, embodiment_id_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, groot::kInitialActions, initial_actions_));

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status GrootOfflineRunner::RunOnce(const GrootOfflineRequest& request, GrootRunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("GR00T result is null");
  *result = GrootRunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.input_ids, &input_ids_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.attention_mask, &attention_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.pixel_values_0, &pixel_values_0_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.pixel_values_1, &pixel_values_1_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.state, &state_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.embodiment_id, &embodiment_id_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.initial_actions, &initial_actions_, stream_.get()));

  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(timers_.backbone.Start(stream_.get()));
  Status status = runtime::RunStage(backbone_, &backbone_plan_, &backbone_workspace_, stream_.get());
  if (status.ok()) status = timers_.backbone.Stop(stream_.get());
  if (status.ok()) status = timers_.action_loop.Start(stream_.get());
  if (status.ok()) status = runtime::RunStage(action_loop_, &action_plan_, &action_workspace_, stream_.get());
  if (status.ok()) status = timers_.action_loop.Stop(stream_.get());
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  if (!status.ok()) return status;

  auto* action = action_workspace_.named_outputs[std::string(groot::kNormalizedActions)];
  RETURN_IF_ERROR(CopyDeviceTensorToFloatVector(*action, stream_.get(), &result->action));
  RETURN_IF_ERROR(timers_.backbone.ElapsedMs(&result->backbone_ms));
  RETURN_IF_ERROR(timers_.action_loop.ElapsedMs(&result->action_loop_ms));
  result->action_shape = action->spec.shape.dims;
  return Status::Ok();
}

double GrootOfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> GrootOfflineRunner::input_specs() const {
  return {
      input_ids_.spec,
      attention_mask_.spec,
      pixel_values_0_.spec,
      pixel_values_1_.spec,
      state_.spec,
      embodiment_id_.spec,
      initial_actions_.spec,
  };
}

}  // namespace pi_cpp
