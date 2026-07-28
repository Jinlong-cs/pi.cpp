#include "pi_cpp/runtime/evo1_offline.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/evo1_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

using runtime::DeviceTensor;
using runtime::StagePlan;
using runtime::StageWorkspace;

Status CopyInt64Scalar(DeviceTensor* tensor, std::int64_t value, cudaStream_t stream) {
  if (tensor->spec.dtype != DType::kInt64 || tensor->spec.shape.NumElements() != 1) {
    return Status::InvalidArgument("expected int64 scalar tensor: " + tensor->spec.name);
  }
  return runtime::CheckCuda(
      cudaMemcpyAsync(tensor->data.get(), &value, sizeof(value), cudaMemcpyHostToDevice, stream),
      "cudaMemcpy int64 scalar H2D failed for " + tensor->spec.name);
}

Status CopyActionsToFloatVector(const DeviceTensor& tensor, cudaStream_t stream, std::vector<float>* values) {
  if (tensor.spec.dtype != DType::kFloat32) return Status::InvalidArgument("Evo-1 final actions must be float32");
  values->assign(static_cast<std::size_t>(tensor.spec.shape.NumElements()), 0.0F);
  RETURN_IF_ERROR(runtime::CopyDeviceToHost(tensor, values->data(), tensor.spec.NumBytes(), stream));
  return runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed for Evo-1 action copy");
}

Status RunEvo1Stages(TrtEngine& backbone,
                     TrtEngine& action_step,
                     std::array<DeviceTensor, 2>* action_buffers,
                     DeviceTensor* timestep,
                     StagePlan* backbone_plan,
                     StagePlan* action_plan,
                     StageWorkspace* backbone_workspace,
                     StageWorkspace* action_workspace,
                     Evo1StageTimers* timers,
                     std::size_t action_input_index,
                     std::size_t action_output_index,
                     cudaStream_t stream,
                     const Evo1OfflineRequest& request,
                     Evo1RunResult* result) {
  RETURN_IF_ERROR(timers->backbone.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(backbone, backbone_plan, backbone_workspace, stream));
  RETURN_IF_ERROR(timers->backbone.Stop(stream));

  DeviceTensor* current_actions = &(*action_buffers)[0];
  DeviceTensor* next_actions = &(*action_buffers)[1];

  RETURN_IF_ERROR(timers->action_loop.Start(stream));
  for (int step = 0; step < request.steps; ++step) {
    const std::int64_t timestep_value = static_cast<std::int64_t>(
        (static_cast<float>(step) / static_cast<float>(request.steps)) * static_cast<float>(evo1::kTimestepBuckets));
    RETURN_IF_ERROR(CopyInt64Scalar(timestep, timestep_value, stream));

    action_plan->input_views[action_input_index] = current_actions->view();
    action_workspace->output_views[action_output_index] = {
        action_step.outputs()[action_output_index],
        next_actions->data.get(),
    };
    RETURN_IF_ERROR(runtime::RunStage(action_step, action_plan, action_workspace, stream));
    std::swap(current_actions, next_actions);
  }
  RETURN_IF_ERROR(timers->action_loop.Stop(stream));

  RETURN_IF_ERROR(CopyActionsToFloatVector(*current_actions, stream, &result->action));
  RETURN_IF_ERROR(timers->backbone.ElapsedMs(&result->backbone_ms));
  RETURN_IF_ERROR(timers->action_loop.ElapsedMs(&result->action_loop_ms));
  result->action_shape = current_actions->spec.shape.dims;
  return Status::Ok();
}

}  // namespace

Status Evo1StageTimers::Init() {
  RETURN_IF_ERROR(backbone.Init(std::string(evo1::kBackboneStage)));
  return action_loop.Init(std::string(evo1::kActionStepStage));
}

Evo1OfflineRunner::Evo1OfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      backbone_(engine_dir_ / std::string(evo1::kBackboneEngine)),
      action_step_(engine_dir_ / std::string(evo1::kActionStepEngine)) {}

Evo1OfflineRunner::~Evo1OfflineRunner() = default;

Status Evo1OfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&backbone_));
  RETURN_IF_ERROR(runtime::LoadEngine(&action_step_));

  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(backbone_, &backbone_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(action_step_, &action_workspace_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* input_ids_spec = FindTensorSpec(backbone_.inputs(), evo1::kInputIds);
  const auto* attention_mask_spec = FindTensorSpec(backbone_.inputs(), evo1::kAttentionMask);
  const auto* attention_mask_2d_spec = FindTensorSpec(backbone_.inputs(), evo1::kAttentionMask2d);
  const auto* position_ids_spec = FindTensorSpec(backbone_.inputs(), evo1::kPositionIds);
  const auto* pixel_values_spec = FindTensorSpec(backbone_.inputs(), evo1::kPixelValues);
  const auto* state_spec = FindTensorSpec(backbone_.inputs(), evo1::kState);
  const auto* actions_spec = FindTensorSpec(action_step_.inputs(), evo1::kActions);
  const auto* timestep_spec = FindTensorSpec(action_step_.inputs(), evo1::kTimestep);
  const auto* actions_next_spec = FindTensorSpec(action_step_.outputs(), evo1::kActionsNext);
  if (input_ids_spec == nullptr || attention_mask_spec == nullptr || attention_mask_2d_spec == nullptr ||
      position_ids_spec == nullptr || pixel_values_spec == nullptr || state_spec == nullptr || actions_spec == nullptr ||
      timestep_spec == nullptr || actions_next_spec == nullptr) {
    return Status::InvalidArgument("Evo-1 engine tensor contract is missing a required tensor");
  }
  if (actions_spec->dtype != actions_next_spec->dtype || actions_spec->NumBytes() != actions_next_spec->NumBytes()) {
    return Status::InvalidArgument("Evo-1 actions and actions_next contracts must match");
  }

  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*input_ids_spec, &input_ids_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*attention_mask_spec, &attention_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*attention_mask_2d_spec, &attention_mask_2d_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*position_ids_spec, &position_ids_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*pixel_values_spec, &pixel_values_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*state_spec, &state_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*actions_spec, &action_buffers_[0]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*actions_next_spec, &action_buffers_[1]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*timestep_spec, &timestep_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(backbone_, &backbone_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kInputIds, input_ids_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kAttentionMask, attention_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kAttentionMask2d, attention_mask_2d_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kPositionIds, position_ids_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kPixelValues, pixel_values_));
  RETURN_IF_ERROR(runtime::BindStageInput(&backbone_plan_, evo1::kState, state_));

  auto context_tokens = backbone_workspace_.named_outputs.find(std::string(evo1::kContextTokens));
  if (context_tokens == backbone_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("Evo-1 backbone is missing context_tokens output");
  }
  RETURN_IF_ERROR(runtime::PrepareStagePlan(action_step_, &action_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, evo1::kContextTokens, *context_tokens->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, evo1::kActions, action_buffers_[0]));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, evo1::kTimestep, timestep_));
  RETURN_IF_ERROR(runtime::FindStageInputIndex(action_plan_, evo1::kActions, &action_input_index_));

  action_output_index_ = action_step_.outputs().size();
  for (std::size_t index = 0; index < action_step_.outputs().size(); ++index) {
    if (action_step_.outputs()[index].name == evo1::kActionsNext) {
      action_output_index_ = index;
      break;
    }
  }
  if (action_output_index_ == action_step_.outputs().size()) {
    return Status::InvalidArgument("Evo-1 action stage is missing actions_next output");
  }

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status Evo1OfflineRunner::RunOnce(const Evo1OfflineRequest& request, Evo1RunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("Evo-1 result is null");
  if (request.steps != evo1::kInferenceSteps) {
    return Status::InvalidArgument("Evo-1 action engine requires exactly 32 flow steps");
  }
  *result = Evo1RunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.input_ids, &input_ids_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.attention_mask, &attention_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.attention_mask_2d, &attention_mask_2d_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.position_ids, &position_ids_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.pixel_values, &pixel_values_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.state, &state_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.actions, &action_buffers_[0], stream_.get()));

  auto start = std::chrono::steady_clock::now();
  Status status = RunEvo1Stages(backbone_, action_step_, &action_buffers_, &timestep_, &backbone_plan_, &action_plan_,
                               &backbone_workspace_, &action_workspace_, &timers_, action_input_index_,
                               action_output_index_, stream_.get(), request, result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double Evo1OfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> Evo1OfflineRunner::input_specs() const {
  return {
      input_ids_.spec,
      attention_mask_.spec,
      attention_mask_2d_.spec,
      position_ids_.spec,
      pixel_values_.spec,
      state_.spec,
      action_buffers_[0].spec,
  };
}

}  // namespace pi_cpp
