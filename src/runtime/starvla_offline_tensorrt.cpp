#include "pi_cpp/runtime/starvla_offline.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/starvla_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

Status CopyActionsToFloatVector(const runtime::DeviceTensor& tensor,
                                cudaStream_t stream,
                                std::vector<float>* values) {
  if (tensor.spec.dtype != DType::kFloat32) return Status::InvalidArgument("StarVLA actions must be float32");
  values->assign(static_cast<std::size_t>(tensor.spec.shape.NumElements()), 0.0F);
  RETURN_IF_ERROR(runtime::CopyDeviceToHost(tensor, values->data(), tensor.spec.NumBytes(), stream));
  return runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed for StarVLA action copy");
}

Status RunStarVlaPolicy(TrtEngine& policy,
                        runtime::StagePlan* policy_plan,
                        runtime::StageWorkspace* policy_workspace,
                        StarVlaStageTimers* timers,
                        cudaStream_t stream,
                        StarVlaRunResult* result) {
  RETURN_IF_ERROR(timers->policy.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(policy, policy_plan, policy_workspace, stream));
  RETURN_IF_ERROR(timers->policy.Stop(stream));

  auto actions = policy_workspace->named_outputs.find(std::string(starvla::kActions));
  if (actions == policy_workspace->named_outputs.end()) {
    return Status::InvalidArgument("StarVLA policy is missing actions output");
  }
  RETURN_IF_ERROR(CopyActionsToFloatVector(*actions->second, stream, &result->action));
  RETURN_IF_ERROR(timers->policy.ElapsedMs(&result->policy_ms));
  result->action_shape = actions->second->spec.shape.dims;
  return Status::Ok();
}

}  // namespace

Status StarVlaStageTimers::Init() { return policy.Init(std::string(starvla::kPolicyStage)); }

StarVlaOfflineRunner::StarVlaOfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)), policy_(engine_dir_ / std::string(starvla::kPolicyEngine)) {}

StarVlaOfflineRunner::~StarVlaOfflineRunner() = default;

Status StarVlaOfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&policy_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(policy_, &policy_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStagePlan(policy_, &policy_plan_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* input_ids_spec = FindTensorSpec(policy_.inputs(), starvla::kInputIds);
  const auto* attention_mask_spec = FindTensorSpec(policy_.inputs(), starvla::kAttentionMask);
  const auto* position_ids_spec = FindTensorSpec(policy_.inputs(), starvla::kPositionIds);
  const auto* visual_select_spec = FindTensorSpec(policy_.inputs(), starvla::kVisualSelect);
  const auto* action_select_spec = FindTensorSpec(policy_.inputs(), starvla::kActionSelect);
  const auto* pixel_values_spec = FindTensorSpec(policy_.inputs(), starvla::kPixelValues);
  if (input_ids_spec == nullptr || attention_mask_spec == nullptr || position_ids_spec == nullptr ||
      visual_select_spec == nullptr || action_select_spec == nullptr || pixel_values_spec == nullptr) {
    return Status::InvalidArgument("StarVLA policy engine tensor contract is missing a required tensor");
  }
  if (policy_workspace_.named_outputs.find(std::string(starvla::kActions)) == policy_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("StarVLA policy engine is missing actions output");
  }

  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*input_ids_spec, &input_ids_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*attention_mask_spec, &attention_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*position_ids_spec, &position_ids_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*visual_select_spec, &visual_select_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*action_select_spec, &action_select_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*pixel_values_spec, &pixel_values_));

  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kInputIds, input_ids_));
  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kAttentionMask, attention_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kPositionIds, position_ids_));
  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kVisualSelect, visual_select_));
  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kActionSelect, action_select_));
  RETURN_IF_ERROR(runtime::BindStageInput(&policy_plan_, starvla::kPixelValues, pixel_values_));

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status StarVlaOfflineRunner::RunOnce(const StarVlaOfflineRequest& request, StarVlaRunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("StarVLA result is null");
  *result = StarVlaRunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.input_ids, &input_ids_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.attention_mask, &attention_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.position_ids, &position_ids_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.visual_select, &visual_select_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.action_select, &action_select_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.pixel_values, &pixel_values_, stream_.get()));

  auto start = std::chrono::steady_clock::now();
  Status status = RunStarVlaPolicy(policy_, &policy_plan_, &policy_workspace_, &timers_, stream_.get(), result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double StarVlaOfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> StarVlaOfflineRunner::input_specs() const {
  return {
      input_ids_.spec,
      attention_mask_.spec,
      position_ids_.spec,
      visual_select_.spec,
      action_select_.spec,
      pixel_values_.spec,
  };
}

}  // namespace pi_cpp
