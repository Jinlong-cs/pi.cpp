#include "pi_cpp/runtime/dit4dit_offline.hpp"

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
#include "pi_cpp/runtime/dit4dit_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

using runtime::DeviceTensor;

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
      const std::uint32_t float_exponent = exponent + (127 - 15);
      bits = sign | (float_exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    const std::uint32_t float_exponent = exponent + (127 - 15);
    bits = sign | (float_exponent << 23) | (mantissa << 13);
  }

  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Status CopyDeviceTensorToFloatVector(const DeviceTensor& tensor, cudaStream_t stream, std::vector<float>* values) {
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
  return Status::InvalidArgument("unsupported tensor dtype for float decode: " + tensor.spec.name);
}

Status RunDit4DitStages(TrtEngine& vae,
                        TrtEngine& feature,
                        TrtEngine& action_loop,
                        runtime::StagePlan* vae_plan,
                        runtime::StagePlan* feature_plan,
                        runtime::StagePlan* action_plan,
                        runtime::StageWorkspace* vae_workspace,
                        runtime::StageWorkspace* feature_workspace,
                        runtime::StageWorkspace* action_workspace,
                        Dit4DitStageTimers* timers,
                        cudaStream_t stream,
                        Dit4DitRunResult* result) {
  RETURN_IF_ERROR(timers->vae.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(vae, vae_plan, vae_workspace, stream));
  RETURN_IF_ERROR(timers->vae.Stop(stream));

  RETURN_IF_ERROR(timers->feature.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(feature, feature_plan, feature_workspace, stream));
  RETURN_IF_ERROR(timers->feature.Stop(stream));

  RETURN_IF_ERROR(timers->action_loop.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(action_loop, action_plan, action_workspace, stream));
  RETURN_IF_ERROR(timers->action_loop.Stop(stream));

  auto action = action_workspace->named_outputs.find(std::string(dit4dit::kNormalizedActions));
  if (action == action_workspace->named_outputs.end()) {
    return Status::InvalidArgument("DiT4DiT action loop is missing normalized_actions output");
  }
  RETURN_IF_ERROR(CopyDeviceTensorToFloatVector(*action->second, stream, &result->action));
  RETURN_IF_ERROR(timers->vae.ElapsedMs(&result->vae_ms));
  RETURN_IF_ERROR(timers->feature.ElapsedMs(&result->feature_ms));
  RETURN_IF_ERROR(timers->action_loop.ElapsedMs(&result->action_loop_ms));
  result->action_shape = action->second->spec.shape.dims;
  return Status::Ok();
}

}  // namespace

Status Dit4DitStageTimers::Init() {
  RETURN_IF_ERROR(vae.Init(std::string(dit4dit::kVaeStage)));
  RETURN_IF_ERROR(feature.Init(std::string(dit4dit::kFeatureStage)));
  return action_loop.Init(std::string(dit4dit::kActionLoopStage));
}

Dit4DitOfflineRunner::Dit4DitOfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      vae_(engine_dir_ / std::string(dit4dit::kVaeEngine)),
      feature_(engine_dir_ / std::string(dit4dit::kFeatureEngine)),
      action_loop_(engine_dir_ / std::string(dit4dit::kActionLoopEngine)) {}

Dit4DitOfflineRunner::~Dit4DitOfflineRunner() = default;

Status Dit4DitOfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&vae_));
  RETURN_IF_ERROR(runtime::LoadEngine(&feature_));
  RETURN_IF_ERROR(runtime::LoadEngine(&action_loop_));

  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(vae_, &vae_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(feature_, &feature_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(action_loop_, &action_workspace_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* video_spec = FindTensorSpec(vae_.inputs(), dit4dit::kVideo);
  const auto* sample_noise_spec = FindTensorSpec(vae_.inputs(), dit4dit::kSampleNoise);
  const auto* latents_spec = FindTensorSpec(feature_.inputs(), dit4dit::kLatents);
  const auto* cond_mask_spec = FindTensorSpec(feature_.inputs(), dit4dit::kCondMask);
  const auto* cond_indicator_spec = FindTensorSpec(feature_.inputs(), dit4dit::kCondIndicator);
  const auto* prompt_embeds_spec = FindTensorSpec(feature_.inputs(), dit4dit::kPromptEmbeds);
  const auto* padding_mask_spec = FindTensorSpec(feature_.inputs(), dit4dit::kPaddingMask);
  const auto* sigma_t_spec = FindTensorSpec(feature_.inputs(), dit4dit::kSigmaT);
  const auto* state_spec = FindTensorSpec(action_loop_.inputs(), dit4dit::kState);
  const auto* initial_actions_spec = FindTensorSpec(action_loop_.inputs(), dit4dit::kInitialActions);
  if (video_spec == nullptr || sample_noise_spec == nullptr || latents_spec == nullptr ||
      cond_mask_spec == nullptr || cond_indicator_spec == nullptr || prompt_embeds_spec == nullptr ||
      padding_mask_spec == nullptr || sigma_t_spec == nullptr || state_spec == nullptr ||
      initial_actions_spec == nullptr) {
    return Status::InvalidArgument("DiT4DiT engine tensor contract is missing a required input tensor");
  }

  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*video_spec, &video_bcthw_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*sample_noise_spec, &sample_noise_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*latents_spec, &latents_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*cond_mask_spec, &cond_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*cond_indicator_spec, &cond_indicator_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*prompt_embeds_spec, &prompt_embeds_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*padding_mask_spec, &padding_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*sigma_t_spec, &sigma_t_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*state_spec, &state_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*initial_actions_spec, &initial_actions_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(vae_, &vae_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&vae_plan_, dit4dit::kVideo, video_bcthw_));
  RETURN_IF_ERROR(runtime::BindStageInput(&vae_plan_, dit4dit::kSampleNoise, sample_noise_));

  auto cond_latents = vae_workspace_.named_outputs.find(std::string(dit4dit::kCondLatents));
  if (cond_latents == vae_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("DiT4DiT VAE is missing cond_latents output");
  }
  RETURN_IF_ERROR(runtime::PrepareStagePlan(feature_, &feature_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kLatents, latents_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kCondLatents, *cond_latents->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kCondMask, cond_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kCondIndicator, cond_indicator_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kPromptEmbeds, prompt_embeds_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kPaddingMask, padding_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&feature_plan_, dit4dit::kSigmaT, sigma_t_));

  auto vl_embs = feature_workspace_.named_outputs.find(std::string(dit4dit::kVlEmbs));
  if (vl_embs == feature_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("DiT4DiT feature stage is missing vl_embs output");
  }
  RETURN_IF_ERROR(runtime::PrepareStagePlan(action_loop_, &action_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, dit4dit::kVlEmbs, *vl_embs->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, dit4dit::kState, state_));
  RETURN_IF_ERROR(runtime::BindStageInput(&action_plan_, dit4dit::kInitialActions, initial_actions_));

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status Dit4DitOfflineRunner::RunOnce(const Dit4DitOfflineRequest& request, Dit4DitRunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("DiT4DiT result is null");
  *result = Dit4DitRunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.video_bcthw, &video_bcthw_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.sample_noise, &sample_noise_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.latents, &latents_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.cond_mask, &cond_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.cond_indicator, &cond_indicator_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.prompt_embeds, &prompt_embeds_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.padding_mask, &padding_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.sigma_t, &sigma_t_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.state, &state_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.initial_actions, &initial_actions_, stream_.get()));

  auto start = std::chrono::steady_clock::now();
  Status status = RunDit4DitStages(vae_, feature_, action_loop_, &vae_plan_, &feature_plan_, &action_plan_,
                                   &vae_workspace_, &feature_workspace_, &action_workspace_, &timers_, stream_.get(),
                                   result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double Dit4DitOfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> Dit4DitOfflineRunner::input_specs() const {
  return {
      video_bcthw_.spec,
      sample_noise_.spec,
      latents_.spec,
      cond_mask_.spec,
      cond_indicator_.spec,
      prompt_embeds_.spec,
      padding_mask_.spec,
      sigma_t_.spec,
      state_.spec,
      initial_actions_.spec,
  };
}

}  // namespace pi_cpp
