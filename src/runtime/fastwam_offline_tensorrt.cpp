#include "pi_cpp/runtime/fastwam_offline.hpp"

#include <cuda_runtime_api.h>

#include <array>
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
#include "pi_cpp/runtime/fastwam_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

using runtime::DeviceTensor;
using runtime::StagePlan;
using runtime::StageWorkspace;

Status ValidateFastWamRequest(const FastWamOfflineRequest& request) {
  if (request.steps <= 0) return Status::InvalidArgument("FastWAM steps must be positive");
  if (request.action_dim <= 0) return Status::InvalidArgument("FastWAM action_dim must be positive");
  if (request.input_image.data.empty()) {
    return Status::InvalidArgument("FastWAM request is missing input_image tensor data");
  }
  if (request.context.data.empty()) return Status::InvalidArgument("FastWAM request is missing context tensor data");
  if (request.context_mask.data.empty()) {
    return Status::InvalidArgument("FastWAM request is missing context_mask tensor data");
  }
  if (request.latents_action.data.empty()) {
    return Status::InvalidArgument("FastWAM request is missing latents_action tensor data");
  }
  if (request.timestep_action.data.empty()) {
    return Status::InvalidArgument("FastWAM request is missing timestep_action tensor data");
  }
  if (request.scheduler_deltas.size() < static_cast<std::size_t>(request.steps)) {
    return Status::InvalidArgument("FastWAM scheduler_deltas is smaller than steps");
  }
  if (request.action_mean.size() < static_cast<std::size_t>(request.action_dim)) {
    return Status::InvalidArgument("FastWAM action_mean is smaller than action_dim");
  }
  if (request.action_std.size() < static_cast<std::size_t>(request.action_dim)) {
    return Status::InvalidArgument("FastWAM action_std is smaller than action_dim");
  }
  return Status::Ok();
}

std::uint16_t FloatToHalfBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    if (exponent < -10) return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half_mantissa = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1U) ++half_mantissa;
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }
  if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7C00U);

  std::uint32_t half = sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
  if (mantissa & 0x1000U) ++half;
  return static_cast<std::uint16_t>(half);
}

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

Status SetFloatScalar(DeviceTensor* tensor, float value, cudaStream_t stream) {
  if (tensor->spec.shape.NumElements() != 1) {
    return Status::InvalidArgument("expected scalar tensor: " + tensor->spec.name);
  }
  if (tensor->spec.dtype == DType::kFloat32) {
    return runtime::CheckCuda(cudaMemcpyAsync(tensor->data.get(), &value, sizeof(value), cudaMemcpyHostToDevice, stream),
                              "cudaMemcpy scalar H2D failed for " + tensor->spec.name);
  }
  if (tensor->spec.dtype == DType::kFloat16) {
    const std::uint16_t half_value = FloatToHalfBits(value);
    return runtime::CheckCuda(
        cudaMemcpyAsync(tensor->data.get(), &half_value, sizeof(half_value), cudaMemcpyHostToDevice, stream),
        "cudaMemcpy half scalar H2D failed for " + tensor->spec.name);
  }
  return Status::InvalidArgument("unsupported scalar dtype for " + tensor->spec.name);
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
    RETURN_IF_ERROR(runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed for " + tensor.spec.name));
    for (std::size_t index = 0; index < elements; ++index) (*values)[index] = HalfBitsToFloat(half_values[index]);
    return Status::Ok();
  }
  return Status::InvalidArgument("unsupported tensor dtype for float decode: " + tensor.spec.name);
}

Status CopyTimestepToDevice(const FastWamOfflineRequest& request,
                            int step,
                            const TensorSpec& expected,
                            DeviceTensor* timestep,
                            cudaStream_t stream) {
  if (request.timestep_action.dtype != expected.dtype) {
    return Status::InvalidArgument("dtype mismatch for tensor: " + expected.name);
  }
  if (expected.shape.NumElements() != 1) {
    return Status::InvalidArgument("FastWAM timestep_action engine input must contain one element");
  }
  const auto bytes_per_step = expected.NumBytes();
  const auto offset = static_cast<std::size_t>(step) * bytes_per_step;
  if (request.timestep_action.data.size() < offset + bytes_per_step) {
    return Status::InvalidArgument("FastWAM timestep_action host tensor is smaller than steps");
  }
  return runtime::CheckCuda(
      cudaMemcpyAsync(timestep->data.get(), request.timestep_action.data.data() + offset, bytes_per_step,
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpy timestep H2D failed for " + expected.name);
}

Status DecodeAction(const std::vector<float>& normalized,
                    const FastWamOfflineRequest& request,
                    int64_t batch,
                    int64_t horizon,
                    int64_t latent_dim,
                    std::vector<float>* action) {
  if (request.action_dim > latent_dim) return Status::InvalidArgument("FastWAM action_dim exceeds latent dim");
  action->assign(static_cast<std::size_t>(batch * horizon * request.action_dim), 0.0F);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t t = 0; t < horizon; ++t) {
      for (int d = 0; d < request.action_dim; ++d) {
        const auto raw_index = static_cast<std::size_t>((b * horizon + t) * latent_dim + d);
        const auto out_index = static_cast<std::size_t>((b * horizon + t) * request.action_dim + d);
        (*action)[out_index] =
            normalized[raw_index] * request.action_std[static_cast<std::size_t>(d)] +
            request.action_mean[static_cast<std::size_t>(d)];
      }
    }
  }
  return Status::Ok();
}

Status BindActionPlanInputs(const TrtEngine& action_step,
                            const StageWorkspace& video_workspace,
                            DeviceTensor* latents_action,
                            DeviceTensor* timestep_action,
                            DeviceTensor* scheduler_delta,
                            DeviceTensor* context,
                            DeviceTensor* context_mask,
                            StagePlan* action_plan,
                            int* kv_direct) {
  *kv_direct = 0;
  for (const auto& spec : action_step.inputs()) {
    if (spec.name == fastwam::kLatentsAction) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *latents_action));
      continue;
    }
    if (spec.name == fastwam::kTimestepAction) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *timestep_action));
      continue;
    }
    if (spec.name == fastwam::kSchedulerDelta) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *scheduler_delta));
      continue;
    }
    if (spec.name == fastwam::kContext) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *context));
      continue;
    }
    if (spec.name == fastwam::kContextMask) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *context_mask));
      continue;
    }

    auto video_it = video_workspace.named_outputs.find(spec.name);
    if (video_it == video_workspace.named_outputs.end()) {
      return Status::InvalidArgument("missing FastWAM action input: " + spec.name);
    }
    DeviceTensor* source = video_it->second;
    if (source->spec.dtype == spec.dtype && source->spec.NumBytes() == spec.NumBytes()) {
      RETURN_IF_ERROR(runtime::BindStageInput(action_plan, spec.name, *source));
      ++(*kv_direct);
      continue;
    }
    return Status::InvalidArgument("FastWAM requires direct FP16 KV handoff for action input: " + spec.name);
  }
  return Status::Ok();
}

Status ValidateKvHandoffContract(const TrtEngine& video_prefill, const TrtEngine& action_step) {
  for (const auto& spec : action_step.inputs()) {
    if (spec.name == fastwam::kLatentsAction || spec.name == fastwam::kTimestepAction ||
        spec.name == fastwam::kSchedulerDelta || spec.name == fastwam::kContext ||
        spec.name == fastwam::kContextMask) {
      continue;
    }
    const auto* source = FindTensorSpec(video_prefill.outputs(), spec.name);
    if (source == nullptr) return Status::InvalidArgument("missing FastWAM video output for action input: " + spec.name);
    if (source->dtype != DType::kFloat16 || spec.dtype != DType::kFloat16 || source->NumBytes() != spec.NumBytes()) {
      return Status::InvalidArgument(
          "FastWAM video/action KV contract must be direct FP16 for " + spec.name +
          "; rebuild video_prefill with all video_k/video_v outputs exported as FP16");
    }
  }
  return Status::Ok();
}

Status RunFastWamStages(TrtEngine& vae_image_encoder,
                        TrtEngine& video_prefill,
                        TrtEngine& action_step,
                        std::array<DeviceTensor, 2>* latents_action_buffers,
                        DeviceTensor* timestep_action,
                        DeviceTensor* scheduler_delta,
                        StagePlan* vae_plan,
                        StagePlan* video_plan,
                        StagePlan* action_plan,
                        StageWorkspace* vae_workspace,
                        StageWorkspace* video_workspace,
                        StageWorkspace* action_workspace,
                        FastWamStageTimers* timers,
                        std::size_t action_latents_input_index,
                        std::size_t action_latents_next_output_index,
                        cudaStream_t stream,
                        const FastWamOfflineRequest& request,
                        FastWamRunResult* result) {
  RETURN_IF_ERROR(timers->vae_image_encoder.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(vae_image_encoder, vae_plan, vae_workspace, stream));
  RETURN_IF_ERROR(timers->vae_image_encoder.Stop(stream));

  RETURN_IF_ERROR(timers->video_prefill.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(video_prefill, video_plan, video_workspace, stream));
  RETURN_IF_ERROR(timers->video_prefill.Stop(stream));

  RETURN_IF_ERROR(timers->action_loop.Start(stream));

  DeviceTensor* current_latents = &(*latents_action_buffers)[0];
  DeviceTensor* next_latents = &(*latents_action_buffers)[1];

  const auto* timestep_spec = FindTensorSpec(action_step.inputs(), fastwam::kTimestepAction);
  if (timestep_spec == nullptr) return Status::InvalidArgument("FastWAM action engine is missing timestep_action input");

  for (int step = 0; step < request.steps; ++step) {
    RETURN_IF_ERROR(CopyTimestepToDevice(request, step, *timestep_spec, timestep_action, stream));
    RETURN_IF_ERROR(SetFloatScalar(scheduler_delta, request.scheduler_deltas[static_cast<std::size_t>(step)], stream));

    action_plan->input_views[action_latents_input_index] = {
        action_step.inputs()[action_latents_input_index],
        current_latents->data.get(),
    };
    action_workspace->output_views[action_latents_next_output_index] = {
        action_step.outputs()[action_latents_next_output_index],
        next_latents->data.get(),
    };

    RETURN_IF_ERROR(runtime::RunStage(action_step, action_plan, action_workspace, stream));
    std::swap(current_latents, next_latents);
  }
  RETURN_IF_ERROR(timers->action_loop.Stop(stream));

  const auto& latents_spec = current_latents->spec;
  if (latents_spec.shape.dims.size() != 3) return Status::InvalidArgument("FastWAM latents_action must have rank 3");
  std::vector<float> normalized_latents;
  RETURN_IF_ERROR(CopyDeviceTensorToFloatVector(*current_latents, stream, &normalized_latents));
  const auto decode_start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(DecodeAction(normalized_latents, request, latents_spec.shape.dims[0], latents_spec.shape.dims[1],
                               latents_spec.shape.dims[2], &result->action));
  result->action_decode_ms = runtime::ElapsedMs(decode_start);

  RETURN_IF_ERROR(timers->vae_image_encoder.ElapsedMs(&result->vae_image_encoder_ms));
  RETURN_IF_ERROR(timers->video_prefill.ElapsedMs(&result->video_prefill_ms));
  RETURN_IF_ERROR(timers->action_loop.ElapsedMs(&result->action_loop_ms));
  result->action_shape = {latents_spec.shape.dims[0], latents_spec.shape.dims[1], request.action_dim};
  return Status::Ok();
}

}  // namespace

Status FastWamStageTimers::Init() {
  RETURN_IF_ERROR(vae_image_encoder.Init(std::string(fastwam::kVaeStage)));
  RETURN_IF_ERROR(video_prefill.Init(std::string(fastwam::kVideoPrefillStage)));
  return action_loop.Init(std::string(fastwam::kActionStepStage));
}

FastWamOfflineRunner::FastWamOfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      vae_image_encoder_(engine_dir_ / std::string(fastwam::kVaeEngine)),
      video_prefill_(engine_dir_ / std::string(fastwam::kVideoPrefillEngine)),
      action_step_(engine_dir_ / std::string(fastwam::kActionStepEngine)) {}

FastWamOfflineRunner::~FastWamOfflineRunner() = default;

Status FastWamOfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&vae_image_encoder_));
  RETURN_IF_ERROR(runtime::LoadEngine(&video_prefill_));
  RETURN_IF_ERROR(runtime::LoadEngine(&action_step_));

  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(vae_image_encoder_, &vae_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(video_prefill_, &video_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(action_step_, &action_workspace_));
  RETURN_IF_ERROR(ValidateKvHandoffContract(video_prefill_, action_step_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* image_spec = FindTensorSpec(vae_image_encoder_.inputs(), fastwam::kInputImage);
  const auto* context_spec = FindTensorSpec(video_prefill_.inputs(), fastwam::kContext);
  const auto* context_mask_spec = FindTensorSpec(video_prefill_.inputs(), fastwam::kContextMask);
  const auto* latents_spec = FindTensorSpec(action_step_.inputs(), fastwam::kLatentsAction);
  const auto* timestep_spec = FindTensorSpec(action_step_.inputs(), fastwam::kTimestepAction);
  const auto* scheduler_delta_spec = FindTensorSpec(action_step_.inputs(), fastwam::kSchedulerDelta);
  const auto* latents_next_spec = FindTensorSpec(action_step_.outputs(), fastwam::kLatentsActionNext);
  if (image_spec == nullptr || context_spec == nullptr || context_mask_spec == nullptr || latents_spec == nullptr ||
      timestep_spec == nullptr || scheduler_delta_spec == nullptr || latents_next_spec == nullptr) {
    return Status::InvalidArgument("FastWAM engine tensor contract is missing a required input tensor");
  }
  if (latents_spec->dtype != latents_next_spec->dtype || latents_spec->NumBytes() != latents_next_spec->NumBytes()) {
    return Status::InvalidArgument("FastWAM latents_action and latents_action_next contracts must match");
  }
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*image_spec, &input_image_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*context_spec, &context_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*context_mask_spec, &context_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*latents_spec, &latents_action_buffers_[0]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*latents_next_spec, &latents_action_buffers_[1]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*timestep_spec, &timestep_action_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*scheduler_delta_spec, &scheduler_delta_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(vae_image_encoder_, &vae_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&vae_plan_, fastwam::kInputImage, input_image_));

  auto first_frame_latents = vae_workspace_.named_outputs.find(std::string(fastwam::kFirstFrameLatents));
  if (first_frame_latents == vae_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("FastWAM VAE is missing first_frame_latents output");
  }
  RETURN_IF_ERROR(runtime::PrepareStagePlan(video_prefill_, &video_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&video_plan_, fastwam::kFirstFrameLatents, *first_frame_latents->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&video_plan_, fastwam::kContext, context_));
  RETURN_IF_ERROR(runtime::BindStageInput(&video_plan_, fastwam::kContextMask, context_mask_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(action_step_, &action_plan_));
  RETURN_IF_ERROR(BindActionPlanInputs(action_step_, video_workspace_, &latents_action_buffers_[0], &timestep_action_,
                                       &scheduler_delta_, &context_, &context_mask_, &action_plan_, &kv_direct_));

  RETURN_IF_ERROR(runtime::FindStageInputIndex(action_plan_, fastwam::kLatentsAction, &action_latents_input_index_));
  action_latents_next_output_index_ = action_step_.outputs().size();
  for (std::size_t index = 0; index < action_step_.outputs().size(); ++index) {
    if (action_step_.outputs()[index].name == fastwam::kLatentsActionNext) {
      action_latents_next_output_index_ = index;
      break;
    }
  }
  if (action_latents_next_output_index_ == action_step_.outputs().size()) {
    return Status::InvalidArgument("FastWAM action stage is missing latents_action_next output");
  }

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status FastWamOfflineRunner::RunOnce(const FastWamOfflineRequest& request, FastWamRunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("FastWAM result is null");
  *result = FastWamRunResult{};
  result->load_ms = load_ms_;
  result->kv_direct = kv_direct_;
  result->kv_cast = 0;

  RETURN_IF_ERROR(ValidateFastWamRequest(request));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.input_image, &input_image_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.context, &context_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.context_mask, &context_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.latents_action, &latents_action_buffers_[0], stream_.get()));

  auto start = std::chrono::steady_clock::now();
  Status status = RunFastWamStages(vae_image_encoder_, video_prefill_, action_step_, &latents_action_buffers_,
                                   &timestep_action_, &scheduler_delta_, &vae_plan_, &video_plan_, &action_plan_,
                                   &vae_workspace_, &video_workspace_, &action_workspace_, &timers_,
                                   action_latents_input_index_, action_latents_next_output_index_, stream_.get(), request,
                                   result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double FastWamOfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> FastWamOfflineRunner::input_specs() const {
  return {
      input_image_.spec,
      context_.spec,
      context_mask_.spec,
      latents_action_buffers_[0].spec,
      timestep_action_.spec,
  };
}

Status RunFastWamOnce(const std::filesystem::path& engine_dir,
                      const FastWamOfflineRequest& request,
                      FastWamRunResult* result) {
  FastWamOfflineRunner runner(engine_dir);
  RETURN_IF_ERROR(runner.Load());
  return runner.RunOnce(request, result);
}

}  // namespace pi_cpp
