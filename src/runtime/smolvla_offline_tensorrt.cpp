#include "pi_cpp/runtime/smolvla_offline.hpp"

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
#include "pi_cpp/runtime/smolvla_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

using runtime::DeviceTensor;
using runtime::StagePlan;
using runtime::StageWorkspace;

Status ValidateSmolVlaRequest(const SmolVlaOfflineRequest& request) {
  if (request.action_dim <= 0) return Status::InvalidArgument("SmolVLA action_dim must be positive");
  if (request.image.data.empty()) return Status::InvalidArgument("SmolVLA request is missing image tensor data");
  if (request.image_mask.data.empty()) {
    return Status::InvalidArgument("SmolVLA request is missing image_mask tensor data");
  }
  if (request.tokenized_prompt.data.empty()) {
    return Status::InvalidArgument("SmolVLA request is missing tokenized_prompt tensor data");
  }
  if (request.tokenized_prompt_mask.data.empty()) {
    return Status::InvalidArgument("SmolVLA request is missing tokenized_prompt_mask tensor data");
  }
  if (request.state.data.empty()) return Status::InvalidArgument("SmolVLA request is missing state tensor data");
  if (request.x_t.data.empty()) return Status::InvalidArgument("SmolVLA request is missing x_t tensor data");
  if (request.action_mean.size() < static_cast<std::size_t>(request.action_dim)) {
    return Status::InvalidArgument("SmolVLA action_mean is smaller than action_dim");
  }
  if (request.action_std.size() < static_cast<std::size_t>(request.action_dim)) {
    return Status::InvalidArgument("SmolVLA action_std is smaller than action_dim");
  }
  return Status::Ok();
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

Status DecodeAction(const DeviceTensor& x_t,
                    const SmolVlaOfflineRequest& request,
                    cudaStream_t stream,
                    std::vector<float>* action) {
  if (x_t.spec.shape.dims.size() != 3) return Status::InvalidArgument("SmolVLA final x_t must have rank 3");
  const int64_t batch = x_t.spec.shape.dims[0];
  const int64_t horizon = x_t.spec.shape.dims[1];
  const int64_t latent_dim = x_t.spec.shape.dims[2];
  if (request.action_dim > latent_dim) return Status::InvalidArgument("SmolVLA action_dim exceeds x_t latent dim");

  std::vector<float> normalized;
  RETURN_IF_ERROR(CopyDeviceTensorToFloatVector(x_t, stream, &normalized));

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

Status RunSmolVlaStages(TrtEngine& prefix_embed,
                        TrtEngine& prefix_lm,
                        TrtEngine& suffix_loop,
                        StagePlan* prefix_embed_plan,
                        StagePlan* prefix_lm_plan,
                        StagePlan* suffix_loop_plan,
                        StageWorkspace* prefix_embed_workspace,
                        StageWorkspace* prefix_lm_workspace,
                        StageWorkspace* suffix_loop_workspace,
                        SmolVlaStageTimers* timers,
                        cudaStream_t stream,
                        const SmolVlaOfflineRequest& request,
                        SmolVlaRunResult* result) {
  RETURN_IF_ERROR(timers->prefix_embed.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(prefix_embed, prefix_embed_plan, prefix_embed_workspace, stream));
  RETURN_IF_ERROR(timers->prefix_embed.Stop(stream));

  RETURN_IF_ERROR(timers->prefix_lm.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(prefix_lm, prefix_lm_plan, prefix_lm_workspace, stream));
  RETURN_IF_ERROR(timers->prefix_lm.Stop(stream));

  RETURN_IF_ERROR(timers->suffix_loop.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(suffix_loop, suffix_loop_plan, suffix_loop_workspace, stream));
  RETURN_IF_ERROR(timers->suffix_loop.Stop(stream));

  auto x_t_out = suffix_loop_workspace->named_outputs.find(std::string(smolvla::kSuffixLoopOutputXT));
  if (x_t_out == suffix_loop_workspace->named_outputs.end()) {
    return Status::InvalidArgument("suffix_loop output contract is missing " +
                                   std::string(smolvla::kSuffixLoopOutputXT));
  }
  RETURN_IF_ERROR(DecodeAction(*x_t_out->second, request, stream, &result->action));
  RETURN_IF_ERROR(timers->prefix_embed.ElapsedMs(&result->prefix_embed_ms));
  RETURN_IF_ERROR(timers->prefix_lm.ElapsedMs(&result->prefix_lm_ms));
  RETURN_IF_ERROR(timers->suffix_loop.ElapsedMs(&result->suffix_loop_ms));
  result->action_shape = {x_t_out->second->spec.shape.dims[0], x_t_out->second->spec.shape.dims[1], request.action_dim};
  return Status::Ok();
}

}  // namespace

Status SmolVlaStageTimers::Init() {
  RETURN_IF_ERROR(prefix_embed.Init(std::string(smolvla::kPrefixEmbedStage)));
  RETURN_IF_ERROR(prefix_lm.Init(std::string(smolvla::kPrefixLmStage)));
  return suffix_loop.Init(std::string(smolvla::kSuffixLoopStage));
}

SmolVlaOfflineRunner::SmolVlaOfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      prefix_embed_(engine_dir_ / std::string(smolvla::kPrefixEmbedEngine)),
      prefix_lm_(engine_dir_ / std::string(smolvla::kPrefixLmEngine)),
      suffix_loop_(engine_dir_ / std::string(smolvla::kSuffixLoopEngine)) {}

SmolVlaOfflineRunner::~SmolVlaOfflineRunner() = default;

Status SmolVlaOfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&prefix_embed_));
  RETURN_IF_ERROR(runtime::LoadEngine(&prefix_lm_));
  RETURN_IF_ERROR(runtime::LoadEngine(&suffix_loop_));

  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(prefix_embed_, &prefix_embed_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(prefix_lm_, &prefix_lm_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(suffix_loop_, &suffix_loop_workspace_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* image_spec = FindTensorSpec(prefix_embed_.inputs(), smolvla::kImage);
  const auto* image_mask_spec = FindTensorSpec(prefix_embed_.inputs(), smolvla::kImageMask);
  const auto* tokenized_prompt_spec = FindTensorSpec(prefix_embed_.inputs(), smolvla::kTokenizedPrompt);
  const auto* tokenized_prompt_mask_spec = FindTensorSpec(prefix_embed_.inputs(), smolvla::kTokenizedPromptMask);
  const auto* state_spec = FindTensorSpec(prefix_embed_.inputs(), smolvla::kState);
  const auto* x_t_spec = FindTensorSpec(suffix_loop_.inputs(), smolvla::kSuffixLoopInputXT);
  if (image_spec == nullptr || image_mask_spec == nullptr || tokenized_prompt_spec == nullptr ||
      tokenized_prompt_mask_spec == nullptr || state_spec == nullptr || x_t_spec == nullptr) {
    return Status::InvalidArgument("SmolVLA engine tensor contract is missing a required input tensor");
  }
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*image_spec, &image_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*image_mask_spec, &image_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*tokenized_prompt_spec, &tokenized_prompt_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*tokenized_prompt_mask_spec, &tokenized_prompt_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*state_spec, &state_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*x_t_spec, &x_t_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(prefix_embed_, &prefix_embed_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, smolvla::kImage, image_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, smolvla::kImageMask, image_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, smolvla::kTokenizedPrompt, tokenized_prompt_));
  RETURN_IF_ERROR(
      runtime::BindStageInput(&prefix_embed_plan_, smolvla::kTokenizedPromptMask, tokenized_prompt_mask_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, smolvla::kState, state_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(prefix_lm_, &prefix_lm_plan_));
  auto prefix_embs = prefix_embed_workspace_.named_outputs.find(std::string(smolvla::kPrefixEmbs));
  auto prefix_pad_masks = prefix_embed_workspace_.named_outputs.find(std::string(smolvla::kPrefixPadMasks));
  auto prefix_att_masks = prefix_embed_workspace_.named_outputs.find(std::string(smolvla::kPrefixAttMasks));
  if (prefix_embs == prefix_embed_workspace_.named_outputs.end() ||
      prefix_pad_masks == prefix_embed_workspace_.named_outputs.end() ||
      prefix_att_masks == prefix_embed_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("prefix_embed is missing a required SmolVLA output tensor");
  }
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, smolvla::kPrefixEmbs, *prefix_embs->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, smolvla::kPrefixPadMasks, *prefix_pad_masks->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, smolvla::kPrefixAttMasks, *prefix_att_masks->second));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(suffix_loop_, &suffix_loop_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_loop_plan_, smolvla::kSuffixLoopInputXT, x_t_));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_loop_plan_, smolvla::kPrefixPadMasks, *prefix_pad_masks->second));
  if (suffix_loop_workspace_.named_outputs.find(std::string(smolvla::kSuffixLoopOutputXT)) ==
      suffix_loop_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("suffix_loop output contract is missing " +
                                   std::string(smolvla::kSuffixLoopOutputXT));
  }
  for (const auto& spec : suffix_loop_.inputs()) {
    if (spec.name == smolvla::kSuffixLoopInputXT || spec.name == smolvla::kPrefixPadMasks) {
      continue;
    }
    auto cache_it = prefix_lm_workspace_.named_outputs.find(spec.name);
    if (cache_it == prefix_lm_workspace_.named_outputs.end()) {
      return Status::InvalidArgument("missing suffix input: " + spec.name);
    }
    RETURN_IF_ERROR(runtime::BindStageInput(&suffix_loop_plan_, spec.name, *cache_it->second));
  }

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status SmolVlaOfflineRunner::RunOnce(const SmolVlaOfflineRequest& request, SmolVlaRunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("SmolVLA result is null");
  *result = SmolVlaRunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(ValidateSmolVlaRequest(request));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.image, &image_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.image_mask, &image_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.tokenized_prompt, &tokenized_prompt_, stream_.get()));
  RETURN_IF_ERROR(
      runtime::CopyHostToDevice(request.tokenized_prompt_mask, &tokenized_prompt_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.state, &state_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.x_t, &x_t_, stream_.get()));

  auto start = std::chrono::steady_clock::now();
  Status status = RunSmolVlaStages(prefix_embed_, prefix_lm_, suffix_loop_, &prefix_embed_plan_, &prefix_lm_plan_,
                                   &suffix_loop_plan_, &prefix_embed_workspace_, &prefix_lm_workspace_,
                                   &suffix_loop_workspace_, &timers_, stream_.get(), request, result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double SmolVlaOfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> SmolVlaOfflineRunner::input_specs() const {
  return {
      image_.spec,
      image_mask_.spec,
      tokenized_prompt_.spec,
      tokenized_prompt_mask_.spec,
      state_.spec,
      x_t_.spec,
  };
}

}  // namespace pi_cpp
