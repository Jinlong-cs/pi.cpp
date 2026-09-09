#include "pi_cpp/runtime/pi05_offline.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "pi_cpp/core/status.hpp"
#include "pi_cpp/core/tensor.hpp"
#include "pi_cpp/core/trt_engine.hpp"
#include "pi_cpp/runtime/pi05_contract.hpp"
#include "pi_cpp/runtime/utils/tensorrt_runtime.hpp"

namespace pi_cpp {

namespace {

using runtime::DeviceTensor;
using runtime::StagePlan;
using runtime::StageWorkspace;

Status ValidatePi05Request(const Pi05OfflineRequest& request) {
  if (request.image.data.empty()) return Status::InvalidArgument("PI0.5 request is missing image tensor data");
  if (request.image_mask.data.empty()) {
    return Status::InvalidArgument("PI0.5 request is missing image_mask tensor data");
  }
  if (request.tokenized_prompt.data.empty()) {
    return Status::InvalidArgument("PI0.5 request is missing tokenized_prompt tensor data");
  }
  if (request.tokenized_prompt_mask.data.empty()) {
    return Status::InvalidArgument("PI0.5 request is missing tokenized_prompt_mask tensor data");
  }
  if (request.x_t.data.empty()) return Status::InvalidArgument("PI0.5 request is missing x_t tensor data");
  return Status::Ok();
}

Status CopyNormalizedAction(const DeviceTensor& x_t, cudaStream_t stream, std::vector<float>* action) {
  if (x_t.spec.dtype != DType::kFloat32) return Status::InvalidArgument("PI0.5 final x_t must be float32");
  if (x_t.spec.shape.dims.size() != 3) return Status::InvalidArgument("PI0.5 final x_t must have rank 3");
  const int64_t batch = x_t.spec.shape.dims[0];
  const int64_t horizon = x_t.spec.shape.dims[1];
  const int64_t latent_dim = x_t.spec.shape.dims[2];
  if (batch <= 0 || horizon <= 0 || latent_dim <= 0) {
    return Status::InvalidArgument("PI0.5 final x_t has invalid shape");
  }

  action->assign(static_cast<std::size_t>(x_t.spec.shape.NumElements()), 0.0F);
  RETURN_IF_ERROR(runtime::CopyDeviceToHost(x_t, action->data(), x_t.spec.NumBytes(), stream));
  RETURN_IF_ERROR(runtime::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed before action copy"));
  return Status::Ok();
}

Status RunPi05Stages(TrtEngine& prefix_embed,
                     TrtEngine& prefix_lm,
                     TrtEngine& suffix_step,
                     std::array<DeviceTensor, 2>* x_t_buffers,
                     DeviceTensor* timestep,
                     DeviceTensor* dt,
                     StagePlan* prefix_embed_plan,
                     StagePlan* prefix_lm_plan,
                     StagePlan* suffix_step_plan,
                     StageWorkspace* prefix_embed_workspace,
                     StageWorkspace* prefix_lm_workspace,
                     StageWorkspace* suffix_step_workspace,
                     Pi05StageTimers* timers,
                     std::size_t suffix_x_t_input_index,
                     std::size_t suffix_x_t_next_output_index,
                     cudaStream_t stream,
                     Pi05RunResult* result) {
  RETURN_IF_ERROR(timers->prefix_embed.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(prefix_embed, prefix_embed_plan, prefix_embed_workspace, stream));
  RETURN_IF_ERROR(timers->prefix_embed.Stop(stream));

  RETURN_IF_ERROR(timers->prefix_lm.Start(stream));
  RETURN_IF_ERROR(runtime::RunStage(prefix_lm, prefix_lm_plan, prefix_lm_workspace, stream));
  RETURN_IF_ERROR(timers->prefix_lm.Stop(stream));

  RETURN_IF_ERROR(runtime::SetFloat32Scalar(dt, pi05::kDefaultDt, stream));

  DeviceTensor* current_x_t = &(*x_t_buffers)[0];
  DeviceTensor* next_x_t = &(*x_t_buffers)[1];

  RETURN_IF_ERROR(timers->suffix_loop.Start(stream));
  for (int step = 0; step < pi05::kDefaultDenoiseSteps; ++step) {
    RETURN_IF_ERROR(runtime::SetFloat32Scalar(timestep, 1.0F + static_cast<float>(step) * pi05::kDefaultDt, stream));

    suffix_step_plan->input_views[suffix_x_t_input_index] = current_x_t->view();
    suffix_step_workspace->output_views[suffix_x_t_next_output_index] = next_x_t->view();

    RETURN_IF_ERROR(runtime::RunStage(suffix_step, suffix_step_plan, suffix_step_workspace, stream));
    std::swap(current_x_t, next_x_t);
  }
  RETURN_IF_ERROR(timers->suffix_loop.Stop(stream));

  RETURN_IF_ERROR(CopyNormalizedAction(*current_x_t, stream, &result->action));
  RETURN_IF_ERROR(timers->prefix_embed.ElapsedMs(&result->prefix_embed_ms));
  RETURN_IF_ERROR(timers->prefix_lm.ElapsedMs(&result->prefix_lm_ms));
  RETURN_IF_ERROR(timers->suffix_loop.ElapsedMs(&result->suffix_loop_ms));
  return Status::Ok();
}

}  // namespace

Status Pi05StageTimers::Init() {
  RETURN_IF_ERROR(prefix_embed.Init(std::string(pi05::kPrefixEmbedStage)));
  RETURN_IF_ERROR(prefix_lm.Init(std::string(pi05::kPrefixLmStage)));
  return suffix_loop.Init(std::string(pi05::kSuffixLoopStage));
}

Pi05OfflineRunner::Pi05OfflineRunner(std::filesystem::path engine_dir)
    : engine_dir_(std::move(engine_dir)),
      prefix_embed_(engine_dir_ / std::string(pi05::kPrefixEmbedEngine)),
      prefix_lm_(engine_dir_ / std::string(pi05::kPrefixLmEngine)),
      suffix_step_(engine_dir_ / std::string(pi05::kSuffixStepEngine)) {}

Pi05OfflineRunner::~Pi05OfflineRunner() = default;

Status Pi05OfflineRunner::Load() {
  auto start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(runtime::LoadEngine(&prefix_embed_));
  RETURN_IF_ERROR(runtime::LoadEngine(&prefix_lm_));
  RETURN_IF_ERROR(runtime::LoadEngine(&suffix_step_));

  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(prefix_embed_, &prefix_embed_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(prefix_lm_, &prefix_lm_workspace_));
  RETURN_IF_ERROR(runtime::PrepareStageWorkspace(suffix_step_, &suffix_step_workspace_));
  RETURN_IF_ERROR(timers_.Init());
  RETURN_IF_ERROR(stream_.Init());

  const auto* image_spec = FindTensorSpec(prefix_embed_.inputs(), pi05::kPrefixEmbedInputImage);
  const auto* image_mask_spec = FindTensorSpec(prefix_embed_.inputs(), pi05::kPrefixEmbedInputImageMask);
  const auto* tokenized_prompt_spec = FindTensorSpec(prefix_embed_.inputs(), pi05::kPrefixEmbedInputTokenizedPrompt);
  const auto* tokenized_prompt_mask_spec =
      FindTensorSpec(prefix_embed_.inputs(), pi05::kPrefixEmbedInputTokenizedPromptMask);
  const auto* x_t_spec = FindTensorSpec(suffix_step_.inputs(), pi05::kSuffixStepInputXT);
  const auto* timestep_spec = FindTensorSpec(suffix_step_.inputs(), pi05::kSuffixStepInputTimestep);
  const auto* dt_spec = FindTensorSpec(suffix_step_.inputs(), pi05::kSuffixStepInputDt);
  if (image_spec == nullptr || image_mask_spec == nullptr || tokenized_prompt_spec == nullptr ||
      tokenized_prompt_mask_spec == nullptr || x_t_spec == nullptr || timestep_spec == nullptr || dt_spec == nullptr) {
    return Status::InvalidArgument("PI0.5 engine tensor contract is missing a required input tensor");
  }
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*image_spec, &image_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*image_mask_spec, &image_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*tokenized_prompt_spec, &tokenized_prompt_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*tokenized_prompt_mask_spec, &tokenized_prompt_mask_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*x_t_spec, &x_t_buffers_[0]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*x_t_spec, &x_t_buffers_[1]));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*timestep_spec, &timestep_));
  RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*dt_spec, &dt_));

  const auto* state_spec = FindTensorSpec(suffix_step_.inputs(), pi05::kSuffixStepInputState);
  const auto* embodiment_spec = FindTensorSpec(suffix_step_.inputs(), pi05::kSuffixStepInputEmbodimentId);
  has_state_input_ = state_spec != nullptr;
  has_embodiment_input_ = embodiment_spec != nullptr;
  if (has_state_input_) {
    RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*state_spec, &state_));
  }
  if (has_embodiment_input_) {
    RETURN_IF_ERROR(runtime::AllocateDeviceTensor(*embodiment_spec, &embodiment_id_));
  }

  RETURN_IF_ERROR(runtime::PrepareStagePlan(prefix_embed_, &prefix_embed_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, pi05::kPrefixEmbedInputImage, image_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, pi05::kPrefixEmbedInputImageMask, image_mask_));
  RETURN_IF_ERROR(
      runtime::BindStageInput(&prefix_embed_plan_, pi05::kPrefixEmbedInputTokenizedPrompt, tokenized_prompt_));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_embed_plan_, pi05::kPrefixEmbedInputTokenizedPromptMask,
                                          tokenized_prompt_mask_));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(prefix_lm_, &prefix_lm_plan_));

  auto prefix_embs = prefix_embed_workspace_.named_outputs.find(std::string(pi05::kPrefixEmbedOutputPrefixEmbs));
  auto prefix_position_ids =
      prefix_embed_workspace_.named_outputs.find(std::string(pi05::kPrefixEmbedOutputPrefixPositionIds));
  auto prefix_attention_mask_4d =
      prefix_embed_workspace_.named_outputs.find(std::string(pi05::kPrefixEmbedOutputPrefixAttentionMask4d));
  auto prefix_pad_masks =
      prefix_embed_workspace_.named_outputs.find(std::string(pi05::kPrefixEmbedOutputPrefixPadMasks));
  if (prefix_embs == prefix_embed_workspace_.named_outputs.end() ||
      prefix_position_ids == prefix_embed_workspace_.named_outputs.end() ||
      prefix_attention_mask_4d == prefix_embed_workspace_.named_outputs.end() ||
      prefix_pad_masks == prefix_embed_workspace_.named_outputs.end()) {
    return Status::InvalidArgument("prefix_embed is missing a required PI0.5 output tensor");
  }
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, pi05::kPrefixLmInputPrefixEmbs, *prefix_embs->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, pi05::kPrefixLmInputPrefixPositionIds,
                                          *prefix_position_ids->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&prefix_lm_plan_, pi05::kPrefixLmInputPrefixAttentionMask4d,
                                          *prefix_attention_mask_4d->second));

  RETURN_IF_ERROR(runtime::PrepareStagePlan(suffix_step_, &suffix_step_plan_));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputPrefixPadMasks, *prefix_pad_masks->second));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputXT, x_t_buffers_[0]));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputTimestep, timestep_));
  RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputDt, dt_));
  if (has_state_input_) {
    RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputState, state_));
  }
  if (has_embodiment_input_) {
    RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, pi05::kSuffixStepInputEmbodimentId, embodiment_id_));
  }
  RETURN_IF_ERROR(runtime::FindStageInputIndex(suffix_step_plan_, pi05::kSuffixStepInputXT, &suffix_x_t_input_index_));

  suffix_x_t_next_output_index_ = suffix_step_.outputs().size();
  for (std::size_t index = 0; index < suffix_step_.outputs().size(); ++index) {
    if (suffix_step_.outputs()[index].name == pi05::kSuffixStepOutputXTNext) {
      suffix_x_t_next_output_index_ = index;
      break;
    }
  }
  if (suffix_x_t_next_output_index_ == suffix_step_.outputs().size()) {
    return Status::InvalidArgument("suffix_step output contract is missing " + std::string(pi05::kSuffixStepOutputXTNext));
  }
  for (const auto& spec : suffix_step_.inputs()) {
    if (spec.name == pi05::kSuffixStepInputPrefixPadMasks || spec.name == pi05::kSuffixStepInputXT ||
        spec.name == pi05::kSuffixStepInputTimestep || spec.name == pi05::kSuffixStepInputDt ||
        spec.name == pi05::kSuffixStepInputState || spec.name == pi05::kSuffixStepInputEmbodimentId) {
      continue;
    }
    auto cache_it = prefix_lm_workspace_.named_outputs.find(spec.name);
    if (cache_it == prefix_lm_workspace_.named_outputs.end()) {
      return Status::InvalidArgument("missing suffix input: " + spec.name);
    }
    RETURN_IF_ERROR(runtime::BindStageInput(&suffix_step_plan_, spec.name, *cache_it->second));
  }

  load_ms_ = runtime::ElapsedMs(start);
  return Status::Ok();
}

Status Pi05OfflineRunner::RunOnce(const Pi05OfflineRequest& request, Pi05RunResult* result) {
  if (result == nullptr) return Status::InvalidArgument("PI0.5 result is null");
  *result = Pi05RunResult{};
  result->load_ms = load_ms_;

  RETURN_IF_ERROR(ValidatePi05Request(request));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.image, &image_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.image_mask, &image_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.tokenized_prompt, &tokenized_prompt_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.tokenized_prompt_mask, &tokenized_prompt_mask_, stream_.get()));
  RETURN_IF_ERROR(runtime::CopyHostToDevice(request.x_t, &x_t_buffers_[0], stream_.get()));
  if (has_state_input_) {
    if (request.state.data.empty()) {
      return Status::InvalidArgument("PI0.5 suffix engine requires a state tensor");
    }
    RETURN_IF_ERROR(runtime::CopyHostToDevice(request.state, &state_, stream_.get()));
  }
  if (has_embodiment_input_) {
    if (request.embodiment_id.data.empty()) {
      return Status::InvalidArgument("PI0.5 suffix engine requires an embodiment_id tensor");
    }
    RETURN_IF_ERROR(runtime::CopyHostToDevice(request.embodiment_id, &embodiment_id_, stream_.get()));
  }

  auto start = std::chrono::steady_clock::now();
  Status status = RunPi05Stages(prefix_embed_, prefix_lm_, suffix_step_, &x_t_buffers_, &timestep_, &dt_,
                                &prefix_embed_plan_, &prefix_lm_plan_, &suffix_step_plan_, &prefix_embed_workspace_,
                                &prefix_lm_workspace_, &suffix_step_workspace_, &timers_, suffix_x_t_input_index_,
                                suffix_x_t_next_output_index_, stream_.get(), result);
  if (status.ok()) status = stream_.Synchronize();
  result->infer_ms = runtime::ElapsedMs(start);
  return status;
}

double Pi05OfflineRunner::load_ms() const { return load_ms_; }

std::vector<TensorSpec> Pi05OfflineRunner::input_specs() const {
  std::vector<TensorSpec> specs = {
      image_.spec,
      image_mask_.spec,
      tokenized_prompt_.spec,
      tokenized_prompt_mask_.spec,
      x_t_buffers_[0].spec,
  };
  if (has_state_input_) {
    specs.push_back(state_.spec);
  }
  if (has_embodiment_input_) {
    specs.push_back(embodiment_id_.spec);
  }
  return specs;
}

}  // namespace pi_cpp
