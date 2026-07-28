#pragma once

#include <string_view>

namespace pi_cpp::fastwam {

inline constexpr std::string_view kModelName = "fastwam";
inline constexpr std::string_view kDefaultEngineDir = "assets/fastwam_libero/engines";

inline constexpr std::string_view kVaeStage = "vae_image_encoder";
inline constexpr std::string_view kVideoPrefillStage = "video_prefill";
inline constexpr std::string_view kActionStepStage = "action_step_dynamic_kv";

inline constexpr std::string_view kVaeEngine = "vae_image_encoder.engine";
inline constexpr std::string_view kVideoPrefillEngine = "video_prefill.engine";
inline constexpr std::string_view kActionStepEngine = "action_step_dynamic_kv.engine";

inline constexpr std::string_view kInputImage = "input_image";
inline constexpr std::string_view kFirstFrameLatents = "first_frame_latents";
inline constexpr std::string_view kContext = "context";
inline constexpr std::string_view kContextMask = "context_mask";
inline constexpr std::string_view kLatentsAction = "latents_action";
inline constexpr std::string_view kTimestepAction = "timestep_action";
inline constexpr std::string_view kSchedulerDelta = "scheduler_delta";
inline constexpr std::string_view kLatentsActionNext = "latents_action_next";

inline constexpr int kDefaultActionHorizon = 32;
inline constexpr int kDefaultActionDim = 14;
inline constexpr int kDefaultInferenceSteps = 10;
inline constexpr int kVideoLayers = 30;

struct FastWamModelProfile {
  int image_height = 384;
  int image_width = 320;
  int image_channels = 3;
  int text_tokens = 128;
  int context_tokens = 129;
  int context_dim = 4096;
  int action_horizon = kDefaultActionHorizon;
  int action_dim = kDefaultActionDim;
  int video_layers = kVideoLayers;
};

inline constexpr FastWamModelProfile kDefaultModelProfile{};

}  // namespace pi_cpp::fastwam
