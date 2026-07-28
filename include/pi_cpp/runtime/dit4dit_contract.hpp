#pragma once

#include <string_view>

namespace pi_cpp::dit4dit {

inline constexpr std::string_view kModelName = "dit4dit";
inline constexpr std::string_view kDefaultEngineDir = "assets/dit4dit_libero/engines";

inline constexpr std::string_view kVaeStage = "cosmos_vae_encode";
inline constexpr std::string_view kFeatureStage = "cosmos_feature";
inline constexpr std::string_view kActionLoopStage = "action_loop";

inline constexpr std::string_view kVaeEngine = "cosmos_vae_encode.engine";
inline constexpr std::string_view kFeatureEngine = "cosmos_feature.engine";
inline constexpr std::string_view kActionLoopEngine = "action_loop.engine";

inline constexpr std::string_view kVideo = "video_bcthw";
inline constexpr std::string_view kSampleNoise = "sample_noise";
inline constexpr std::string_view kCondLatents = "cond_latents";
inline constexpr std::string_view kLatents = "latents";
inline constexpr std::string_view kCondMask = "cond_mask";
inline constexpr std::string_view kCondIndicator = "cond_indicator";
inline constexpr std::string_view kPromptEmbeds = "prompt_embeds";
inline constexpr std::string_view kPaddingMask = "padding_mask";
inline constexpr std::string_view kSigmaT = "sigma_t";
inline constexpr std::string_view kVlEmbs = "vl_embs";
inline constexpr std::string_view kState = "state";
inline constexpr std::string_view kInitialActions = "initial_actions";
inline constexpr std::string_view kNormalizedActions = "normalized_actions";

inline constexpr int kCameras = 2;
inline constexpr int kViewHeight = 224;
inline constexpr int kViewWidth = 224;
inline constexpr int kVideoFrames = 5;
inline constexpr int kStateDim = 16;
inline constexpr int kActionHorizon = 8;
inline constexpr int kActionDim = 7;
inline constexpr int kActionLatentDim = 8;

}  // namespace pi_cpp::dit4dit
