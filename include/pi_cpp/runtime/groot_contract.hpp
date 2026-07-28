#pragma once

#include <string_view>

namespace pi_cpp::groot {

inline constexpr std::string_view kModelName = "groot";
inline constexpr std::string_view kDefaultEngineDir = "assets/groot_n1d6/engines";

inline constexpr std::string_view kBackboneStage = "groot_backbone";
inline constexpr std::string_view kActionLoopStage = "groot_action_loop";

inline constexpr std::string_view kBackboneEngine = "backbone.engine";
inline constexpr std::string_view kActionLoopEngine = "action_loop.engine";

inline constexpr std::string_view kInputIds = "input_ids";
inline constexpr std::string_view kAttentionMask = "attention_mask";
inline constexpr std::string_view kPixelValues0 = "pixel_values_0";
inline constexpr std::string_view kPixelValues1 = "pixel_values_1";
inline constexpr std::string_view kBackboneFeatures = "backbone_features";
inline constexpr std::string_view kBackboneAttentionMask = "backbone_attention_mask";
inline constexpr std::string_view kImageMask = "image_mask";
inline constexpr std::string_view kState = "state";
inline constexpr std::string_view kEmbodimentId = "embodiment_id";
inline constexpr std::string_view kInitialActions = "initial_actions";
inline constexpr std::string_view kNormalizedActions = "normalized_actions";

}  // namespace pi_cpp::groot
