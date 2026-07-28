#pragma once

#include <string_view>

namespace pi_cpp::semanticvla {

inline constexpr std::string_view kModelName = "semanticvla";
inline constexpr std::string_view kDefaultEngineDir = "assets/semanticvla_libero/engines";

inline constexpr std::string_view kBackboneStage = "semanticvla_backbone";
inline constexpr std::string_view kActionStepStage = "semanticvla_action_step";

inline constexpr std::string_view kBackboneEngine = "backbone.engine";
inline constexpr std::string_view kActionStepEngine = "action_step.engine";

inline constexpr std::string_view kInputIds = "input_ids";
inline constexpr std::string_view kAttentionMask = "attention_mask";
inline constexpr std::string_view kPositionIds = "position_ids";
inline constexpr std::string_view kVisualSelect = "visual_select";
inline constexpr std::string_view kPixelValues = "pixel_values";
inline constexpr std::string_view kLastHidden = "last_hidden";

inline constexpr std::string_view kActions = "actions";
inline constexpr std::string_view kTimestep = "timestep";
inline constexpr std::string_view kActionsNext = "actions_next";

inline constexpr int kDefaultInferenceSteps = 10;
inline constexpr int kDefaultTimestepBuckets = 1000;

}  // namespace pi_cpp::semanticvla
