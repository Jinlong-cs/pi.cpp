#pragma once

#include <string_view>

namespace pi_cpp::evo1 {

inline constexpr std::string_view kModelName = "evo1";
inline constexpr std::string_view kDefaultEngineDir = "assets/evo1_libero/engines";

inline constexpr std::string_view kBackboneStage = "evo1_backbone";
inline constexpr std::string_view kActionStepStage = "evo1_action_step";

inline constexpr std::string_view kBackboneEngine = "backbone.engine";
inline constexpr std::string_view kActionStepEngine = "action_step.engine";

inline constexpr std::string_view kInputIds = "input_ids";
inline constexpr std::string_view kAttentionMask = "attention_mask";
inline constexpr std::string_view kAttentionMask2d = "attention_mask_2d";
inline constexpr std::string_view kPositionIds = "position_ids";
inline constexpr std::string_view kPixelValues = "pixel_values";
inline constexpr std::string_view kState = "state";
inline constexpr std::string_view kContextTokens = "context_tokens";

inline constexpr std::string_view kActions = "actions";
inline constexpr std::string_view kTimestep = "timestep";
inline constexpr std::string_view kActionsNext = "actions_next";

inline constexpr int kInferenceSteps = 32;
inline constexpr int kTimestepBuckets = 1000;

}  // namespace pi_cpp::evo1
