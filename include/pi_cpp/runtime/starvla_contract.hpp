#pragma once

#include <string_view>

namespace pi_cpp::starvla {

inline constexpr std::string_view kModelName = "starvla";
inline constexpr std::string_view kDefaultEngineDir = "assets/starvla_libero/engines";

inline constexpr std::string_view kPolicyStage = "starvla_policy";
inline constexpr std::string_view kPolicyEngine = "policy.engine";

inline constexpr std::string_view kInputIds = "input_ids";
inline constexpr std::string_view kAttentionMask = "attention_mask";
inline constexpr std::string_view kPositionIds = "position_ids";
inline constexpr std::string_view kVisualSelect = "visual_select";
inline constexpr std::string_view kActionSelect = "action_select";
inline constexpr std::string_view kPixelValues = "pixel_values";
inline constexpr std::string_view kActions = "actions";

}  // namespace pi_cpp::starvla
