#pragma once

#include <string_view>

namespace pi_cpp::pi05 {

inline constexpr std::string_view kModelName = "pi05";

inline constexpr std::string_view kPrefixEmbedStage = "prefix_embed";
inline constexpr std::string_view kPrefixLmStage = "prefix_lm";
inline constexpr std::string_view kSuffixStepStage = "suffix_step";
inline constexpr std::string_view kSuffixLoopStage = "suffix_loop";

inline constexpr std::string_view kPrefixEmbedEngine = "prefix_embed.engine";
inline constexpr std::string_view kPrefixLmEngine = "prefix_lm.engine";
inline constexpr std::string_view kSuffixStepEngine = "suffix_step.engine";

inline constexpr std::string_view kPrefixEmbedInputImage = "image";
inline constexpr std::string_view kPrefixEmbedInputImageMask = "image_mask";
inline constexpr std::string_view kPrefixEmbedInputTokenizedPrompt = "tokenized_prompt";
inline constexpr std::string_view kPrefixEmbedInputTokenizedPromptMask = "tokenized_prompt_mask";

inline constexpr std::string_view kPrefixEmbedOutputPrefixEmbs = "prefix_embs";
inline constexpr std::string_view kPrefixEmbedOutputPrefixPositionIds = "prefix_position_ids";
inline constexpr std::string_view kPrefixEmbedOutputPrefixAttentionMask4d = "prefix_attention_mask_4d";
inline constexpr std::string_view kPrefixEmbedOutputPrefixPadMasks = "prefix_pad_masks";

inline constexpr std::string_view kPrefixLmInputPrefixEmbs = "prefix_embs";
inline constexpr std::string_view kPrefixLmInputPrefixPositionIds = "prefix_position_ids";
inline constexpr std::string_view kPrefixLmInputPrefixAttentionMask4d = "prefix_attention_mask_4d";

inline constexpr std::string_view kSuffixStepInputPrefixPadMasks = "prefix_pad_masks";
inline constexpr std::string_view kSuffixStepInputXT = "x_t";
inline constexpr std::string_view kSuffixStepInputTimestep = "timestep";
inline constexpr std::string_view kSuffixStepInputDt = "dt";
inline constexpr std::string_view kSuffixStepOutputXTNext = "x_t_next";

inline constexpr int kDefaultActionDim = 14;
inline constexpr int kDefaultDenoiseSteps = 10;
inline constexpr float kDefaultDt = -0.1F;

}  // namespace pi_cpp::pi05
