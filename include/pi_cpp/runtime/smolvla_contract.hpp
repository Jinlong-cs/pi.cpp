#pragma once

#include <string_view>

namespace pi_cpp::smolvla {

inline constexpr std::string_view kModelName = "smolvla";
inline constexpr std::string_view kDefaultEngineDir = "assets/smolvla_libero/engines";

inline constexpr std::string_view kPrefixEmbedStage = "prefix_embed";
inline constexpr std::string_view kPrefixLmStage = "prefix_lm";
inline constexpr std::string_view kSuffixLoopStage = "suffix_loop";

inline constexpr std::string_view kPrefixEmbedEngine = "prefix_embed.engine";
inline constexpr std::string_view kPrefixLmEngine = "prefix_lm.engine";
inline constexpr std::string_view kSuffixLoopEngine = "suffix_loop.engine";

inline constexpr std::string_view kImage = "image";
inline constexpr std::string_view kImageMask = "image_mask";
inline constexpr std::string_view kTokenizedPrompt = "tokenized_prompt";
inline constexpr std::string_view kTokenizedPromptMask = "tokenized_prompt_mask";
inline constexpr std::string_view kState = "state";

inline constexpr std::string_view kPrefixEmbs = "prefix_embs";
inline constexpr std::string_view kPrefixPadMasks = "prefix_pad_masks";
inline constexpr std::string_view kPrefixAttMasks = "prefix_att_masks";

inline constexpr std::string_view kSuffixLoopInputXT = "x_t";
inline constexpr std::string_view kSuffixLoopOutputXT = "x_t_out";

inline constexpr int kDefaultActionDim = 7;
inline constexpr int kDefaultDenoiseSteps = 10;

struct SmolVlaModelProfile {
  int cameras = 2;
  int image_height = 512;
  int image_width = 512;
  int text_tokens = 48;
  int prefix_tokens = 177;
  int hidden = 960;
  int action_horizon = 50;
  int max_action_dim = 32;
  int action_dim = kDefaultActionDim;
  int layers = 16;
};

inline constexpr SmolVlaModelProfile kDefaultModelProfile{};

}  // namespace pi_cpp::smolvla
