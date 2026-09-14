#pragma once

// Back-compat surface: the split-denoise runner used to live under the
// pi05 name. The generic implementation is split_denoise_runner.hpp;
// these aliases preserve the historical class/struct names for the
// pybind layer and any external callers.

#include "pi_cpp/runtime/split_denoise_runner.hpp"

namespace pi_cpp {

using Pi05StageTimers = SplitDenoiseStageTimers;
using Pi05OfflineRequest = SplitDenoiseRequest;
using Pi05RunResult = SplitDenoiseResult;
using Pi05OfflineRunner = SplitDenoiseRunner;

}  // namespace pi_cpp
