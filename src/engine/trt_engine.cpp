#include "pi_cpp/core/trt_engine.hpp"

#include <utility>

namespace pi_cpp {

TrtEngine::TrtEngine(std::filesystem::path path) : path_(std::move(path)) {}

TrtEngine::~TrtEngine() = default;

TrtEngine::TrtEngine(TrtEngine&&) noexcept = default;

TrtEngine& TrtEngine::operator=(TrtEngine&&) noexcept = default;

}  // namespace pi_cpp

