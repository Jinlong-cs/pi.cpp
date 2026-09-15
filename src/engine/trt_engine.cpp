#include "pi_cpp/core/trt_engine.hpp"

#include <utility>

namespace pi_cpp {

TrtEngine::TrtEngine(std::filesystem::path path) : path_(std::move(path)) {}

TrtEngine::~TrtEngine() = default;

TrtEngine::TrtEngine(TrtEngine&&) noexcept = default;

TrtEngine& TrtEngine::operator=(TrtEngine&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  // Release the current TRT objects in TensorRT's destruction order
  // (context, then engine, then runtime); the defaulted member-wise move
  // assignment would delete the runtime first while its engine is alive.
  context_.reset();
  engine_.reset();
  runtime_.reset();
  path_ = std::move(other.path_);
  inputs_ = std::move(other.inputs_);
  outputs_ = std::move(other.outputs_);
  loaded_ = other.loaded_;
  other.loaded_ = false;
  logger_ = std::move(other.logger_);
  runtime_ = std::move(other.runtime_);
  engine_ = std::move(other.engine_);
  context_ = std::move(other.context_);
  bound_inputs_ = std::move(other.bound_inputs_);
  bound_outputs_ = std::move(other.bound_outputs_);
  return *this;
}

}  // namespace pi_cpp

