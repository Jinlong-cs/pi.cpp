#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pi_cpp/core/status.hpp"

namespace pi_cpp {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes), storage_(bytes) {}

  [[nodiscard]] void* data() { return storage_.data(); }
  [[nodiscard]] const void* data() const { return storage_.data(); }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

  Status Resize(std::size_t bytes) {
    bytes_ = bytes;
    storage_.resize(bytes);
    return Status::Ok();
  }

 private:
  // Placeholder storage for the initial skeleton. Replace with cudaMalloc/cudaFree
  // once CUDA is wired into the build.
  std::size_t bytes_ = 0;
  std::vector<std::uint8_t> storage_;
};

}  // namespace pi_cpp
