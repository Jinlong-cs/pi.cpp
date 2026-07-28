#pragma once

#include <iostream>
#include <string_view>

namespace pi_cpp {

class TrtLogger {
 public:
  enum class Severity { kInternalError, kError, kWarning, kInfo, kVerbose };

  void Log(Severity severity, std::string_view message) const {
    if (severity == Severity::kVerbose) return;
    std::cerr << "[TensorRT] " << message << '\n';
  }
};

}  // namespace pi_cpp
