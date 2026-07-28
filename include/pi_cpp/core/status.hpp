#pragma once

#include <string>
#include <utility>

namespace pi_cpp {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kRuntimeError,
  kUnimplemented,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }
  static Status InvalidArgument(std::string message) {
    return {StatusCode::kInvalidArgument, std::move(message)};
  }
  static Status NotFound(std::string message) { return {StatusCode::kNotFound, std::move(message)}; }
  static Status RuntimeError(std::string message) {
    return {StatusCode::kRuntimeError, std::move(message)};
  }
  static Status Unimplemented(std::string message) {
    return {StatusCode::kUnimplemented, std::move(message)};
  }

  [[nodiscard]] bool ok() const { return code_ == StatusCode::kOk; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}  // namespace pi_cpp

#define RETURN_IF_ERROR(expr)            \
  do {                                   \
    ::pi_cpp::Status status__ = (expr);  \
    if (!status__.ok()) return status__; \
  } while (false)
