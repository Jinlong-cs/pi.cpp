#!/usr/bin/env bash
set -euo pipefail

cd "${PI_CPP_HOME:-/workspaces/pi.cpp}"

case " $* " in
  *" --model pi05 "*)
    echo "[pi.cpp] ensuring PI0.5 TensorRT engines via Makefile"
    make engines_pi05
    ;;
  *" --model fastwam "*)
    echo "[pi.cpp] ensuring FastWAM LIBERO TensorRT engines via Makefile"
    make engines_fastwam
    ;;
  *" --model semanticvla "*)
    echo "[pi.cpp] ensuring SemanticVLA LIBERO TensorRT engines via Makefile"
    make engines_semanticvla
    ;;
  *" --model smolvla "*)
    echo "[pi.cpp] ensuring SmolVLA LIBERO TensorRT engines via Makefile"
    make engines_smolvla
    ;;
esac

exec "$@"
