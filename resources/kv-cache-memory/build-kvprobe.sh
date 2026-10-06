#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
# Build tools/kvprobe against the pinned llama.cpp worktree's CUDA build (99b95488c).
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
L=${LLAMA_DIR:-$HOME/repo/llama.cpp-kvvideo}
g++ -O2 -std=c++17 "$T/kvprobe.cpp" -o "$T/kvprobe" -I"$L/include" -I"$L/common" -I"$L/ggml/include" \
    -I"$L/vendor" -L"$L/build/bin" -lllama -lllama-common -lggml -lggml-base -Wl,-rpath,"$L/build/bin"
