#!/usr/bin/env bash
set -euo pipefail

python3 scripts/compile_pipeline.py \
  --input Tiny_NN_Recent.mlir \
  --work-dir build/test-smoke \
  --emit-executable
