# MLIR_Educational_Example

This project is a teaching-oriented MLIR pipeline that starts from a small StableHLO-like input and lowers it to LLVM dialect MLIR, then optionally to an executable.

## What is included

- A small custom `myhw` dialect written in C++.
- A manual lowering pass from a StableHLO-lite input subset into `myhw`.
- A second lowering pass from `myhw` into standard MLIR control-flow, arithmetic, and memref operations.
- A Python driver that runs the compiler and can generate a demo executable.
- A documented compilation flow in `docs/COMPILATION_FLOW.md`.

## Important note

The Debian MLIR 19 packages installed on this machine do not include StableHLO libraries, so the front end in this project accepts the sample StableHLO syntax as unregistered ops and lowers the supported subset manually.

## Build

Configure the project from the repository root:

```bash
cmake -S . -B build -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir
cmake --build build -j
```

## Run

Lower the sample model to LLVM dialect MLIR:

```bash
python3 scripts/compile_pipeline.py --input Tiny_NN_Recent.mlir --work-dir build/pipeline
```

Generate the LLVM dialect file and a demo executable:

```bash
python3 scripts/compile_pipeline.py --input Tiny_NN_Recent.mlir --work-dir build/pipeline --emit-executable
```

The generated LLVM dialect MLIR file is written to:

```text
build/pipeline/lowered_to_llvm.mlir
```
