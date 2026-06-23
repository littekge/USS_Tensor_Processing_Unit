# Neural Network Assembler - Complete Build Specification
> **Purpose:** This document is the single source of truth for the Neural Network Assembler project. It defines the project identity, current state, architecture, and full incremental build plan.

## Project Identity

- **Name:** Neural Network Assembler
- **Version:** 0.1
- **Development Platform:** Debian 13 Linux
- **Languages:** Python 3.13, MLIR, C/C++
- **License:** MIT License
- **Change Log:** `log.md`
- **Specifications:** — located at `../../Specifications/`
    - **Instruction Set Architecture:** `Functional_TPU_ISA_v0.2.md` — TPU instruction set
    - **Messaging Protocol:** `Functional_TPU_Message_Protocol_v0.1.md` — defines byte format for communicating with the TPU

## General Project Description
This project aims to assemble neural network computation graphs and weights into a formatted binary message according to the `Functional_TPU_Message_Protocol_v0.1.md` and `Functional_TPU_ISA_v0.2.md`. It has two main flows:
- Quantization and mapping of neural network weights to ISA compliant memory locations.
- Assembly of `Functional_TPU_ISA_v0.2.md` machine code from a StableHLO MLIR computation graph.

**Design Goals:**
- Multi-stage lowering and assembly pipeline.
- Obvious intermediate representations of computation graph.
- Custom *Functional TPU* MLIR dialect.
- f32 to int8 quantization of neural network weights.

## Lowering Pipeline

The lowering pipeline to the `Functional_TPU_Message_Protocol_v0.1.md` format has 4 stages:
1. Neural Network Import
    1. import MLIR representation as `/tmp/initial.mlir`.
    2. import neural network weights as `/tmp/weighs.npz`.
2. Weight Processing
    1. Read weights from `/tmp/weights.npz`.
    2. Quantize neural network weights to int8 from f32.
    3. Algorithmically map weights to ISA compliant memory locations.
    4. Convert mapped weights to the `Functional_TPU_Message_Protocol_v0.1.md` MEM format and save as `/tmp/MEM.bin`.
    5. Save weight addresses as `/tmp/weight_map.json` — should correspond to the input arguments in `/tmp/initial.mlir`. This should be easy since `/tmp/weighs.npz` is already ordered and labeled in accordance with `/tmp/initial.mlir`.
3. MLIR lowering
    1. Run StableHLO default optimization pass on `/tmp/initial.mlir`, saving the result as `/tmp/optimized.mlir`.
    2. Legalize `/tmp/optimized.mlir` to the *Functional_TPU* MLIR dialect, saving the result as `/tmp/optimized.tpu.mlir`.
    3. Assemble `/tmp/optimized.tpu.mlir` into `Functional_TPU_ISA_v0.2.md` machine code referencing `/tmp/weight_map.json` to determine weight addresses.
    4. Convert machine code to the `Functional_TPU_Message_Protocol_v0.1.md` PROGRAM format and save as `/tmp/PROGRAM.bin`.
4. Final Conversion
    1. Concatenate `/tmp/PROGRAM.bin` and `/tmp/MEM.bin` into a single string of binary
    2. Append START and END function codes to create a full transmission.
    3. Save the completed binary to `/out/TRANSMISSION.bin`.

## Current State (v0.1 — What's Working)

- **Main Guard:** The project can be run using `python ./src/Convert.py` with relevant arguments, or by directly calling the `Convert(model_name, run_name)` function.
- **Neural Network Imports:** The `NN_Import` function in `/src/Convert.py` imports the 