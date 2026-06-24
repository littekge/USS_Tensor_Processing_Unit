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
- Obvious intermediate representations of the computation graph.
- Custom *Functional TPU* MLIR dialect.
- f32 to Q0.7 fixed point quantization of neural network weights.
- Memory mapping algorithms.
- MLIR pass to decompose large matrix multiplications into a series of smaller matrix multiplication and addition instructions based on an easily accessible parameter `MAX_MATMUL_SIZE`.

## Lowering Pipeline

The lowering pipeline to the `Functional_TPU_Message_Protocol_v0.1.md` format has 4 stages:
1. Neural Network Import
    1. import MLIR representation as `/tmp/initial.mlir`.
    2. import neural network weights as `/tmp/weights.npz`.
2. Weight Processing
    1. Read weights from `/tmp/weights.npz`.
    2. Quantize neural network weights from f32 to Q0.7 fixed point (1 sign bit + 0 integer bits + 7 fractional bits = 8 bits total).
    3. Algorithmically map weights to ISA compliant memory locations.
    4. Convert mapped weights to the `Functional_TPU_Message_Protocol_v0.1.md` MEM format and save as `/tmp/MEM.bin`.
    5. Save weight addresses as `/tmp/weight_map.json` — should correspond to the input arguments in `/tmp/initial.mlir`. This should be easy since `/tmp/weights.npz` is already ordered and labeled in accordance with `/tmp/initial.mlir`.
3. MLIR Lowering
    1. Run StableHLO default optimization pass on `/tmp/initial.mlir`, saving the result as `/tmp/optimized.mlir`.
    2. Legalize `/tmp/optimized.mlir` to the *Functional_TPU* MLIR dialect, saving the result as `/tmp/optimized.tpu.mlir`.
    3. Assemble `/tmp/optimized.tpu.mlir` into `Functional_TPU_ISA_v0.2.md` machine code referencing `/tmp/weight_map.json` to determine weight addresses.
    4. Convert machine code to the `Functional_TPU_Message_Protocol_v0.1.md` PROGRAM format and save as `/tmp/PROGRAM.bin`.
4. Final Conversion
    1. Concatenate `/tmp/MEM.bin` and `/tmp/PROGRAM.bin` into a single string of binary
    2. Append START and STOP function codes to create a full transmission.
    3. Save the completed binary to `/out/TRANSMISSION.bin`.

## Current State (v0.1 — What's Working)

v0.1 is **complete**: the full pipeline runs end-to-end and produces `/out/TRANSMISSION.bin`.

The project is an installable Python package named **`nn_assembler`** (the `src/`
directory is mapped to that import name via `pyproject.toml`). After a one-time
`pip install -e .` from the project root, it can be used any of these ways:

- From another file: `from nn_assembler import Convert; Convert("Tiny_NN", "Recent")`
- As a module: `python -m nn_assembler Tiny_NN Recent`
- As a console script: `nn-assemble Tiny_NN Recent`
- As the original script (still supported): `python ./src/Convert.py Tiny_NN Recent`

- **Main Guard:** The project can be run using `python ./src/Convert.py` with relevant arguments, or by directly calling the `Convert(model_name, run_name)` function.
- **Neural Network Imports:** The `NN_Import` function in `/src/Convert.py` imports neural networks based on *model_name* and *run_name* and saves them to `/tmp/initial.mlir` (StableHLO computation graph) and `/tmp/weights.npz` (neural network weights). The `Neural_Networks` directory is located by an upward search from the package (overridable via the `NN_ASSEMBLER_NETWORKS_DIR` env var or the `nn_dir` argument), so the project is not coupled to a fixed directory depth.
- **First Optimization Pass:** `/src/Process_MLIR.py` runs a base optimization pass provided by the StableHLO-opt binary, saving the result to `/tmp/optimized.mlir`.
- **Weight Processing:** `/src/Process_Weights.py` quantizes weights f32 → Q0.7, maps them to addresses (input → `0x1`, weights contiguous from `0x2`), and writes `/tmp/MEM.bin` and `/tmp/weight_map.json`.
- **Custom Dialect + Legalization:** `/src/MLIR/` defines the *Functional TPU* dialect (Python implementation — see `/src/MLIR/README.md`) and the StableHLO → TPU legalization pass; `Process_MLIR` writes `/tmp/optimized.tpu.mlir`. Shape-only ops (reshape) are folded so each dialect op maps 1:1 to an instruction.
- **Assembler:** `/src/Assembler.py` translates the dialect to ISA machine code and exports `/tmp/PROGRAM.bin`. Intermediate results and the network input share scratch address `0x1`.
- **Serializer:** `/src/Serializer.py` wraps `MEM.bin` then `PROGRAM.bin` with START/STOP into `/out/TRANSMISSION.bin`.
- **Protocol encoders:** `/src/Protocol.py` centralizes the Message Protocol byte format (function codes, MEM/PROGRAM block builders).
- **Tests:** 15 pytest tests in `/test/` cover quantization, weight mapping, the dialect, legalization, instruction encoding, and the full pipeline.

## Architecture

- `pyproject.toml` — packaging metadata; maps the `src/` directory to the importable `nn_assembler` package and defines the `nn-assemble` console script.
- `/examples/` — example code for claude to reference
- `/out/` — Location of final output binary
- `/src/` — Source files (the `nn_assembler` package)
    - `/src/__init__.py` — package entry; re-exports `Convert`, `NN_import`, `main`
    - `/src/__main__.py` — enables `python -m nn_assembler`
    - `/src/Protocol.py` — Message Protocol byte encoders (function codes, MEM/PROGRAM blocks)
    - `/src/MLIR/` — Custom MLIR dialect and passes, should be mostly self-contained
    - `/src/Convert.py` — Top level file; entry point for program, binds logic from other files together. Also implements step 1 of lowering (*Neural Network Import*)
    - `/src/Process_Weights.py` — implements step 2 of lowering (*Weight Processing*)
    - `/src/Process_MLIR.py` —  implements the first half of step 3 of lowering (*MLIR Lowering*); this file accesses the custom dialect and passes in `/src/MLIR/`
    - `/src/Assembler.py` — assembles the *Functional TPU* custom MLIR dialect into machine code, implementing the second half of lowering step 3.
    - `/src/Serializer.py` — implements the 4th lowering step (*Final Conversion*).
- `/test/` — All tests used to verify program functionality
- `/tmp/` — Temporary program files

## Build Plan
The build plan for this project is *versioned*. Early versions will **NOT** implement the final design goals of the project; simpler logic will be tested and verified to work before more complex behavior is added. Build steps for future versions will be added incrementally as needed, and the design goals/lowering pipeline may be updated to reflect new versions.

### v0.1 — Basic Functionality ✅ COMPLETE
Implements basic functionality of the program. Does **NOT** implement complex MLIR passes (no matmul partitioning), only the legalization pass from StableHLO to the *Functional TPU* custom dialect.

> All steps below are complete. The custom dialect (Step 4) is implemented in Python rather than TableGen/C++ — `main.md` Step 4 grants free reign over the language; see `/src/MLIR/README.md` for the rationale and the dialect grammar.

#### Step 1: Weight Quantization ✅
Build logic to...
- Correctly read `/tmp/weights.npz` into *numpy* arrays/tensors.
- Quantize weights into Q0.7 fixed point from f32, maintaining as much accuracy as possible.

#### Step 2: Weight Mapping ✅
Reference the `Functional_TPU_ISA_v0.2.md` and build logic to...
- Map the neural network input to address 0x1.
- Map the remaining weights to valid memory addresses.

#### Step 3: Weight Formatting ✅
Reference the `Functional_TPU_Message_Protocol_v0.1.md` and build logic to...
- Export the memory map in the *Functional TPU Message Protocol* format and save it as `/tmp/MEM.bin`.
- Export the weight addresses as `/tmp/weight_map.json`.

#### Step 4: Custom MLIR Dialect ✅
Build a custom MLIR dialect in `/src/MLIR/`. The formatting of the code and files in that directory are entirely up to you; you may choose the coding language (likely tablegen and c++ to conform to what MLIR expects). However, the dialect must follow a few key rules:
- **Direct translation** to the `Functional_TPU_ISA_v0.2.md`. The custom dialect must have a nearly 1:1 relation to machine code, and I should be able to easily determine the layout of the machine code from the custom dialect.
- **Minimize** functions that will be synthesized away during assembly.
The goal of the custom dialect is to be a bridge between StableHLO and machine code. If someone writes a computation graph in our custom dialect, our assembler will be able to translate it to machine code.

#### Step 5: Legalization Pass ✅
Build a legalization pass to convert StableHLO to our custom dialect.
- The legalization does not need to be complete, this version of the pass only needs to consider the operations present in `/examples/v0.1_example.mlir` and `/examples/v0.1_optimized_example.mlir`.
- The logic that actually runs the pass should be in `/src/Process_MLIR.py`, but the pass should be defined in `/src/MLIR/`. This legalization pass is the second pass, and occurs after the first optimization pass.
- Save the result of the legalization pass to `/tmp/optimized.tpu.mlir`.

#### Step 6: Assembler ✅
Build `/src/Assembler.py`:
- Reference the weight map json file to convert the output of the final MLIR pass to `Functional_TPU_ISA_v0.2.md` machine code.
- Prioritize using address 0x1 for intermediate results.

#### Step 7: Assembler Export ✅
Reference the `Functional_TPU_Message_Protocol_v0.1.md` and build logic to...
- Export the machine instructions to the *Functional TPU Message Protocol* format and save it as `/tmp/PROGRAM.bin`

#### Step 8: Serializer ✅
Build `/src/Serializer.py`:
- Concatenate `/tmp/MEM.bin` and `/tmp/PROGRAM.bin` into a single binary string.
- Append START and STOP codes to generate a full *Functional TPU Message Protocol* message.
- Save the final result to `/out/TRANSMISSION.bin`.