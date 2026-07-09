# Neural Network Assembler - Complete Build Specification
>
> **Purpose:** This document is the single source of truth for the Neural
> Network Assembler project. It defines the project identity, current state,
> architecture, and full incremental build plan.

## Project Identity

- **Name:** Neural Network Assembler
- **Version:** 0.3.0
- **Development Platform:** Debian 13 Linux
- **Languages:** Python 3.13, MLIR, C/C++
- **License:** MIT License
- **Change Log:** `log.md`
- **Specifications:** — located at `../../Specifications/`
  - **Instruction Set Architecture:** `Functional_TPU_ISA.md` — TPU
  instruction set
  - **Messaging Protocol:** `Functional_TPU_Message_Protocol.md` — defines
  byte format for communicating with the TPU

## General Project Description

This project aims to assemble neural network computation graphs and weights into
a formatted binary message according to the
`Functional_TPU_Message_Protocol.md` and `Functional_TPU_ISA.md`. It
has two main flows:

- Quantization and mapping of neural network weights to ISA compliant memory locations.
- Assembly of `Functional_TPU_ISA.md` machine code from a StableHLO MLIR
computation graph.

**Design Goals:**

- Multi-stage lowering and assembly pipeline.
- Obvious intermediate representations of the computation graph.
- Custom *Functional TPU* MLIR dialect.
- Per-tensor symmetric int8 quantization of neural network weights, with each
layer's requantization multiplier decomposed into the ISA dyadic (`M0 · 2^-n`)
form.
- Memory mapping algorithms.
- MLIR pass to decompose large matrix multiplications into a series of smaller
matrix multiplication and addition instructions based on an easily accessible
parameter `MAX_MATMUL_SIZE`.
- Data-memory allocation: reserve `0x1` for network I/O staging and place all
intermediate results in main data memory (`0x2+`) with address reuse.

## Lowering Pipeline

The lowering pipeline to the `Functional_TPU_Message_Protocol.md` format
has 4 stages:

1. Neural Network Import
    1. import MLIR representation as `/tmp/initial.mlir`.
    2. import neural network weights as `/tmp/weights.npz`.
2. Weight Processing
    1. Read weights and the per-layer requantization data (`__M__*` /
    `__scales__*`) from `/tmp/weights.npz`.
    2. Quantize neural network weights from f32 to per-tensor symmetric int8
    (`w_q = round(w / S_w)`, `S_w = absmax(W)/127`, clamped to [-127, 127]),
    using the per-tensor `S_w` exported in `__scales__` so the quantized weights
    stay consistent with the exported requantization multiplier `M`.
    3. Decompose each layer's `M` into the ISA dyadic form `M0 · 2^-n` (see the
    v0.3 build plan).
    4. Algorithmically map weights (and the now-unused bias tensors) to ISA
    compliant memory locations.
    5. Convert mapped weights to the `Functional_TPU_Message_Protocol.md`
    MEM format and save as `/tmp/MEM.bin`.
    6. Save weight addresses and per-layer `M0`/`n` as `/tmp/weight_map.json` —
    should correspond to the input arguments in `/tmp/initial.mlir`. This should
    be easy since `/tmp/weights.npz` is already ordered and labeled in accordance
    with `/tmp/initial.mlir`.
3. MLIR Lowering
    1. Run StableHLO default optimization pass on `/tmp/initial.mlir`, saving
    the result as `/tmp/optimized.mlir`.
    2. Legalize `/tmp/optimized.mlir` to the *Functional_TPU* MLIR dialect,
    annotating each `mult` op with its layer's `M0`/`n` from
    `/tmp/weight_map.json`, and saving the result as `/tmp/optimized.tpu.mlir`.
    3. Run the bias-removal pass to drop bias-add operations and reroute their
    consumers to the matmul result, saving the result as
    `/tmp/optimized.nobias.tpu.mlir`. Bias tensors remain in weight memory but
    are no longer referenced by any instruction.
    4. Assemble the final dialect into `Functional_TPU_ISA.md` machine code,
    referencing `/tmp/weight_map.json` for weight addresses and `M0`/`n`.
    5. Convert machine code to the `Functional_TPU_Message_Protocol.md`
    PROGRAM format and save as `/tmp/PROGRAM.bin`.
4. Final Conversion
    1. Concatenate `/tmp/MEM.bin` and `/tmp/PROGRAM.bin` into a single string of
    binary
    2. Append START and STOP function codes to create a full transmission.
    3. Save the completed binary to `/out/TRANSMISSION.bin`.

## Current State (v0.1 — What's Working)

v0.1 is **complete**: the full pipeline runs end-to-end and produces `/out/TRANSMISSION.bin`.

The project is an installable Python package named **`nn_assembler`** (the
`nn_assembler/` directory is the package). After a one-time editable install
from the project root it can be used any of these ways:

```python
pip install -e . --config-settings editable_mode=compat
```

(The `editable_mode=compat` flag makes setuptools write a plain static path into
the `.pth` file instead of its default dynamic import hook, so editors / static
analyzers such as Pylance can resolve `nn_assembler` imports. It adds no files to
the project tree. A plain `pip install -e .` still works at runtime but leaves the
package unresolved in the IDE.)

- From another file: `from nn_assembler import Convert; Convert("Tiny_NN", "Recent")`
- As a module: `python -m nn_assembler Tiny_NN Recent`
- As a console script: `nn-assemble Tiny_NN Recent`
- As the original script (still supported):
`python ./nn_assembler/Convert.py Tiny_NN Recent`

- **Main Guard:** The project can be run using
`python ./nn_assembler/Convert.py` with relevant arguments, or by directly
calling the `Convert(model_name, run_name)` function.
- **Neural Network Imports:** The `NN_Import` function in
`/nn_assembler/Convert.py` imports neural networks based on *model_name* and
*run_name* and saves them to `/tmp/initial.mlir` (StableHLO computation graph)
and `/tmp/weights.npz` (neural network weights). The `Neural_Networks` directory
is located by an upward search from the package (overridable via the
`NN_ASSEMBLER_NETWORKS_DIR` env var or the `nn_dir` argument), so the project is
not coupled to a fixed directory depth.
- **First Optimization Pass:** `/nn_assembler/Process_MLIR.py` runs a base
optimization pass provided by the StableHLO-opt binary, saving the result to `/tmp/optimized.mlir`.
- **Weight Processing:** `/nn_assembler/Process_Weights.py` quantizes weights
f32 → per-tensor symmetric int8 (v0.3; `S_w` from `__scales__`), decomposes each
layer's `M` into the dyadic pair `M0 · 2^-n`, maps tensors to addresses (input →
`0x1`, weights/biases contiguous from `0x2`), and writes `/tmp/MEM.bin` and
`/tmp/weight_map.json` (weight entries carry `M0`/`n`).
- **Custom Dialect + Legalization:** `/nn_assembler/MLIR/` defines the
*Functional TPU* dialect (Python implementation — see
`/nn_assembler/MLIR/README.md`) and the StableHLO → TPU legalization pass, which
annotates each `tpu.mult` with its `M0`/`n`; `Process_MLIR` writes
`/tmp/optimized.tpu.mlir`. Shape-only ops (reshape) are folded so each dialect op
maps 1:1 to an instruction.
- **Bias Removal (v0.3):** `/nn_assembler/MLIR/bias_removal.py` drops bias `add`
ops and reroutes consumers to the matmul result; `Process_MLIR` writes
`/tmp/optimized.nobias.tpu.mlir`, the final dialect the assembler consumes.
- **Assembler:** `/nn_assembler/Assembler.py` translates the (bias-removed)
dialect to ISA machine code — encoding `M0` (bits 59-52) and `n` (bits 51-44) in
each MUL — and exports `/tmp/PROGRAM.bin`. Intermediate results and the network
input share scratch address `0x1`.
- **Serializer:** `/nn_assembler/Serializer.py` wraps `MEM.bin` then
`PROGRAM.bin` with FLASH/STOP into `/out/TRANSMISSION.bin`.
- **Protocol encoders:** `/nn_assembler/Protocol.py` centralizes the Message
Protocol byte format (function codes, MEM/PROGRAM block builders).
- **Tests:** 26 pytest tests in `/test/` cover per-tensor int8 quantization,
dyadic `M → (M0, n)` decomposition, weight mapping, the dialect (incl. `M0`/`n`
round-trip), legalization + annotation, bias removal, instruction encoding, and
the full pipeline.

## Architecture

- `pyproject.toml` — packaging metadata; declares the `nn_assembler` package
and defines the `nn-assemble` console script.
- `/examples/` — example code for claude to reference
- `/out/` — Location of final output binary
- `/nn_assembler/` — Source files (the `nn_assembler` package)
  - `/nn_assembler/__init__.py` — package entry; re-exports `Convert`,
  `NN_import`, `main`
  - `/nn_assembler/__main__.py` — enables `python -m nn_assembler`
  - `/nn_assembler/Protocol.py` — Message Protocol byte encoders (function
  codes, MEM/PROGRAM blocks)
  - `/nn_assembler/MLIR/` — Custom MLIR dialect and passes, should be mostly self-contained
  - `/nn_assembler/Convert.py` — Top level file; entry point for program,
  binds logic from other files together. Also implements step 1 of lowering
  (*Neural Network Import*)
  - `/nn_assembler/Process_Weights.py` — implements step 2 of lowering
  (*Weight Processing*)
  - `/nn_assembler/Process_MLIR.py` —  implements the first half of step 3 of
  lowering (*MLIR Lowering*); this file accesses the custom dialect and passes
  in `/nn_assembler/MLIR/`
  - `/nn_assembler/Assembler.py` — assembles the *Functional TPU* custom MLIR
  dialect into machine code, implementing the second half of lowering step 3.
  - `/nn_assembler/Serializer.py` — implements the 4th lowering step (*Final Conversion*).
- `/test/` — All tests used to verify program functionality
- `/tmp/` — Temporary program files

## Build Plan

The build plan for this project is *versioned*. Early versions will **NOT**
implement the final design goals of the project; simpler logic will be tested
and verified to work before more complex behavior is added. Build steps for
future versions will be added incrementally as needed, and the design
goals/lowering pipeline may be updated to reflect new versions.

### v0.1 — Basic Functionality ✅ COMPLETE

Implements basic functionality of the program. Does **NOT** implement complex
MLIR passes (no matmul partitioning), only the legalization pass from StableHLO
to the *Functional TPU* custom dialect.

> All steps below are complete. The custom dialect (Step 4) is implemented in
> Python rather than TableGen/C++ — `main.md` Step 4 grants free reign over the
> language; see `/nn_assembler/MLIR/README.md` for the rationale and the dialect
> grammar.

#### Step 1: Weight Quantization ✅

Build logic to...

- Correctly read `/tmp/weights.npz` into *numpy* arrays/tensors.
- Quantize weights into Q0.7 fixed point from f32, maintaining as much accuracy
as possible.

#### Step 2: Weight Mapping ✅

Reference the `Functional_TPU_ISA.md` and build logic to...

- Map the neural network input to address 0x1.
- Map the remaining weights to valid memory addresses.

#### Step 3: Weight Formatting ✅

Reference the `Functional_TPU_Message_Protocol.md` and build logic to...

- Export the memory map in the *Functional TPU Message Protocol* format and save
it as `/tmp/MEM.bin`.
- Export the weight addresses as `/tmp/weight_map.json`.

#### Step 4: Custom MLIR Dialect ✅

Build a custom MLIR dialect in `/nn_assembler/MLIR/`. The formatting of the code
and files in that directory are entirely up to you; you may choose the coding
language (likely tablegen and c++ to conform to what MLIR expects). However, the
dialect must follow a few key rules:

- **Direct translation** to the `Functional_TPU_ISA.md`. The custom dialect
must have a nearly 1:1 relation to machine code, and I should be able to easily
determine the layout of the machine code from the custom dialect.
- **Minimize** functions that will be synthesized away during assembly.
The goal of the custom dialect is to be a bridge between StableHLO and machine
code. If someone writes a computation graph in our custom dialect, our assembler
will be able to translate it to machine code.

#### Step 5: Legalization Pass ✅

Build a legalization pass to convert StableHLO to our custom dialect.

- The legalization does not need to be complete, this version of the pass only
needs to consider the operations present in `/examples/v0.1_example.mlir` and `/examples/v0.1_optimized_example.mlir`.
- The logic that actually runs the pass should be in
`/nn_assembler/Process_MLIR.py`, but the pass should be defined in
`/nn_assembler/MLIR/`. This legalization pass is the second pass, and occurs
after the first optimization pass.
- Save the result of the legalization pass to `/tmp/optimized.tpu.mlir`.

#### Step 6: Assembler ✅

Build `/nn_assembler/Assembler.py`:

- Reference the weight map json file to convert the output of the final MLIR
pass to `Functional_TPU_ISA.md` machine code.
- Prioritize using address 0x1 for intermediate results.

#### Step 7: Assembler Export ✅

Reference the `Functional_TPU_Message_Protocol.md` and build logic to...

- Export the machine instructions to the *Functional TPU Message Protocol*
format and save it as `/tmp/PROGRAM.bin`

#### Step 8: Serializer ✅

Build `/nn_assembler/Serializer.py`:

- Concatenate `/tmp/MEM.bin` and `/tmp/PROGRAM.bin` into a single binary string.
- Append START and STOP codes to generate a full *Functional TPU Message
Protocol* message.
- Save the final result to `/out/TRANSMISSION.bin`.

### v0.1.1 — Message Protocol Update ✅ COMPLETE

Minor version update to implement changes to *Functional TPU Message
Protocol*. The new protocol is functionally identical to the first; only the
formatting of the specification sheet and the categorization/naming of function
codes changed. Specifically, the `START` header code was renamed to `FLASH`
(ASCII `U` / `0x55` unchanged), and codes were grouped into *header*, *command*,
and *trailer* categories. `PROGRAM` was **not** renamed — it kept its name
(`P` / `0x50`) and was merely reclassified as a *command code*. A new `INPUT`
header code (`I` / `0x49`) was added but is **out of scope** for v0.1.1 (the
current flow programs the whole peripheral via `FLASH`).

#### Step 1: Comment and Variable Updates ✅

Update files to reference the new `FLASH` header code instead of the old `START`
function code.

- Rename the `START` constant to `FLASH` in `Protocol.py` and update all
references (`Serializer.py`, tests).
- Group the `Protocol.py` constants into header/command/trailer categories with
comments mirroring the spec.
- Update comments and docstrings (including the stale `v0.1` protocol reference
in `Protocol.py`).
- `INPUT` is skipped entirely for this version.

#### Step 2: Verification ✅

Check through existing code to ensure that the update to the message protocol
did not invalidate any old functionality.

- Fix any issues that arise.

#### Step 3: Tests ✅

Run all existing tests to verify functionality.

- Fix any issues that arise.

### v0.3 — Per-Layer Requantization & Bias Removal ✅ COMPLETE

Implements the assembler side of the system-wide v0.3 quantization upgrade. The
version jumps 0.1.1 → 0.3.0 to align with the project-wide v0.3 milestone (ISA,
hardware spec, and neural-network export all move to 0.3.0 together). Matmul
partitioning is still deferred (no `MAX_MATMUL_SIZE` splitting yet).

**Background.** v0.3 replaces uniform Q0.7 requantization with a per-layer
multiplier `M = (S_in · S_w) / S_out`, encoded in the MUL instruction as the
dyadic pair `M0 · 2^-n` (`M0` = bits 59-52, `n` = bits 51-44; see
`Functional_TPU_ISA.md`). The neural-network export (`Neural_Networks/Operations.py`)
already writes per-layer `__M__*` and `__scales__*` arrays into `weights.npz`.
Biases are dropped from the instruction stream — validated as accuracy-neutral for
the target networks (LeNet-5 top-1 unchanged when biases are dropped vs. present)
— but remain resident in weight memory.

> **Design decision — weight quantization scale.** Weights move from Q0.7 (fixed
> `1/128`) to **per-tensor symmetric int8** (`S_w = absmax(W)/127`). This is a
> correctness requirement, not an optimization: the exported `M` was computed
> with the per-tensor `S_w`, so the quantized weight values and `M` must share
> that scale or every layer's output is wrong by a factor of ~`128·S_w`.

#### Step 1: Per-Tensor Weight Quantization ✅

Update `Process_Weights.py`:

- Replace Q0.7 weight quantization with per-tensor symmetric int8:
`w_q = clamp(round(w / S_w), -127, 127)`, reading `S_w` from the npz
`__scales__<name>` array (keeps weights consistent with the exported `M`).
- Continue quantizing and mapping bias tensors (their scale is irrelevant since
they are unused downstream) so that weight addresses remain stable.
- Confirm the MEM format and address layout are unchanged — only byte values
differ.

#### Step 2: Requantization Multiplier Decomposition ✅

Add dyadic decomposition to `Process_Weights.py`:

- Define `REQUANT_MANTISSA_BITS = 8` as a module constant (must match the ISA
`M0` field width).
- For each layer, read `__M__<name>` and decompose `M → (M0, n)`:
  - `e = floor(log2(M)) + 1`; `m = M · 2^-e`; `M0 = round(m · 2^B)`; `n = B - e`.
  - Renormalize the mantissa overflow: if `M0 == 2^B`, set `M0 = 2^(B-1)` and
  `n -= 1` (lossless).
  - Assert `128 <= M0 <= 255` and `0 <= n <= 255` (both are unsigned ISA fields).
  A failing `n >= 0` assert implies `M >= 256`, i.e. a degenerate/mis-calibrated
  layer.
  - Handle `M == 0` (dead layer) with a descriptive assert.
- Add `M0` and `n` to each weight tensor's entry in `weight_map.json` (bias
entries get none).

#### Step 3: Dialect — `M0`/`n` on `mult` ✅

Update `nn_assembler/MLIR/dialect.py`:

- Add `M0` and `n` integer attributes to `MultOp`.
- Update the textual serialize/parse so the attributes round-trip and appear in
`/tmp/optimized.tpu.mlir`, keeping the dialect ~1:1 with the MUL instruction.

#### Step 4: Legalization — Annotate `mult` with `M0`/`n` ✅

Update `nn_assembler/MLIR/legalize.py` (and `Process_MLIR.py` to supply the map):

- When lowering `dot_general → MultOp`, resolve the weight operand to its
`weight_map.json` entry and set the op's `M0`/`n` attributes.
- Pass `weight_map.json` into the pass from `Process_MLIR.py`.

#### Step 5: Bias-Removal Pass ✅

Add a new TPU-dialect pass in `nn_assembler/MLIR/`:

- Identify `AddOp`s whose operand traces to a `*.bias` function argument.
- Replace all uses of the add's result with the non-bias operand (the matmul
result), then erase the `AddOp`.
- Target **only** bias adds — leave any non-bias adds (e.g. future
matmul-partition partial sums) untouched.
- Wire into `Process_MLIR.py` after legalization; write the intermediate to
`/tmp/optimized.nobias.tpu.mlir`.

#### Step 6: Assembler — Encode `M0`/`n` ✅

Update `Assembler.py`:

- In the MUL encoder, place `M0` in bits 59-52 and `n` in bits 51-44 (from the
`MultOp` attributes); leave the remaining reserved bits 0.
- No bias `ADD` instructions are emitted (removed in Step 5); non-bias `AddOp`
encoding is unchanged.

#### Step 7: End-to-End Verification ✅

- Run the full pipeline on `Tiny_NN` and confirm `/out/TRANSMISSION.bin` is
produced.
- Verify the emitted MUL instructions carry the expected `M0`/`n` and that no
bias `ADD` instructions remain.
- **Scope:** the dialect supports only linear layers (`mult`/`add`/`relu`); conv
and pooling are unsupported, so `LeNet_5` is a numerical reference only (validated
in Python at 95.34% top-1, lossless vs. float) and is not an assembler E2E target.

#### Tests

Add/extend pytest coverage per step:

- `test_process_weights.py`: per-tensor weight values; `M → (M0, n)`
decomposition incl. renormalization and the asserts; `weight_map.json` carries
`M0`/`n` on weight entries only.
- `test_dialect_and_legalize.py`: `MultOp` serialize/parse round-trip with
`M0`/`n`; legalization annotates correct values; bias-removal drops and reroutes
bias adds while preserving non-bias adds.
- `test_assembler.py`: MUL bit-fields (`M0` at 59-52, `n` at 51-44) match
hand-derived values; the Tiny_NN program contains no `ADD` opcodes.
- `test_serializer_and_e2e.py`: the full Tiny_NN pipeline still yields a framed
`TRANSMISSION.bin`.

### v0.4 — Expanded Memory & Address Widening

Aligns the assembler with the v0.4 memory overhaul: 24-bit instruction address
fields (`Functional_TPU_ISA.md` v0.4), a 3-byte MEM address
(`Functional_TPU_Message_Protocol.md` v0.3), and a unified data memory in which
`0x1` is reserved for network I/O staging while all intermediates live in main
data memory. Matmul partitioning (`MAX_MATMUL_SIZE`) remains deferred to a later
version; this scope keeps the simple allocator sufficient for the current linear
models.

#### Step 1 — 24-bit Instruction Encoding — ✅ Complete (2026-07-09)

Update `Assembler.py`:

- Rewrite `encode_mult`, `encode_add`, and `encode_relu` bit placements for the
  v0.4 layout — `rs1`/`rs2`/`rd` are now 24-bit fields (the current encoders are
  hardcoded to the old 16-bit positions). Match the field ranges in
  `Functional_TPU_ISA.md` exactly.
- Verify emitted instructions remain 128 bits with reserved bits zeroed.

#### Step 2 — 3-Byte MEM Address & Chunking — ✅ Complete (2026-07-09)

Update `Protocol.py`:

- Emit a 3-byte MEM address (LADD/MADD/UADD, LSB first).
- Chunk any tensor longer than the 16-bit MEM length field (65535 words) across
  multiple MEM commands with incrementing addresses (e.g. `Bigger_NN`'s
  100000-word layer). The length field is **not** widened.

#### Step 3 — Data-Memory Allocation — ✅ Complete (2026-07-09)

Update weight/address mapping (`Process_Weights.py`) and the assembler
(`Assembler.py`):

- Reserve `0x1` for the network input; the final network output must be written
  to `0x1` for the programmer to read back.
- Place each intermediate result in main data memory (`0x2+`) instead of sharing
  `0x1`; reuse freed regions across the (linear) op sequence (simple scope — no
  full liveness analysis yet).
- Keep weights/biases mapped contiguously as before.

#### Step 4 — End-to-End Verification — ✅ Complete (2026-07-09)

- Run the full pipeline on `Tiny_NN`; confirm a framed `/out/TRANSMISSION.bin`.
- Verify MUL/ADD/RELU carry 24-bit addresses, MEM blocks use 3-byte addresses,
  intermediates occupy main memory, and the final result targets `0x1`.

#### Tests

- `test_assembler.py`: 24-bit field placement for mult/add/relu; instructions
  remain 128-bit with reserved bits zeroed.
- `test_protocol.py`: 3-byte MEM address; multi-command chunking of a tensor
  longer than 65535 words.
- `test_process_weights.py`: intermediates allocated in main memory with reuse,
  `0x1` reserved for I/O, and the final output mapped to `0x1`.
- `test_serializer_and_e2e.py`: the full Tiny_NN pipeline still frames correctly.
