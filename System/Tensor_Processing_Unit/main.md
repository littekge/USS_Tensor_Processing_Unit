# Functional TPU - Complete Build Specification

> **Purpose:** This document is the single source of truth for the Functional
> TPU project. It defines the project identity, current state, architecture, and
> full incremental build plan.

## Project Identity

- **Name:** Functional TPU
- **Version:** 0.6.0
- **Target Platform:** Terasic DE1-SoC FPGA
- **Language:** Verilog HDL 2001
- **Development Environment:** Quartus Prime Lite Edition
- **Development Platform:** Windows 11
- **License:** MIT License
- **Change Log:** `log.md`
- **Specifications:** — located at `../../Specifications/`
  - **Instruction Set Architecture:** `Functional_TPU_ISA.md` (TPU
  instruction set)
  - **Messaging Protocol:** `Functional_TPU_Message_Protocol.md` (defines
  byte format for communicating with the TPU)
  - **Hardware Spec:** `Functional_TPU_Hardware_Specification.md`
  (detailed description of Verilog module implementations)

## General Project Description

The Functional TPU is a tensor processing unit roughly based on Google's first
generation TPU. This project aims to create Verilog HDL code that implements the
instruction set detailed in `Functional_TPU_ISA.md` in order to accelerate
small neural networks using an FPGA.

**Design Goals:**

- Output-stationary systolic array for matrix multiplication using Verilog
generate statements.
- Storage of tensors as flattened arrays.
- Vector processors to handle loading and storing of tensors.
- Unified data memory (weights + intermediates) behind a `Data_Memory` wrapper,
with a separate `0x1` I/O-staging buffer and a separate program memory.
- Proper requantization of data in between processing steps.
- Dynamic programming of TPU through SPI link.
- Easy modification of systolic array size using Verilog parameters; per-layer
requantization supplied at runtime via instruction-encoded `M0`/`n`.

## Build Parameters

- **Systolic Array Size:** N=8 (creates 8x8 systolic array)
- **Multiply Accumulate Unit Accumulator Size:** 32 bits
- **Memory Datatype:** 8 bit signed integer
- **Memory Latency:** 2 clock cycles
- **Re-quantization Method:** Per-layer dyadic requantization — each MUL
instruction supplies an 8-bit multiplier `M0` and 8-bit right-shift `n`; the
Vector_Processor computes `clamp((M0 * x + (1 << (n - 1))) >> n)` on systolic
array outputs, saturating to the memory datatype (`M0`/`n` mantissa width B=8)
- **Data Address Width:** 24-bit ISA address fields; physical data memory is
2^17 = 131072 words (addresses above the implemented range raise an error)
- **Memory Specs by Module:**
  - **Data_Memory:** wrapper presenting one flat 24-bit dual-port (a/b) address
  space; instantiates the 0x1 buffer and both Mem_Unit blocks and performs
  reserved-address / block-select decode:
    - `0x0` → hardwired-zero register (write raises `WRITE_ERROR`)
    - `0x1` → `TPU_0x1_Buffer` (I/O staging)
    - `addr[16]` → selects `Mem_Unit` block A (0) or B (1) for the 2–131071 range
    - any address bit above [16] set → out-of-range error
  - **Mem_Unit (×2):**
    - **Width:** 8 bits
    - **Depth:** 65536 words (Quartus IP max per instance; two blocks = 131072)
  - **TPU_0x1_Buffer:**
    - **Width:** 8 bits
    - **Depth:** 1024 words
  - **Program_Memory:**
    - **Width:** 128 bits
    - **Depth:** 8192 words
  - **Vector_Buffer:**
    - **Width:** 8 bits
    - **Depth:** 65536 words
  - **Systolic_Array_Input_Buffer:**
    - **Width:** 8 bits
    - **Depth:** N words
  - **SPI_Input_buffer:**
    - **Width:** 8 bits
    - **Depth:** 256 words

## Current State — Built and Verified Through v0.5.1

The TPU is implemented and Questa-verified through **v0.5.1**; the per-version
build history is the Build Plan below, with detailed results in `log.md`. Working
capabilities:

- **Compute:** matrix multiply on an output-stationary systolic array with signed
MACs, tensor addition (ALU), and ReLU (Activator), with per-layer dyadic
requantization applied to systolic-array outputs on writeback.
- **Large matmuls:** decomposed into array-sized tiles addressed in place via the
leading-dimension registers (`stride` → `ld1`/`ld2`; `mult`/`multip` walk strided
sub-blocks, contraction accumulated in-array).
- **Memory:** a unified 24-bit dual-port `Data_Memory` wrapping two 65536-word
`Mem_Unit` blocks and the 1024-word `TPU_0x1_Buffer` (I/O staging), plus an
8192-word `Program_Memory`.
- **IO / programming:** SPI programming through a synchronous-oversampling
`SPI_Slave`; the Programmer message-protocol FSM (device and external modes); and
the Feeder + Controller fetch–decode–execute–writeback cycle.

**In progress — v0.6 (Convolution and Pooling):** ISA and Hardware Specification
finalized (both v0.6.0); RTL build plan below; user-provided `Pooler.v` skeleton
created. No v0.6 RTL implemented yet.

## Architecture

The project architecture is depicted below, with each .v file corresponding to a
module in the `Functional_TPU_Hardware_Specification.md`. Note that the
VGA_DEBUG folder is not a part of the main project or hardware spec but may be
used to debug the TPU.

```markdown
TPU/
├── TPU.v (top-level module)
├── CONTROL/
│   └── Controller.v
├── MEMORY/
│   ├── Data_Memory.v (top-level: 0x1 buffer + 2× Mem_Unit + address decode)
│   ├── 0X1_BUFFER/
│   │   └── TPU_0x1_Buffer.v
│   ├── MEM_UNIT/
│   │   └── Mem_Unit.v (instantiated twice by Data_Memory as blocks A and B)
│   └── VECTOR_BUFFER/
│       └── Vector_Buffer.v
├── PROCESSING/
│   ├── Activator.v
│   ├── ALU.v
│   ├── Pooler.v
│   └── Vector_Processor.v
├── PROGRAMMER/
│   ├── Feeder.v
│   ├── Programmer.v
│   ├── PROGRAM_MEMORY/
│   │   └── Program_Memory.v
│   └── SPI_LINK/
│       ├── SPI_Interface.v (top-level for SPI subsystem)
│       ├── SPI_Slave.v
│       └── SPI_INPUT_BUFFER/
│           └── SPI_Input_buffer.v
├── SYSTOLIC_ARRAY/
│   ├── Multiply_Accumulate_Unit.v
│   ├── Systolic_Array_Input_Buffer.v
│   └── Systolic_Array.v (top-level for systolic array subsystem)
└── VGA_DEBUG/
    ├── debug.v
    ├── seven_segment.v
    ├── seven_segment_negative.v
    ├── three_decimal_vals_w_neg.v
    └── VGA_CONTROLLER/
        ├── ascii_master_controller.v
        ├── clock_divider.v
        ├── vga_controller.v
        └── vga_driver.v
```

## Build Plan

The build plan for this project is *versioned*. Early versions will **NOT**
implement the final design goals of the project; simpler logic will be tested
and verified to work before more complex behavior is added. Build steps for
future versions will be added incrementally as needed, and the design
goals/lowering pipeline may be updated to reflect new versions. See
`Functional_TPU_Hardware_Specification.md` for detailed descriptions
of each module.

### v0.1 — Basic Functionality

Implements initial TPU functionality. At this stage, the TPU can perform matrix
multiplication, tensor addition, and ReLu activation.

1. ~~Build and instantiate TPU and Programmer modules and MUX connections to the
correct memory modules. Define the *trst* signal.~~
2. ~~Build and instantiate the Feeder module. Update the TPU module to route
connections from the program memory to the feeder.~~
3. ~~Build and instantiate the Controller module. Update the TPU module to route
connections from the feeder to the controller.~~
4. ~~Build Vector_Processor module and instantiate vector processors *a* and *b*
in the TPU module. Update the TPU module to route connections between the vector
processor, controller, and memory.~~
5. ~~Build and instantiate the Activator module. Update the TPU module to route
input and control signals from vector processor *a* to the activator and output
from the activator to the vector buffer, implementing intermediate combinational
logic and quantization in the process.~~
6. ~~Build and instantiate the ALU module. Update the TPU module to
appropriately route inputs and control signals from both vector processors to
the ALU and outputs from the ALU to the vector buffer, implementing intermediate
combinational logic and quantization in the process. Ensure that the
*element_valid* signals of the two vector processors are properly connected to
the *enable* signal of the ALU.~~
7. ~~Build and instantiate the Systolic_Array module and submodules
(Systolic_Array_Input_Buffer and Multiply_Accumulate_Unit). Update the TPU
module to route control signals from the controller to the systolic array.~~
8. ~~Test the entire TPU by~~
    1. ~~Creating a basic Functional TPU ISA program that uses each instruction
    at least once.~~
    2. ~~Formatting the program and relevant test weights into the Functional
    TPU message protocol.~~
    3. ~~Simulating programming over SPI and checking that each instruction
    functions as expected.~~

### v0.2 — Improved IO

Adds improved IO functionality to the Programmer module to better demonstrate
TPU functionality.

#### Step 1 — Programmer Module — ✅ Complete (2026-06-30)

Update the Programmer module to reflect the current version of the
*Functional TPU Hardware Specification*.

- Add meta-states (can be either a single large state machine or a nested state
machine, whatever you think is easier to read).
- Add IO signals.
- Verify new behavior is properly implemented.
- Rewrite Programmer unit test to thoroughly test new functionality
(`./tests/TB_Step1_Programmer.v`).

> **Clarification (2026-06-30):** In `EXTERNAL_INPUT` (an INPUT-header
> transmission), MEM command addresses use **unified address routing** —
> interpreted exactly as in `PROGRAM`/FLASH mode (routed to weight memory or the
> 0x1 buffer by address). `EXTERNAL_INPUT` differs from `PROGRAM` only in that it
> maintains the current program (does not reset or write Program_Memory), so the
> PROGRAM command code is invalid inside an INPUT transmission.

#### Step 2 — TPU Module — ✅ Complete (2026-06-30)

Update the TPU module to reflect the current version of the
*Functional TPU Hardware Specification*.

- Update *trst* implementation.
- Implement signal passthrough for programmer.

> **Clarification (2026-06-30):** The new Programmer IO (*mode_select*,
> *input_data*, *input_data_valid*, *device_ready*, *output_data*,
> *output_data_valid*) is exposed **at the TPU module boundary only**. The FPGA
> wrapper (`Tensor_Processing_Unit.v`) pin mapping is left to the user and is not
> modified by this build step (respects the "do not modify debug sections" rule).

#### Step 3 — Feeder Module — ✅ Complete (2026-06-30)

Update the Feeder module to reflect the current version of the
*Functional TPU Hardware Specification*.

- Add *end_reached* signal.
- Rewrite Feeder unit test to thoroughly test new functionality
(`./tests/TB_Step2_Feeder.v`).

#### Step 4 — Additional Minor Changes — ✅ Complete (2026-06-30)

Search the *Functional TPU Hardware Specification* for additional changes.

- Implement additional changes.
- Revise relevant tests and run to confirm functionality.

#### Step 5 — Final Test — ✅ Complete (2026-06-30)

Revise the full system test to test new functionality.

- Rewrite System test to thoroughly test new functionality
(`./tests/TB_Step8_FullSystem.v`).
- Run full system test and fix any issues that arise.

> **Clarification (2026-06-30):** The revised full-system test exercises **both**
> IO modes end-to-end: device mode (single input via *input_data* /
> *input_data_valid* → program runs → single *output_data* value) and external
> mode (INPUT transmission over SPI → program runs → 10 *output_data* values with
> the *device_ready* handshake).

### v0.2.1 — Systolic Array Signed Multiplication Fix

Corrects a latent datatype bug in the systolic array. The Build Parameters
declare the Memory Datatype as an 8-bit *signed* integer, but the
Multiply_Accumulate_Unit performs an *unsigned* multiply. As a result, negative
operands (e.g. a weight of −1 = 0xFF) accumulate as large positive values (255)
instead of their true signed value, corrupting any dot product that contains a
negative operand and biasing results strongly positive. This is a suspected
contributor to the requantization saturation (constant 0x7F output) previously
investigated. The Hardware Specification requires no change — signed operands are
already implied by the Memory Datatype; only the RTL must be corrected to match.

#### Step 1 — Multiply_Accumulate_Unit Signed Multiply — ✅ Complete (2026-07-02)

Correct the multiply in the Multiply_Accumulate_Unit module to operate on signed
operands.

- Treat *a_in* and *b_in* as signed 8-bit values in the *c* = *c* + *a_in* *
  *b_in* accumulation so the 32-bit accumulator holds the correct signed product.
- Confirm the accumulator and the downstream Vector_Processor requantization
  (which already interprets the value as signed) now agree.
- Update the Systolic Array unit test (`./tests/TB_Step7_SystolicArray.v`) to
  exercise negative operands through the array and verify the signed product.

#### Step 2 — Regression and Requantization Impact — ✅ Complete (2026-07-02)

Confirm the fix resolves negative-operand corruption without regressions.

- Run the full testbench regression suite and fix any issues that arise.
- Add or extend a full-system test (`./tests/TB_Step8_FullSystem.v`) that streams
  negative weights through the systolic array end to end.
- Record the impact on the previously observed 0x7F saturation behavior in
  `log.md`.

### v0.3 — Per-Layer Requantization

Replaces uniform Q0.7 requantization with per-layer dyadic requantization. Each
MUL instruction now carries an 8-bit multiplier `M0` (bits 59-52) and an 8-bit
right-shift `n` (bits 51-44); the Vector_Processor applies
`result = clamp((M0 * x + (1 << (n - 1))) >> n)` to each systolic array output
before writeback. See `Functional_TPU_ISA.md` and the Hardware Specification
(Vector_Processor and Controller descriptions) for the authoritative definitions.

**Scope note.** This is an RTL change confined to the **Vector_Processor**
(requantization datapath) and the **Controller** (instruction decode + routing):

- The **ALU already clamps** rather than requantizes, so it matches the updated
  spec with no change (the spec's "Requantization" bullet was reworded to
  "Overflow handling" to reflect the existing RTL).
- The MUL instruction stays 128 bits (`M0`/`n` consume previously reserved bits),
  so **Program_Memory and the Message Protocol are unaffected**.
- The **32-bit MAC accumulator** already feeds requantization at full width.
- The B=8 mantissa was validated in Python as lossless (LeNet-5 top-1 95.34%,
  matching float), so hardware output should match the software reference.

#### Step 1 — Vector_Processor Requantization — ✅ Complete (2026-07-06)

Update `Vector_Processor.v` to implement the v0.3 requantization datapath.

- Add `scale` (`M0`, 8-bit unsigned) and `shift` (`n`, 8-bit unsigned) inputs.
- Replace the existing requantization on systolic array output reads with
  `result = clamp((scale * x + (1 << (shift - 1))) >> shift)`, where `x` is the
  signed 32-bit accumulator:
  - Multiply `scale * x` (scale unsigned, `x` signed); size the intermediate to
    hold the product without overflow (accumulator width + 8 bits).
  - Add the rounding bias `1 << (shift - 1)` **before** the arithmetic right shift
    by `shift` (round-to-nearest, not truncation).
  - Saturate the result to the 8-bit signed memory datatype.
- Requantization applies only when reading from systolic array outputs; other
  reads (e.g. vector buffer) are unaffected.
- Update `tests/TB_Step4_VectorProcessor.v` to drive representative `scale`/`shift`
  values and verify the rounded, saturated result against hand-computed vectors.

#### Step 2 — Controller Decode and Routing — ✅ Complete (2026-07-06)

Update `Controller.v` to decode and route the requantization parameters.

- Decode `scale` (bits 59-52) and `shift` (bits 51-44) from MUL instructions.
- Drive the vector processor's `scale`/`shift` inputs during the systolic array
  writeback step. For non-MUL instructions these are don't-care.
- Update `tests/TB_Step3_Controller.v` to verify `scale`/`shift` are decoded and
  asserted to the correct vector processor at the correct time.

#### Step 3 — Regression and Full-System Verification — ✅ Complete (2026-07-07)

Confirm the requantization change end to end.

- Run the full testbench regression suite and fix any issues that arise.
- Extend `tests/TB_Step8_FullSystem.v` to program a matmul with representative
  `M0`/`n` and verify the requantized output matches the expected values (derived
  from the Python reference / hand computation).
- Record the results in `log.md`.

### v0.4 — Memory Expansion

Expands device memory and unifies the data address space. Data memory grows to
2^17 = 131072 words behind a new `Data_Memory` wrapper that abstracts the `0x1`
I/O-staging buffer and two 65536-word `Mem_Unit` blocks (Quartus caps on-chip RAM
IP at 2^16 words per instance) into one flat, 24-bit, dual-port address space.
Program memory grows to 8192 instructions. Instruction address fields widen
16 → 24 bits per `Functional_TPU_ISA.md` v0.4, and the MEM command carries a
3-byte address per `Functional_TPU_Message_Protocol.md` v0.3. The `0x1` buffer is
retained but repurposed as I/O staging only; all intermediates now live in main
data memory.

**Prerequisite (user-provided).** The RAM IP and module skeletons are created by
the user before agents begin, so the architecture appears exactly as described
above: two `Mem_Unit` blocks, the 8192-deep `Program_Memory`, the 1024-deep
`TPU_0x1_Buffer`, and the `Data_Memory.v` wrapper skeleton. Agents implement
logic and wiring only — no new `.v` files are created (per the architecture rule).

**Scope note.** This is an RTL change to the memory subsystem and the modules
that address it. The compute datapath (Systolic_Array, Activator, ALU,
Vector_Buffer) is unaffected apart from address-width changes threaded through.

#### Step 1 — Data_Memory Wrapper — ✅ Complete (2026-07-09)

Implement `Data_Memory.v` (user-provided skeleton):

- Instantiate the `TPU_0x1_Buffer` and both `Mem_Unit` blocks (A, B); expose one
  flat dual-port (a/b) interface with 24-bit address inputs.
- Address decode per port: `0x0` → hardwired zero on read, write raises
  `WRITE_ERROR`; `0x1` → `TPU_0x1_Buffer`; `addr[16]` selects block A (0) / B (1)
  for the 2–131071 range; any address bit above [16] set → out-of-range error.
- Preserve the 2-cycle memory latency; thread the program/compute MUX inputs
  (data/address/wren for ports a and b) through the wrapper.
- Add `tests/TB_Step1_DataMemory.v` covering each decode case (zero-register read,
  `0x1` routing, block A/B select, `0x0` write error, out-of-range error).

#### Step 2 — TPU Top-Level Rewire — ✅ Complete (2026-07-09)

Update `TPU.v`:

- Replace the direct `Weight_Memory` + `TPU_0x1_Buffer` instantiations with a
  single `Data_Memory` instance.
- Route the existing memory MUX (programmer while programming; feeder →
  program memory; vector processor *a*/*b* → data memory) into the wrapper ports.
- Widen all data-memory address nets to 24 bits.
- Revise `tests/TB_Step8_FullSystem.v` dependencies for the rewire.

#### Step 3 — Vector_Processor Address Widening — ✅ Complete (2026-07-09)

Update `Vector_Processor.v`:

- Widen `mem_source_address` / `mem_dest_address` and internal address registers
  to 24 bits.
- Remove the memory-abstraction logic (weight-vs-0x1 routing) now owned by
  `Data_Memory`; issue flat addresses only. Keep logical access (flatten↔matrix
  ordering, length/dim walking, requantization).
- Update `tests/TB_Step4_VectorProcessor.v` for 24-bit addressing.

#### Step 4 — Programmer Address & 3-Byte MEM — ✅ Complete (2026-07-09)

Update `Programmer.v`:

- Widen `mem_addr` to 24 bits.
- Decode the 3-byte MEM address (LADD/MADD/UADD, LSB first) per Message Protocol
  v0.3.
- Continue staging device/external I/O at address `0x1`; write data through the
  `Data_Memory` wrapper (port *a*).
- Update `tests/TB_Step1_Programmer.v` for the 3-byte address.

#### Step 5 — Feeder & Program Memory Depth — ✅ Complete (2026-07-09)

Update `Feeder.v`:

- Widen `pc` to 13 bits; set `PM_MAX_ADDRESS = 8191`; update `FEED_ERROR` /
  overflow detection for the deeper program memory.
- Confirm the `Program_Memory` IP is 8192 words deep.
- Update `tests/TB_Step2_Feeder.v`.

#### Step 6 — Regression — ✅ Complete (2026-07-09)

- Update `tests/run_regression.sh` for the new/renamed testbenches.
- Run the full regression on the Questa PC and record results in `log.md`.
- On non-Questa machines record `Verification: PENDING` and leave the v0.4 steps
  unmarked until the regression passes on the Questa PC.

### v0.5 — Partitioning Update (Leading-Dimension Addressing)

Enables matmuls larger than the 8x8 systolic array by tiling, with sub-block
operands addressed in place via leading dimensions. Implements the full v0.5
ISA/HW spec set: the `multip` instruction (opcode 1000, funct3 0x1) with
accumulator-clear-after-writeback; the `stride` instruction (CONFIG format,
opcode 1111) that loads the `ld1`/`ld2` leading-dimension registers; and strided
operand addressing in the Vector_Processor. A `stride` sets the parent row
strides once per matmul; `mult`/`multip` then read/write sub-blocks; a `multip`
run terminated by a `mult` accumulates tiled products in-array (per
`Functional_TPU_ISA.md` v0.5 and `Functional_TPU_Hardware_Specification.md` v0.5).

**Scope note.** Control-path and addressing change to the compute datapath; the
v0.4 memory subsystem is untouched. No new `.v` files; debug sections untouched.

#### Step 1 — MAC & Systolic_Array Clear Split — ✅ Complete (2026-07-14)

- `Multiply_Accumulate_Unit.v`: `clear` resets only the operand pipeline registers
  (`o_a`, `o_b`); add an `accumulator_clear` input that zeroes the accumulator
  `o_c`. `trst` still resets all registers.
- `Systolic_Array.v`: add an `accumulator_clear` input routed to every MAC; `clear`
  still flushes input buffers, the rdreq pipeline, and MAC operand registers.
- Update `tests/TB_Step7_SystolicArray.v`: accumulator survives `clear`, zeroed
  only by `accumulator_clear`/`trst`.

#### Step 2 — Controller: funct3 Decode, Clear Retiming, and stride/ld Registers — ✅ Complete (2026-07-14)

- `Controller.v`: decode funct3 on opcode 1000 (`mult` 0x0 / `multip` 0x1); move
  the accumulator clear to after writeback, asserted for one cycle only when
  funct3 selects clearing (`mult`, not `multip`).
- Decode the `stride` instruction (opcode 1111): latch `im1`/`im2` into new
  `ld1`/`ld2` registers; no execute or writeback phase; reset `ld1`=`ld2`=0 on
  `trst`.
- Supply the active leading dimension to the vector processors: `ld1` when loading
  `src1`, `ld2` when loading `src2`, and `ld2` during writeback to `dest`.
- Update `tests/TB_Step3_Controller.v`: both variants decode; `accumulator_clear`
  asserted after writeback for `mult` and suppressed for `multip`; `stride` latches
  `ld1`/`ld2`.

#### Step 3 — Vector_Processor: Strided Sub-Block Addressing — ✅ Complete (2026-07-14)

- `Vector_Processor.v`: add a `leading_dimension` input; generalize operand-load
  and writeback address generation to step rows by the active leading dimension
  instead of the matrix width, defaulting to tile width when 0 (contiguous).
  Applies to `src1`→top buffers, `src2`→left buffers, and `dest` writeback.
- Update `tests/TB_Step4_VectorProcessor.v`: strided sub-block read and writeback;
  contiguous (leading dimension 0) matches v0.4 behavior.

#### Step 4 — TPU Top-Level Routing — ✅ Complete (2026-07-14)

- `TPU.v`: route the Controller's `accumulator_clear` to the Systolic_Array
  (independent of the shared `ctrl_clear`); route the `leading_dimension` control
  net from the Controller to the Vector_Processors.
- Update `tests/TB_Step8_FullSystem.v`: an end-to-end program issuing a `stride`
  and a tiled matmul larger than 8x8 (a `multip` run terminated by a `mult`, with
  strided sub-block operands), checked against the requantized reference.

#### Step 5 — Regression — ✅ Complete (2026-07-14)

- Update `tests/run_regression.sh` for changed testbench dependencies
  (TB_Step3/4/7/8).
- Run the full regression on the Questa PC and record results in `log.md`.
- On non-Questa machines record `Verification: PENDING` (naming TB_Step3/4/7/8)
  and leave the v0.5 steps unmarked until the regression passes on the Questa PC.

### v0.5.1 — SPI_Slave Hardening (Synchronous Oversampling Rewrite)

Rewrites the `SPI_Slave` internals as a fully synchronous oversampling receiver
in the `i_Clk` (50 MHz) domain, eliminating the raw-SCK-as-clock and
raw-CS-as-async-reset hazards, the last-byte/CS-deassert race, and the CDC
exposure that make large-model SPI flashing unreliable. Module ports, the
`SPI_MODE` parameter, and their functional contract are unchanged — internals
only.

**Scope note.** All changes are confined to
`TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v`; module ports, the `SPI_MODE` parameter,
and their functional behavior are unchanged; no other `.v` files; the DEBUG
section is untouched. Verification is Questa-gated. The `SPI_Slave` description
in the Hardware Specification (currently "pulled from nandland") will be stale
after this rewrite; per the user this spec update is intentionally deferred
until the hardware is built.

**Contract preserved.** `o_RX_DV` is a single `i_Clk`-cycle pulse per received
byte with `o_RX_Byte` (MSB-first) valid on that cycle; multi-byte per CS-low
window; TX/MISO behavior (preload MSB on CS-assert, MSB-first shift, tri-state
on CS-high, `i_TX_DV` latches `i_TX_Byte`); active-low reset; full `SPI_MODE`
0–3 semantics.

#### Step 1 — Synchronous Oversampling Rewrite — ✅ Complete (2026-07-14)

- Synchronize `i_SPI_Clk`, `i_SPI_MOSI`, and `i_SPI_CS_n` into the `i_Clk`
  domain via 2-stage metastability chains.
- Debounce SCK and CS: accept a level change only after `DEBOUNCE_N` stable
  `i_Clk` samples (localparam, default 3 ≈ 60 ns) to reject slow-edge bounce and
  glitches.
- Edge-detect the debounced SCK per `SPI_MODE` (CPOL/CPHA) to select the sample
  edge; sample `i_SPI_MOSI` MSB-first into a shift register; on the 8th sampled
  bit latch `o_RX_Byte` and assert `o_RX_DV` for exactly one `i_Clk` cycle.
- Synchronous CS framing: CS-assert resets the bit counter; CS-deassert never
  asynchronously clears received-byte state (fixes the last-byte race); a
  partial byte is discarded on mid-byte CS-deassert.
- Preserve TX/MISO synchronously: preload MSB on CS-assert, shift MSB-first on
  the shift edge, tri-state `o_SPI_MISO` when CS is high, and register
  `i_TX_Byte` on `i_TX_DV`.
- Ports and the `SPI_MODE` parameter are unchanged; the DEBUG section is
  untouched.

#### Step 2 — Dedicated Unit Testbench — ✅ Complete (2026-07-14)

- Create `tests/TB_SPI_Slave.v` exercising: functional equivalence (MSB-first
  byte assembly; `o_RX_DV` a single `i_Clk` pulse with valid `o_RX_Byte`;
  multi-byte and back-to-back bytes within one CS-low window) and robustness
  (injected SCK glitches, slow/noisy edges, CS-deassert one bit after the final
  SCK edge); plus a mode-1/2/3 smoke check.
- Document pass/fail conditions per the `tests/` convention.

#### Step 3 — Regression — ✅ Complete (2026-07-14)

- Confirm `TB_Step1_Programmer` and `TB_Step8_FullSystem` still pass unchanged
  (system-level functional-equivalence check).
- Add the `TB_SPI_Slave` block to `tests/run_regression.sh`.
- Run the full regression on the Questa PC and record results in `log.md`. On
  non-Questa machines record `Verification: PENDING` (naming `TB_SPI_Slave`,
  `TB_Step1_Programmer`, `TB_Step8_FullSystem`) and leave the v0.5.1 steps
  unmarked until the regression passes on the Questa PC.

### v0.6 — Convolution and Pooling

Implements the hardware for convolution (via `im2col`) and max pooling per
`Functional_TPU_ISA.md` v0.6 and `Functional_TPU_Hardware_Specification.md` v0.6.
Adds a new `Pooler` module (a streaming reducer, peer to the ALU/Activator),
windowed address generation in the `Vector_Processor` (the `im2col` gather with
zero-fill and the `max` window stream with min-fill), and the `window` (WINCONFIG)
descriptor registers plus windowed-instruction handling in the `Controller`.
Together these let a convolution lower to `im2col` + tiled matmul and a pooling
layer to `max`, completing the datapath needed to run LeNet-5.

**Scope note.** Control-path and datapath change on the compute side; the v0.4
memory subsystem and the systolic array are untouched. No new `.v` files beyond
the user-provided `Pooler.v`; debug sections untouched. Verification is Questa-gated.

**Prerequisite (user-provided).** The `Pooler.v` module skeleton (module
declaration, port list per the Hardware Specification, and the standard
parameter/code/debug sections) is created by the user before agents begin, since
agents may not create new `.v` files. `Pooler.v` lives in `TPU/PROCESSING/`
alongside `Activator.v` and `ALU.v`. Agents implement the module logic and all
inter-module wiring only.

#### Step 1 — Pooler Module

Implement `Pooler.v` (user-provided skeleton) as a streaming reducer.

- Maintain one accumulator register and a function-selected reduction operator;
  for `max` (funct3 0x0) the operator is a signed comparator and the identity is
  the minimum representable value (`-128` for the 8-bit signed memory datatype).
- While `enable` is HIGH, fold one input per clock into the accumulator
  (`acc = max(acc, in)`); reset the accumulator to the reduction identity on
  `clear` or `trst`.
- On `window_end` (coincident with the final enabled value of a window), write the
  accumulator to the vector buffer and reset it for the next window — exactly one
  write per window. Suppress buffer writes when the function is NO OP.
- No requantization (the pooled result is an input value already in the datatype).
- Add `tests/TB_Pooler.v`: single-window max, multi-window streams resetting at
  each `window_end`, min-fill values never winning, NO-OP write suppression
  (follows the `TB_SPI_Slave.v` convention for a module added after v0.1).

#### Step 2 — Controller: Window Descriptor Registers + Windowed Decode

Update `Controller.v`.

- Add the eleven window descriptor registers (`chans`, `inh`, `inw`, `outh`,
  `outw`, `winh`, `winw`, `strh`, `strw`, `padh`, `padw`); load from the `window`
  (WINCONFIG, 1110) instruction; retain until the next `window` or `trst` (no
  default).
- Generalize the `stride` decode-and-latch path into one register-configuration
  path covering LDCONFIG (1111) and WINCONFIG (1110): latch fields into their
  registers, assert `controller_idle`, no Execute/Writeback phase.
- Decode `im2col` (SHAPE, 1101) and `max` (POOL, 1100); supply the descriptor to
  the vector processors and select the gather destination — device memory for
  `im2col`, the Pooler for `max`.
- Set the Pooler function (max / NO-OP); add the Pooler to the inter-operation
  `clear` assertion.
- Skip the Writeback phase for `im2col` (writes directly to memory during Execute).
- Update `tests/TB_Step3_Controller.v`: WINCONFIG latches all eleven registers;
  `im2col`/`max` decode, descriptor routing, and destination select; Pooler
  function/clear asserted; `im2col` skips writeback.

#### Step 3 — Vector_Processor: Windowed Addressing

Update `Vector_Processor.v`.

- Add the eleven window-descriptor combinational inputs.
- Add windowed address generation for `im2col`/`max`: derive each window element's
  input coordinate from the output position, window offset, stride, and padding;
  bounds-check against `inh`×`inw`; substitute the fill in place of a memory read
  when out of bounds — `0` for `im2col` (may source the hardwired `0x0`), minimum
  representable value for `max`. Element count is descriptor-derived, not from
  `length`.
- `im2col`: write every gathered element to `dest` as the dense
  `(chans·winh·winw)×(outh·outw)` matrix. `max`: stream each channel's window to
  the Pooler destination, asserting the new `o_window_end` (coincident with
  `element_valid`) on the final element of each window.
- Add `o_window_end`; add the Pooler as a valid destination.
- Update `tests/TB_Step4_VectorProcessor.v`: im2col gather order + zero-fill on a
  small padded map; max window stream + min-fill + `window_end` timing;
  descriptor-derived counts; confirm the v0.5 contiguous/strided paths are
  unaffected.

#### Step 4 — TPU Top-Level Integration

Update `TPU.v`.

- Instantiate and wire the Pooler: vector processor `a` → Pooler input;
  `element_valid` from `a` → Pooler `enable`; `o_window_end` from `a` → Pooler
  `window_end`.
- Add the Pooler as a third writer to the Vector_Buffer (data/`wrreq` mux with the
  ALU and Activator; only one is non-NO-OP per instruction).
- Add the Pooler to the `trst` reset list.
- Route the Controller's window-descriptor and windowed-mode control nets to the
  vector processors, and the Pooler function-select to the Pooler.
- Update `tests/TB_Step8_FullSystem.v`: an end-to-end program that (a) issues
  `window` + `im2col` and feeds the result to a tiled matmul (a convolution), and
  (b) issues `window` + `max` (a pooling layer), each checked against a reference.

#### Step 5 — Regression

- Update `tests/run_regression.sh` for the new/changed testbenches (`TB_Pooler`,
  `TB_Step3_Controller`, `TB_Step4_VectorProcessor`, `TB_Step8_FullSystem`).
- Run the full regression on the Questa PC and record results in `log.md`.
- On non-Questa machines record `Verification: PENDING` (naming the testbenches)
  and leave the v0.6 steps unmarked until the regression passes on the Questa PC.
