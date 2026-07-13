# Functional TPU - Complete Build Specification

> **Purpose:** This document is the single source of truth for the Functional
> TPU project. It defines the project identity, current state, architecture, and
> full incremental build plan.

## Project Identity

- **Name:** Functional TPU
- **Version:** 0.4.0
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

## Current State (v0.1 - What's Working)

- **SPI Link:** `SPI_Slave.v` and `SPI_Input_buffer.v` are correctly wired
together by `SPI_Interface.v` which is currently the top level module for the
`TPU/PROGRAMMER/TPU_SPI_LINK/` subdirectory. The SPI input buffer has been
verified to properly receive and store data from the SPI slave module.
- **VGA Debugging:** `debug.v` can be successfully used to display 32 bit
integers as hexadecimal values on a monitor connected to the FPGA via it's VGA port.
- **TPU Memory Modules:** `Program_Memory.v`, `TPU_0x1_Buffer.v`,
`Vector_Buffer.v`, and `Weight_Memory.v` have been generated using the Quartus
IP Catalog as device RAMs or FIFO Buffers in accordance with the `Functional_TPU_Hardware_Specification.md`.
- **Programmer:** `Programmer.v` implements the Functional TPU Messaging
Protocol FSM. Verified to correctly decode START/MEM/PROGRAM/STOP function codes
and write to Program_Memory, Weight_Memory port a, and TPU_0x1_Buffer port a.
- **TPU Top Level (Steps 1–4):** `TPU.v` instantiates and wires together
Programmer, Program_Memory, Weight_Memory, TPU_0x1_Buffer, Vector_Buffer,
Feeder, Controller, and both Vector_Processor instances. The memory MUX, *trst*
signal, and *vb_sclr* logic are all implemented.
- **Feeder:** `Feeder.v` implements the FETCH step of the CPU instruction cycle.
Verified to correctly retrieve and forward instructions from Program_Memory to
the Controller, detect the *end* termination opcode, and increment the program counter.
- **Controller:** `Controller.v` implements the DECODE-EXECUTE-WRITEBACK steps
of the CPU instruction cycle. Verified to correctly decode *mult*, *relu*, and
*add* instructions and drive all Vector_Processor and Systolic_Array control signals.
- **Vector_Processor:** `Vector_Processor.v` handles all ISA-compliant memory
accesses and data routing. Verified to correctly stream data from Weight_Memory
and TPU_0x1_Buffer (including zero-source and strided column access), push
rows/columns to systolic array input buffers, read and requantize systolic array
outputs, and read from the Vector_Buffer. MEM_ERROR is correctly triggered on
writes to reserved address 0x0000.
- **Activator:** `Activator.v` correctly implements the ReLu activation
function. Debug values are correctly outputted when in a non-operational state.
- **ALU:** `ALU.v` correctly implements addition, clamping outputs to an 8-bit
signed integer. Debug values are correctly outputted when in a non-operational state.

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

#### Step 1 — MAC & Systolic_Array Clear Split

- `Multiply_Accumulate_Unit.v`: `clear` resets only the operand pipeline registers
  (`o_a`, `o_b`); add an `accumulator_clear` input that zeroes the accumulator
  `o_c`. `trst` still resets all registers.
- `Systolic_Array.v`: add an `accumulator_clear` input routed to every MAC; `clear`
  still flushes input buffers, the rdreq pipeline, and MAC operand registers.
- Update `tests/TB_Step7_SystolicArray.v`: accumulator survives `clear`, zeroed
  only by `accumulator_clear`/`trst`.

#### Step 2 — Controller: funct3 Decode, Clear Retiming, and stride/ld Registers

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

#### Step 3 — Vector_Processor: Strided Sub-Block Addressing

- `Vector_Processor.v`: add a `leading_dimension` input; generalize operand-load
  and writeback address generation to step rows by the active leading dimension
  instead of the matrix width, defaulting to tile width when 0 (contiguous).
  Applies to `src1`→top buffers, `src2`→left buffers, and `dest` writeback.
- Update `tests/TB_Step4_VectorProcessor.v`: strided sub-block read and writeback;
  contiguous (leading dimension 0) matches v0.4 behavior.

#### Step 4 — TPU Top-Level Routing

- `TPU.v`: route the Controller's `accumulator_clear` to the Systolic_Array
  (independent of the shared `ctrl_clear`); route the `leading_dimension` control
  net from the Controller to the Vector_Processors.
- Update `tests/TB_Step8_FullSystem.v`: an end-to-end program issuing a `stride`
  and a tiled matmul larger than 8x8 (a `multip` run terminated by a `mult`, with
  strided sub-block operands), checked against the requantized reference.

#### Step 5 — Regression

- Update `tests/run_regression.sh` for changed testbench dependencies
  (TB_Step3/4/7/8).
- Run the full regression on the Questa PC and record results in `log.md`.
- On non-Questa machines record `Verification: PENDING` (naming TB_Step3/4/7/8)
  and leave the v0.5 steps unmarked until the regression passes on the Questa PC.
