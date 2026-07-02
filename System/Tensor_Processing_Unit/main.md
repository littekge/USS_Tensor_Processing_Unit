# Functional TPU - Complete Build Specification

> **Purpose:** This document is the single source of truth for the Functional
> TPU project. It defines the project identity, current state, architecture, and
> full incremental build plan.

## Project Identity

- **Name:** Functional TPU
- **Version:** 0.2.1
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
- Separate memories for program instructions, weights/biases, and temporary data.
- Proper requantization of data in between processing steps.
- Dynamic programming of TPU through SPI link.
- Easy modification of systolic array size and re-quantization scale using
Verilog parameters.

## Build Parameters

- **Systolic Array Size:** N=8 (creates 8x8 systolic array)
- **Multiply Accumulate Unit Accumulator Size:** 32 bits
- **Memory Datatype:** 8 bit signed integer
- **Memory Latency:** 2 clock cycles
- **Re-quantization Method:** Symmetric uniform quantization with saturation checks
- **Memory Specs by Module:**
  - **Weight_Memory:**
    - **Width:** 8 bits
    - **Depth:** 65536 words
  - **TPU_0x1_Buffer:**
    - **Width:** 8 bits
    - **Depth:** 65536 words
  - **Program_Memory:**
    - **Width:** 128 bits
    - **Depth:** 1024 words
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
│   ├── 0X1_BUFFER/
│   │   └── TPU_0x1_Buffer.v
│   ├── VECTOR_BUFFER/
│   │   └── Vector_Buffer.v
│   └── WEIGHT_MEMORY/
│       └── Weight_Memory.v
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

#### Step 1 — Multiply_Accumulate_Unit Signed Multiply — ⬜ Pending

Correct the multiply in the Multiply_Accumulate_Unit module to operate on signed
operands.

- Treat *a_in* and *b_in* as signed 8-bit values in the *c* = *c* + *a_in* *
  *b_in* accumulation so the 32-bit accumulator holds the correct signed product.
- Confirm the accumulator and the downstream Vector_Processor requantization
  (which already interprets the value as signed) now agree.
- Update the Systolic Array unit test (`./tests/TB_Step7_SystolicArray.v`) to
  exercise negative operands through the array and verify the signed product.

#### Step 2 — Regression and Requantization Impact — ⬜ Pending

Confirm the fix resolves negative-operand corruption without regressions.

- Run the full testbench regression suite and fix any issues that arise.
- Add or extend a full-system test (`./tests/TB_Step8_FullSystem.v`) that streams
  negative weights through the systolic array end to end.
- Record the impact on the previously observed 0x7F saturation behavior in
  `log.md`.
