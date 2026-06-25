# Functional TPU - Complete Build Specification
> **Purpose:** This document is the single source of truth for the Functional TPU project. It defines the project identity, current state, architecture, and full incremental build plan.

## Project Identity

- **Name:** Functional TPU
- **Version:** 0.1
- **Target Platform:** Terasic DE1-SoC FPGA
- **Language:** Verilog HDL 2001
- **Development Environment:** Quartus Prime Lite Edition
- **Development Platform:** Windows 11
- **License:** MIT License
- **Change Log:** `log.md`
- **Specifications:** — located at `../../Specifications/`
    - **Instruction Set Architecture:** `Functional_TPU_ISA_v0.2.md` (TPU instruction set)
    - **Messaging Protocol:** `Functional_TPU_Message_Protocol_v0.1.md` (defines byte format for communicating with the TPU)
    - **Hardware Spec:** `Functional_TPU_Hardware_Specification_v0.1.1.md` (detailed description of Verilog module implementations)
    

## General Project Description
The Functional TPU is a tensor processing unit roughly based on Google's first generation TPU. This project aims to create Verilog HDL code that implements the instruction set detailed in `Functional_TPU_ISA_v0.2.md` in order to accelerate small neural networks using an FPGA.

**Design Goals:**
- Output-stationary systolic array for matrix multiplication using Verilog generate statements.
- Storage of tensors as flattened arrays.
- Vector processors to handle loading and storing of tensors.
- Separate memories for program instructions, weights/biases, and temporary data.
- Proper requantization of data in between processing steps.
- Dynamic programming of TPU through SPI link.
- Easy modification of systolic array size and re-quantization scale using Verilog parameters.

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
- **SPI Link:** `SPI_Slave.v` and `SPI_Input_buffer.v` are correctly wired together by `SPI_Interface.v` which is currently the top level module for the `TPU/PROGRAMMER/TPU_SPI_LINK/` subdirectory. The SPI input buffer has been verified to properly receive and store data from the SPI slave module.
- **VGA Debugging:** `debug.v` can be successfully used to display 32 bit integers as hexadecimal values on a monitor connected to the FPGA via it's VGA port.
- **TPU Memory Modules:** `Program_Memory.v`, `TPU_0x1_Buffer.v`, `Vector_Buffer.v`, and `Weight_Memory.v` have been generated using the Quartus IP Catalog as device RAMs or FIFO Buffers in accordance with the `Functional_TPU_Hardware_Specification_v0.1.1.md`.
- **Programmer:** `Programmer.v` implements the Functional TPU Messaging Protocol FSM. Verified to correctly decode START/MEM/PROGRAM/STOP function codes and write to Program_Memory, Weight_Memory port a, and TPU_0x1_Buffer port a.
- **TPU Top Level (Steps 1–4):** `TPU.v` instantiates and wires together Programmer, Program_Memory, Weight_Memory, TPU_0x1_Buffer, Vector_Buffer, Feeder, Controller, and both Vector_Processor instances. The memory MUX, *trst* signal, and *vb_sclr* logic are all implemented.
- **Feeder:** `Feeder.v` implements the FETCH step of the CPU instruction cycle. Verified to correctly retrieve and forward instructions from Program_Memory to the Controller, detect the *end* termination opcode, and increment the program counter.
- **Controller:** `Controller.v` implements the DECODE-EXECUTE-WRITEBACK steps of the CPU instruction cycle. Verified to correctly decode *mult*, *relu*, and *add* instructions and drive all Vector_Processor and Systolic_Array control signals.
- **Vector_Processor:** `Vector_Processor.v` handles all ISA-compliant memory accesses and data routing. Verified to correctly stream data from Weight_Memory and TPU_0x1_Buffer (including zero-source and strided column access), push rows/columns to systolic array input buffers, read and requantize systolic array outputs, and read from the Vector_Buffer. MEM_ERROR is correctly triggered on writes to reserved address 0x0000. 
- **Activator:** `Activator.v` correctly implements the ReLu activation function. Debug values are correctly outputted when in a non-operational state.
- **ALU:** `ALU.v` correctly implements addition, clamping outputs to an 8-bit signed integer. Debug values are correctly outputted when in a non-operational state.

## Architecture
The project architecture is depicted below, with each .v file corresponding to a module in the `Functional_TPU_Hardware_Specification_v0.1.1.md`. Note that the VGA_DEBUG folder is not a part of the main project or hardware spec but may be used to debug the TPU.

```
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
    └── VGA_CONTROLLER/
        ├── ascii_master_controller.v
        ├── clock_divider.v
        ├── vga_controller.v
        └── vga_driver.v
```

## Build Plan

See `Functional_TPU_Hardware_Specification_v0.1.1.md` for detailed descriptions of each module.

1. ~~Build and instantiate TPU and Programmer modules and MUX connections to the correct memory modules. Define the *trst* signal.~~
2. ~~Build and instantiate the Feeder module. Update the TPU module to route connections from the program memory to the feeder.~~
3. ~~Build and instantiate the Controller module. Update the TPU module to route connections from the feeder to the controller.~~
4. ~~Build Vector_Processor module and instantiate vector processors *a* and *b* in the TPU module. Update the TPU module to route connections between the vector processor, controller, and memory.~~
5. ~~Build and instantiate the Activator module. Update the TPU module to route input and control signals from vector processor *a* to the activator and output from the activator to the vector buffer, implementing intermediate combinational logic and quantization in the process.~~
6. ~~Build and instantiate the ALU module. Update the TPU module to appropriately route inputs and control signals from both vector processors to the ALU and outputs from the ALU to the vector buffer, implementing intermediate combinational logic and quantization in the process. Ensure that the *element_valid* signals of the two vector processors are properly connected to the *enable* signal of the ALU.~~ 
7. ~~Build and instantiate the Systolic_Array module and submodules (Systolic_Array_Input_Buffer and Multiply_Accumulate_Unit). Update the TPU module to route control signals from the controller to the systolic array.~~
8. ~~Test the entire TPU by~~
    1. ~~Creating a basic Functional TPU ISA program that uses each instruction at least once.~~
    2. ~~Formatting the program and relevant test weights into the Functional TPU message protocol.~~
    3. ~~Simulating programming over SPI and checking that each instruction functions as expected.~~
