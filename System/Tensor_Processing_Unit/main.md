# Functional TPU - Complete Build Specification
> **Purpose:** This doucment is the single source of truth for the Functional TPU project. It defines the project identity, current state, architecture, and full incremental build plan.

## Project Identity

- **Name:** Functional TPU
- **Version:** 0.1
- **Language:** Verilog HDL
- **Platform:** Terasic DE1-SoC FPGA
- **License:** MIT License
- **Change Log:** `log.md`
- **Instruction Set Architecture:** `Functional_TPU_ISA_v0.1.md`

## General Project Description
The Functional TPU is a tensor processing unit roughly based on Google's first generation TPU. This project aims to create Verilog HDL code that implements the instruction set detailed in `Functional_TPU_ISA_v0.1.md` in order to accelerate small neural networks.

**Design Goals:**
- Systolic Array for matrix multiplication.
- Storage of tensors as flattened arrays.
- Tensor flattening and unflattening module.
- Seperate memories for program instructions, weights/biases, and temporary data.
- Dynamic programming of TPU through SPI link.

## Current State (v0.1 - What's Working)
- **SPI Link:** `TPU_SPI_INTERFACE/SPI_Slave.v` and `TPU_SPI_INTERFACE/SPI_INPUT_BUFFER/SPI_Input_buffer.v` are correctly wired together by `TPU_SPI_INTERFACE/SPI_Interface.v` which is currently the top level module for the `TPU_SPI_INTERFACE/` subdirectory. The SPI input buffer has been verified to properly recieve and store data from the SPI slave module.
- **VGA Debugging:** `TPU_VGA_DEBUG/Debug.v` can be successfully used to display 32 bit integers as hexidecimal values on a monitor connected to the FPGA via it's VGA port.

## Architecture

