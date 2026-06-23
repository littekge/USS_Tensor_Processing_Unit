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

