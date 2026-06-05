# Functional TPU - Hardware Specification
> **Purpose:** This document outlines the Functional TPU hardware specification, including module instantiation hierarchy and functional descriptions of modules.

## Module Instantiation Hierarchy
Each Verilog module is instantiated following the heirarchy below. Each entry in the heirarchy has first the name of the module, and then the path to the .v file that defines it.

- Tensor_Processing_Unit `/Tensor_Processing_Unit.v`
    - TPU `/TPU/TPU.v`
        - Programmer `/TPU/PROGRAMMER/Programmer.v`
            - SPI_Interface `/TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v`
                - SPI_Slave `/TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v`
                - SPI_Input_buffer `/TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v`
        - Program_Memory `/TPU/PROGRAMMER/PROGRAM_MEMORY/Program_Memory.v`
        - Feeder `/TPU/PROGRAMMER/Feeder.v`
        - TPU_0x1_Buffer `/TPU/MEMORY/TPU_0x1_Buffer.v`
        - Weight_Memory `/TPU/MEMORY/Weight_Memory.v`
        - Vector_Processor `/TPU/PROCESSING/Vector_Processor.v`
        - Controller `/TPU/CONTROL/Controller.v`
        - Activator `/TPU/PROCESSING/Activator.v`
        - Systolic_Array `/TPU/SYSTOLIC_ARRAY/Systolic_Array.v`
            - Multiply_Accumulate_Unit `/TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v`
            - Systolic_Array_Input_Buffer `/TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v`

Note that the folder heirarchy is structured differently from the module instantiation hierarchy. This is intentional and aims to organize files in a way that improves readability. Additionally, the VGA_DEBUG module is not currently used in the build, as it requires lots of memory.

## Module Descriptions
Listed below are descriptions of each Verilog module following the general path of data.

### 
- Path to File:
- Description:

### Tensor_Processing_Unit
- Path to File: `/Tensor_Processing_Unit.v`
- Description: Top level module for the entire project. This module only exists as a wrapper for the TPU module so that the TPU module can interface with clock signals and IO from the FPGA itself. It also allows for easy modification of CLK and RST signals for debugging.

### TPU
- File Path to File: `/TPU/TPU.v`
- Description: Top level module for the TPU. In accordance with the module hierarchy, this module instantiates the majority of the other modules. This module also defines connection routing between its various submodules and serves as a passthrough for relevant signals (CLK, RST, SPI Signals, etc.). 
    - **Global Clock and Reset:** The TPU module passes through a clock (CLK) and reset (RST) signal to all of it's submodules. CLK is the global system clock for the TPU (active HIGH on positive edge). RST is the global system reset signal (active LOW), and should effectively reset the entire TPU to it's default state while asserted LOW.

### SPI_Slave
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v`
- Description: The SPI Slave module is the connection point between 

### Programmer
- Path to File: `/TPU/PROGRAMMER/Programmer.v`
- Description: This module writes data to the program memory, weights memory, and 0x1 buffer, effectively "programming" the TPU. The module itself defines a finite state machine that retrieves data byte by byte from the SPI input buffer (if data is available), decodes it according to the Functional TPU Memory ISA defined in `Functional_TPU_Memory_ISA_v0.1.md`, and writes it to the correct location in memory. It also instantiates the SPI_Interface module. 
    - **Control Flow:**
        - Checks for available bytes in the SPI input buffer via the passthrough from the SPI Interface.
        - If avialable, pops bytes from the buffer until a START instruction is found.
        - Decodes the next instruction 
        - Writes the next N bytes (dependent on memory instruction) to the correct location in memory.
        - Repeats the decode-write cycle until a STOP instruction is recieved.
        - Returns to searching for a START instruction.
    - **Error States:** The state machine defaults to an ERROR state as a defualt in the event of undefined state machine behavior. The state machine enters a MEM_ERROR state if it attempts to decode an invalid instruction.
    - **PROGRAM Signal:** The module defines an output signal PROGRAM (active LOW). When the START instruction is recieved, the module asserts PROGRAM LOW. The program signal is connected via an OR gate with the global reset signal RST, and the resulting signal is passed to the following modules such that either RST or PROGRAM can hold them in a default state: Feeder, Systolic_Array, Controller, Vector_Processor, and Activator. This is to ensure that the TPU internals cannot modify device memory while programming occurs. Likewise, the Programmer module must have full control over all device memory while the PROGRAM signal is low. Once the module decodes a STOP instruction, PROGRAM is asserted HIGH and normal device operation resumes.



