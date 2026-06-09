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
        - Controller `/TPU/CONTROL/Controller.v`
        - TPU_0x1_Buffer `/TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v`
        - Weight_Memory `/TPU/MEMORY/WEIGHT_MEMORY/Weight_Memory.v`
        - Vector_Processor `/TPU/PROCESSING/Vector_Processor.v`
        - Activator `/TPU/PROCESSING/Activator.v`
        - ALU `/TPU/PROCESSING/ALU.v`
        - Vector_Buffer `/TPU/MEMORY/VECTOR_BUFFER/Vector_Buffer.v`
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
- Description: Top level module for the entire project. This module only exists as a wrapper for the TPU module so that the TPU module can interface with clock signals and IO from the FPGA itself. It also allows for easy modification of *clk* and *rst* signals for debugging.

### TPU
- Path to File: `/TPU/TPU.v`
- Description: Top level module for the TPU. In accordance with the module hierarchy, this module instantiates the majority of the other modules. This module also defines connection routing between its various submodules and serves as a passthrough for relevant signals (clk, rst, SPI Signals, etc.). 
    - **Global Clock and Reset:** The TPU module defines a gobal clock (*clk*) and reset (*rst*) signal. *clk* is the global system clock for the TPU (active HIGH on positive edge). *rst* is the global system reset signal (active LOW), and should effectively reset the entire TPU to it's default state while asserted LOW.
    - **Dual Vector Processors:** The TPU module instantiates two Vector_Processor modules, *a* and *b*. Vector processor *a* is connected to the *a* ports of the Weight_Memory and TPU_0x1_Buffer. Likewise, vector processor *b* is connected to the *b* ports.

### SPI_Slave
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v`
- Description: The SPI_Slave module is the connection point between the external SPI signals and TPU internals. This module is pulled from [nandland's SPI Slave](https://github.com/nandland/spi-slave) repository. Relevent documentation on the modules functionality can be found there.

### SPI_Input_buffer
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v`
- Description: The SPI_Input_buffer is a Quartus Prime IP FIFO memory module.
    - **Word Width:** 8 bits
    - **Number of Words:** 256
    - **Read and Write Request:** The module has a *rdreq* and *wrreq* signal to request reads and writes from and to memory respectively. Asserting either of them HIGH will either pop or push data from or to the buffer every clock cycle. As such, each signal should only be asserted high for a single clock cycle to read or write a single value.
    - **Full and Empty Signals:** The module also has *full* and *empty* signals which will be asserted high if the buffer is either full or empty respectively.
    - **Sychronous Clear:** *sclr* may be asserted high to flush the buffer of its contents.

### SPI_Interface
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v`
- Description: The SPI_Interface module instantiates and links together the SPI_Input_buffer and SPI_Slave modules via routing wires, and exposes the SPI connections and buffer outputs as module outputs.
    - **SPI and Buffer Communication:** When the SPI_Slave module recieves a full byte from the SPI input wires, it asserts a DATA VALID signal high for one clock cycle to indicate that the data output of the module is readable. This naturally meshes with the functionality of *wrreq* in the SPI_Input_buffer, and so the two modules can be directly wired together.
    - **Reset Signal:** The *sclr* signal of the SPI Input buffer is connected to the global reset *rst* so that the SPI_Input_buffer is flushed when *rst* is pulled low.
 
### Programmer
- Path to File: `/TPU/PROGRAMMER/Programmer.v`
- Description: This module writes data to the Program_Memory, Weight_Memory, and TPU_0x1_Buffer, effectively "programming" the TPU. The module defines a finite state machine that retrieves data byte by byte from the SPI input buffer (if data is available), decodes it according to the *Functional TPU Messaging Protocol* defined in `Functional_TPU_Message_Protocol_v0.1.md`, and writes it to the correct location in memory. It also instantiates the SPI_Interface module. 
    - **State Machine Flow:**
        - Checks for available bytes in the SPI input buffer via the passthrough from the SPI Interface.
        - If avialable, pops bytes from the buffer until a START code is found.
        - Decodes the next function code and subsequent values (if needed).
        - Writes relevent bytes (dependent on function code) to the correct location in memory.
        - Repeats the decode-write cycle until a STOP instruction is recieved.
        - Returns to searching for a START instruction.
    - **Error States:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior. The state machine enters a COM_ERROR state if it attempts to decode an invalid instruction.
    - **PROGRAM Signal:** The module defines an output signal *program* (active LOW). When the START code is recieved, the module asserts *program* LOW. The *program* signal is connected via an OR gate with the global reset signal *rst*, and the resulting signal *trst* is passed to the following modules such that either *rst* or *program* can hold them in a default state: Feeder, Systolic_Array, Controller, Vector_Processor (*a* and *b*), Activator, ALU, and Vector_Buffer. This is to ensure that the TPU internals cannot modify device memory while programming occurs. Likewise, the Programmer module must have full control over all device memory while the *program* signal is low. Once the module decodes a STOP code, *program* is asserted HIGH and normal device operation resumes.
    - **Program Memory:** When initially writing to the program memory, the Programmer should start from address 0x0. Subsequent writes to program memory should increment the address by 1 such that instructions are stored contiguously.
    - **Dual-Port RAM Access:** While programming the Weights_Memory and TPU_0x1_Buffer, the Programmer uses the port *a*.

### Program_Memory
- Path to File: `/TPU/PROGRAMMER/PROGRAM_MEMORY/Program_Memory.v`
- Description: The Program_Memory is a Quartus IP 1-port RAM module. It stores 128-bit instructions defined by `Functional_TPU_ISA_v0.2.md`.
    - **Word Width:** 128 bits
    - **Number of Words:** 1024
    - **Write and Read Behavior:** The modules output port *q* always outputs the data present at the modules *address* input. When *wren* is asserted high, *q* is undefined and the value stored at *address* is overwritten by the *data* input.

### Weight_Memory
- Path to File: `/TPU/MEMORY/WEIGHT_MEMORY/Weight_Memory.v`
- Description: The Weight_Memory is a Quartus IP 2-port RAM module. It serves as the main memory of the TPU and stores 8-bit neural network weights.
    - **Word Width:** 8 bits
    - **Number of Words:** 65536
    - **Dual Port:** The Weight_Memory is dual port, meaning it has two seperate sets of identical control signals, *a* and *b*.
    - **Write and Read Behavior:** The modules output ports *q_a* and *q_b* always output the data present at *address_a* and *address_b* inputs. When *wren_a* or *wren_b* is asserted high, *q_a* or *q_b* is undefined and the value stored at *address_a* or *address_b* is overwritten by the *data_a* or *data_b* input respectively.

### TPU_0x1_Buffer
- Path to File: `/TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v`
- Description: The TPU_0x1_Buffer is a Quartus IP 1-port RAM module. It functions as the unique isolated memory required for address 0x1 in `Functional_TPU_ISA_v0.2.md`.
    - **Word Width:** 8 bits
    - **Number of Words:** 65536
    - **Dual Port:** The 0x1 buffer is dual port, meaning it has two seperate sets of identical control signals, *a* and *b*.
    - **Write and Read Behavior:** The modules output ports *q_a* and *q_b* always output the data present at *address_a* and *address_b* inputs. When *wren_a* or *wren_b* is asserted high, *q_a* or *q_b* is undefined and the value stored at *address_a* or *address_b* is overwritten by the *data_a* or *data_b* input respectively.

### Feeder
- Path to File: `/TPU/PROGRAMMER/Feeder.v`
- Description: The Feeder module defines a finite state machine that retrieves data from Program_Memory and forwards it to the Controller module. This effectively emulates the FETCH step in a traditional CPU.
    - **Program Counter:** The module contains a 10-bit program counter register (*pc*) that stores the address of the current instruction. When *trst* is asserted low *pc* should be set to zero.
    - ***controller_start* signal:** The module defines a *controller_start* signal (active HIGH) that connects to the the Controller. The Feeder asserts it HIGH for one clock cycle to indicate to the Controller that it may begin operations.
    - **State Machine Flow:**
        - Retrieve an instruction from Program_Memory at address=*pc*.
        - Forward the instruction to the Controller.
        - Assert *controller_start* HIGH for one clock cycle.
        - Wait for a *controller_idle* signal (active HIGH) from the Controller.
        - When *controller_idle* is asserted HIGH increment the program counter and repeat.
    - **Error State:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior.

### Controller
- Path to File: `/TPU/CONTROL/Controller.v`
- Description: The Controller module is the "CPU" of the TPU. Combined with the Feeder module, the two implement the traditional CPU instruction cycle (Fetch-Decode-Execute-Writeback). The Controller module defines a finite state machine that converts instructions from the Feeder into control signals and manages timing and dataflow between other modules. Valid instructions and their formats are defined in `Functional_TPU_ISA_v0.2.md`.
    - ***vector_start* Signals:** The Controller module defines two signals, *vector_start_a* and *vector_start_b* (active HIGH) that connect to vector processor a and b respectively. The Controller asserts them HIGH for one clock cycle to indicate to each Vector_Processor that it may begin operations.
    - ***controller_idle* Signal:** The Controller module defines a *controller_idle* signal (active HIGH) that indicates that the Controller is ready to recieve another instruction.
    - **State Machine Flow:** 
        - Wait for the Feeder to assert *controller_start* HIGH.
        - When the Feeder asserts *controller_start* HIGH, assert *controller_idle* LOW and continue.
        - Combinationally decode instruction from the Feeder and send control signals to other modules. (Decode)
        - Assert *vector_start* signals HIGH for one clock cycle. (Execute)
        - Wait for vector processors to assert *vector_idle* signals HIGH. (Execute)
        - Set writeback control signals for Vector_Processors. (Writeback)
        - Assert *vector_start* signals HIGH for one clock cycle. (Writeback)
        - Wait for vector processors to assert *vector_idle* signals HIGH. (Writeback)
        - Assert *controller_idle* signal HIGH.
        - Repeat.
    - **Error States:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior. The state machine enters a DECODE_ERROR state if it decodes an invalid instruction.

### Vector_Processor
- Path to File: `/TPU/PROCESSING/Vector_Processor.v`
- Description: The Vector_Processor module defines a state machine that handles memory accesses, data routing to and from the Activator and Systolic Array, and formatting of data.
    - **Sources and Destinations:** The Vector_Processor modules primary function is to move vectors/matrices from a source to a destination.
        - **Sources:**
            - Device memory address
            - Systolic array output
            - Vector buffer output
        - **Destinations:**
            - Device memory address
            - Systolic array input buffer a
            - Systolic array input buffer b
            - Activator input
            - ALU input a
            - ALU input b
    - **ISA Complience:** The Vector_Processor module formats data written to the Weight_Memory and TPU_0x1_Buffer in accordance with the `Functional_TPU_ISA_v0.2.md`. Likewise, it expects all data read from memory to be formatted in accordance with the ISA. As a result, the Vector_Processor handles translation of data from flattened arrays in memory to matrices when loading data to the systolic array input buffers and from matrices to flattened arrays when reading from the systolic array outputs and writing back to memory.
    - **Control Signals:** To remain ISA complient, the Vector_Processor module has various control signal inputs that it uses to interpret data.
        - **Dimensionality:** When interfacing with the systolic array, the vector processor checks the *dim0* and *dim1* inputs to determe the dimensionality (*dim0* x *dim1*) of the matrix its processing, and then either reads from the systolic array outputs or writes to the systolic array input buffers in the proper order.
        - **Length:** When interfacing with modules other than the systolic array, the Vector_Processor module checks the *length* input to determine how many elements should be processed.
        - **Memory Addressing:** The Vector_Processor module defines a *source_address* and *dest_address* input for reading from or writing to memory respectively. Combined with the *length*, *dim0*, and *dim1* signals, the vector processor determines the exect number of elements to read or write as well as their order.
        - ***vector_idle* Signal:** The Vector_Processor module defines a *vector_idle* signal (active HIGH) that indicates that the Vector_Processor has finished its control loop and is waiting for a new instruction.
        - **ALU/Activator Signals:**
        - **Systolic Array Signals:**
    - **State Machine Flow:**
        - Wait for the Controller to assert *vector_start* HIGH.
        - When the Controller asserts *vector_start* HIGH, assert *vector_idle* LOW.
        - Decode control signals from Controller.
        - Move data from source to destination using a loop.
        - When the loop finishes, assert *vector_idle* HIGH and repeat.
    - **Error State:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior.

### Activator
- Path to File: `/TPU/PROCESSING/Activator.v`
- Description: The Activator module implements the ACT instruction functions from the `Functional_TPU_ISA_v0.2.md`. 
    - **ALU-Like Behavior:** The Activator module is effectively a specialized Arithemetic Logic Unit (ALU) that implements operations specified by the ISA. 
    - **Control:** The Controller module sets the Activator function based on instruction. If the current instruction does not use the Activator, the Controller sets the Activator function to NO OP.
    - ***enable* Signal:** The Activator module defines an *enable* input signal (active HIGH). While *enable* is asserted HIGH, every clock cycle the activator takes an input value, applies a function to it based on a control signal, and writes the result to the vector buffer. As a result, the activator can process a single value by asserting *enable* HIGH for one clock cycle.
    

### ALU

### Multiply_Accumulate_Unit
- Path to File: `/TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v`
- Description: The Multiply_Accumulate_Unit module is the basic building block of the systolic array.
    - **Data Signals:** The Multiply_Accumulate_Unit module has two data inputs *a_in* and *b_in* and three data outputs *a_out*, *b_out* and *c*. All *a* and *b* signals are 8-bit, while *c* is 32-bit.
    - **Operations:** The Multiply_Accumulate_Unit module implements two operations: addition and multiply-accumulate. a single-bit control signal *op* determines the operation.
        - **Additon:** every clock cycle c=a+b, a_out=a_in, and b_out=b_in

### Systolic_Array








