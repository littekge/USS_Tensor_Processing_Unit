# Functional TPU - Hardware Specification
> **Purpose:** This document outlines the Functional TPU hardware specification, including module instantiation hierarchy and functional descriptions of modules.

## Module Instantiation Hierarchy
Each Verilog module is instantiated following the hierarchy below. Each entry in the hierarchy has first the name of the module, and then the path to the .v file that defines it.

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

Note that the folder hierarchy is structured differently from the module instantiation hierarchy. This is intentional and aims to organize files in a way that improves readability. Additionally, the VGA_DEBUG module is not currently used in the build, as it requires lots of memory.

## Module Descriptions
Listed below are descriptions of each Verilog module following the general path of data.

### Tensor_Processing_Unit
- Path to File: `/Tensor_Processing_Unit.v`
- Description: Top level module for the entire project. This module only exists as a wrapper for the TPU module so that the TPU module can interface with clock signals and IO from the FPGA itself. It also allows for easy modification of *clk* and *rst* signals for debugging.

### TPU
- Path to File: `/TPU/TPU.v`
- Description: Top level module for the TPU. In accordance with the module hierarchy, this module instantiates the majority of the other modules. This module also defines connection routing between its various submodules and serves as a passthrough for relevant signals (clk, rst, SPI Signals, etc.).
    - **Synchronous Control Signals:**
        - **Inputs:** 
            - ***clk***: Global clock (*clk*) input signal. *clk* is the global system clock for the TPU (active HIGH on positive edge). *clk* is passed to all other modules.
            - ***rst***: Global reset (*rst*) input signal (active LOW). *rst* is the global system reset signal, and effectively resets the entire TPU to it's default state while asserted LOW.
    - **Dual Vector Processors:** The TPU module instantiates two Vector_Processor modules, a primary processor *a* and a secondary processor *b*. Relevant connections are described below.
        - **Memory Connections:** During normal operation vector processor *a* is connected to the *a* ports of the Weight_Memory and TPU_0x1_Buffer. Likewise, vector processor *b* is connected to the *b* ports.
        - **ALU:**: Each vector processor handles one input to the ALU.
        - **ALU Synchronization:** The *element_valid* signals from the vector processors are connected by an AND gate whose output is then used as the *enable* signal for the ALU. This ensures that data is written to the vector buffer if and only if both vector processors are properly synchronized and the data sent to both inputs of the ALU is valid.
        - **Activator:** *a* handles input to the activator. The *element_valid* signal from *a* is connected to the *enable* input of the activator to ensure that data is properly processed through the activator and written to the vector buffer.
        - **Vector Buffer:** *a* handles accesses to the output of the vector buffer
        - **Systolic Array Buffers:** *a* handles input to the top side systolic array input buffers and *b* handles input to the left side input buffers.
        - **Systolic Array Output:** *a* handles output from the systolic array.
    - **Separate TPU Reset:** The *program* signal from the programmer is connected via an AND gate to the global reset *rst*, and the resultant signal *trst* is passed to the following modules such that either *rst* or *program* can hold them in a default state: Feeder, Systolic_Array, Controller, Vector_Processor (*a* and *b*), Activator, and ALU. This is to ensure that the TPU internals cannot modify device memory while programming occurs. 
    - **Memory MUX:** The Programmer module has full control over all device memory while the *program* signal is low. To implement this, the *data*, *address*, and *wren* inputs to the program memory as well as the *data_a*, *address_a*, and *wren_a* inputs to the weight memory and 0x1 buffer are connected to the programmer, feeder, and vector processor *a* using a MUX. The programmer controls every input when *program* is LOW. When *program* is HIGH, vector processor *a* controls the *wren_a*, *data_a*, and *address_a* inputs of the 0x1 buffer and weight memory and the feeder controls the *wren*, *data*, and *address* inputs to the program memory.
    - **Vector Buffer Clear:** The module defines a *clear* signal (active HIGH) used to flush the vector buffer. *clear* is ORed with inverted *trst* and passed to the *sclr* input of the vector buffer so that either *trst* going LOW or *clear* going HIGH will flush the buffer.
  
### SPI_Slave
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v`
- Description: The SPI_Slave module is the connection point between the external SPI signals and TPU internals. This module is pulled from [nandland's SPI Slave](https://github.com/nandland/spi-slave) repository. Relevant documentation on the modules functionality can be found there.

### SPI_Input_buffer
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_INPUT_BUFFER/SPI_Input_buffer.v`
- Description: The SPI_Input_buffer is a Quartus Prime IP FIFO memory module.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***rdreq***: Read request. Asserting *rdreq* HIGH will pop data from the buffer every clock cycle to the *q* module output. As such, *rdreq* should only be asserted HIGH for a single clock cycle to pop a single value.
            - ***wrreq***: Write request. Asserting *wrreq* HIGH will push the data from the *data* input to the buffer every clock cycle. As such, *wrreq* should only be asserted HIGH for a single clock cycle to push a single value.
            - ***sclr***: Flushes the buffer of its contents on the next positive clock edge when asserted HIGH.
        - **Outputs**
            - ***full***: HIGH when the buffer is full.
            - ***empty***: HIGH when the buffer is empty.
    - **Word Width:** 8 bits
    - **Depth:** 256 words

### SPI_Interface
- Path to File: `/TPU/PROGRAMMER/SPI_LINK/SPI_Interface.v`
- Description: The SPI_Interface module instantiates and links together the SPI_Input_buffer and SPI_Slave modules via routing wires, and exposes the SPI connections and buffer outputs as module outputs.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***rst***: Module reset signal (active LOW).
    - **SPI and Buffer Communication:** When the SPI_Slave module receives a full byte from the SPI input wires, it asserts a data valid signal high for one clock cycle to indicate that the data output of the module is readable. This naturally meshes with the functionality of *wrreq* in the SPI_Input_buffer, and so the two modules are directly wired together.
    - **SPI Buffer Reset Signal:** *rst* is inverted and connected to the SPI input buffers *sclr* such that the buffer is cleared when *rst* is pulled low.
 
### Programmer
- Path to File: `/TPU/PROGRAMMER/Programmer.v`
- Description: This module writes data to the Program_Memory, Weight_Memory, and TPU_0x1_Buffer, effectively "programming" the TPU. The module defines a finite state machine that retrieves data byte by byte from the SPI input buffer (if data is available), decodes it according to the *Functional TPU Messaging Protocol* defined in `Functional_TPU_Message_Protocol_v0.1.md`, and writes it to the correct location in memory. It also instantiates the SPI_Interface module. 
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***rst***: Module reset signal (active LOW).
        - **Outputs:**
            - ***program***: Indicates that the module is currently programming the device (active LOW).
    - **State Machine Flow:**
        - Check for available bytes in the SPI input buffer via the passthrough from the SPI Interface.
        - If available, pop bytes from the buffer until a START code is found.
        - Assert *program* LOW when a start code is found.
        - Decode the next function code and subsequent values (if needed).
        - Write relevant bytes (dependent on function code) to the correct location in memory.
        - Repeat the decode-write cycle until a STOP code is received.
        - Assert *program* HIGH when a STOP code is received.
        - Return to checking for a START code.
    - **Error States:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior. The state machine enters a COM_ERROR state if it attempts to decode an invalid instruction.
    - **Program Memory:** When initially writing to the program memory, the programmer starts from address 0x0. Subsequent writes to program memory should increment the address by 1 such that instructions are stored contiguously.
    - **Device Memory:** When writing to the weight memory or 0x1 buffer, the programmer uses port *a*.

### Program_Memory
- Path to File: `/TPU/PROGRAMMER/PROGRAM_MEMORY/Program_Memory.v`
- Description: The Program_Memory is a Quartus IP 1-port RAM module. It stores 128-bit instructions defined by `Functional_TPU_ISA_v0.1.md`.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***wren***: Writes data from the *data* input to a location in memory defined by the *address* input when asserted HIGH.
    - **Word Width:** 128 bits
    - **Depth:** 1024 words
    - **Write and Read Behavior:** The modules output port *q* always outputs the data present at the modules *address* input. When *wren* is asserted high, *q* is undefined and the value stored at *address* is overwritten by the *data* input.

### Weight_Memory
- Path to File: `/TPU/MEMORY/WEIGHT_MEMORY/Weight_Memory.v`
- Description: The Weight_Memory is a Quartus IP 2-port RAM module. It serves as the main memory of the TPU and stores 8-bit neural network weights.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***wren_a***: Writes data from the *data_a* input to a location in memory defined by the *address_a* input when asserted HIGH.
            - ***wren_b***: Writes data from the *data_b* input to a location in memory defined by the *address_b* input when asserted HIGH.
    - **Word Width:** 8 bits
    - **Depth:** 65536 words
    - **Dual Port:** The Weight_Memory is dual port, meaning it has two separate sets of identical control signals, *a* and *b*.
    - **Write and Read Behavior:** The modules output ports *q_a* and *q_b* always output the data present at *address_a* and *address_b* inputs. When *wren_a* or *wren_b* is asserted high, *q_a* or *q_b* is undefined and the value stored at *address_a* or *address_b* is overwritten by the *data_a* or *data_b* input respectively.

### TPU_0x1_Buffer
- Path to File: `/TPU/MEMORY/0X1_BUFFER/TPU_0x1_Buffer.v`
- Description: The TPU_0x1_Buffer is a Quartus IP 2-port RAM module. It functions as the unique isolated memory required for address 0x1 in `Functional_TPU_ISA_v0.1.md`.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***wren_a***: Writes data from the *data_a* input to a location in memory defined by the *address_a* input when asserted HIGH.
            - ***wren_b***: Writes data from the *data_b* input to a location in memory defined by the *address_b* input when asserted HIGH.
    - **Word Width:** 8 bits
    - **Depth:** 65536 words
    - **Dual Port:** The 0x1 buffer is dual port, meaning it has two separate sets of identical control signals, *a* and *b*.
    - **Write and Read Behavior:** The modules output ports *q_a* and *q_b* always output the data present at *address_a* and *address_b* inputs. When *wren_a* or *wren_b* is asserted high, *q_a* or *q_b* is undefined and the value stored at *address_a* or *address_b* is overwritten by the *data_a* or *data_b* input respectively.

### Feeder
- Path to File: `/TPU/PROGRAMMER/Feeder.v`
- Description: The Feeder module defines a finite state machine that retrieves data from Program_Memory and forwards it to the Controller module. This effectively emulates the FETCH step in a traditional CPU.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
    - **Program Counter:** The module contains a 10-bit program counter register (*pc*) that stores the address of the current instruction. When *trst* is asserted low *pc* should be set to zero.
    - **Stop Condition:** The module stops feeding instructions when a *syscall* instruction is retrieved from memory. 
    - **State Machine Flow:**
        - Retrieve an instruction from Program_Memory at address=*pc*.
        - If the instruction is *syscall*, enter a DONE state until *trst* resets the module.
        - Else forward the instruction to the controller.
        - Assert the controllers *controller_start* signal HIGH for one clock cycle.
        - Wait for the controller to assert *controller_idle* HIGH.
        - When the controller asserts *controller_idle* HIGH increment the program counter and repeat.
    - **Error State:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior.

### Controller
- Path to File: `/TPU/CONTROL/Controller.v`
- Description: The Controller module is the "CPU" of the TPU. Combined with the Feeder module, the two implement the traditional CPU instruction cycle (Fetch-Decode-Execute-Writeback). The Controller module defines a finite state machine that converts instructions from the Feeder into control signals and manages timing and dataflow between other modules. Valid instructions and their formats are defined in `Functional_TPU_ISA_v0.1.md`.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***controller_start***: Indicates that the controller should begin operations when asserted HIGH for one clock cycle.
        - **Outputs:**
            - ***controller_idle***: Asserted HIGH when the controller is ready to receive a new instruction.
            - ***clear***: Asserted HIGH to clear buffers between each operation.
    - **Dual Vector Processors:** The TPU has two vector processors. The controller has two sets of identical control signals to interface with each vector processor. The controller leverages both processors to perform simultaneous operations whenever possible (eg. simultaneous loading of systolic array input buffers or simultaneous streaming of data to the two inputs of the ALU). When a simultaneous operation is performed, the controller starts both vector processors at the same clock cycle to ensure synchronization. 
    - **State Machine Flow:** 
        - Wait for the Feeder to assert *controller_start* HIGH.
        - When the Feeder asserts *controller_start* HIGH, assert *controller_idle* LOW and continue.
        - Assert *clear* signals for the vector buffer (via *clear* in the TPU module), ALU, activator, and systolic array HIGH for one clock cycle to clear all residual data from the previous operation.
        - Combinationally decode instruction from the Feeder and send control signals to other modules. (Decode)
        - Assert the vector processors *vector_start* signals HIGH for one clock cycle. (Execute)
        - Wait for vector processors to finish execute operations. (Execute)
        - If the instruction uses the systolic array:
            - Assert the arrays *systolic_array_start* signal HIGH for one clock cycle. (Execute)
            - Wait for the systolic array to finish execute operations. (Execute)
        - Set writeback control signals for Vector_Processors. (Writeback)
        - Assert the vector processors *vector_start* signals HIGH for one clock cycle. (Writeback)
        - Wait for vector processors to finish writeback operations. (Writeback)
        - Assert *controller_idle* signal HIGH.
        - Repeat control loop.
    - **Error States:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior. The state machine enters a DECODE_ERROR state if it decodes an invalid instruction.

### Vector_Processor
- Path to File: `/TPU/PROCESSING/Vector_Processor.v`
- Description: The Vector_Processor module defines a state machine that handles memory accesses, data routing to and from the Activator and Systolic Array, and proper formatting of data.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***vector_start*** Indicates that the vector processor should begin operations when asserted HIGH for one clock cycle.
        - **Outputs:**
            - ***vector_idle***: Asserted HIGH when the vector processor has finished an operation.    
            - ***element_valid***: Asserted HIGH for one clock cycle when the modules data route from source to destination is valid and the resultant data transfer will be successful.
    - **ISA compliance:** The Vector_Processor module formats data written to the weight memory and 0x1 buffer in accordance with the `Functional_TPU_ISA_v0.1.md`. Likewise, it expects all data read from memory to be formatted in accordance with the ISA. As a result, the Vector_Processor handles translation of data from flattened arrays in memory to matrices when loading data to the systolic array input buffers and from matrices to flattened arrays when reading from the systolic array outputs and writing back to memory.
    - **Memory Abstraction:** The Vector_Processor module abstracts the weight memory and 0x1 buffer into a single address space as specified in the ISA.
    - **Asynchronous Control Signal Notes:** To remain ISA compliant, the Vector_Processor module has various asynchronous control signal inputs and output that it uses to interpret data.
        - **Dimensionality:** When interfacing with the systolic array, the vector processor checks the *dim0* and *dim1* inputs to determine the dimensionality (*dim0* x *dim1*) of the matrix its processing, and then either reads from the systolic array outputs or writes to the systolic array input buffers in the proper order.
        - **Length:** When interfacing with modules other than the systolic array, the Vector_Processor module checks the *length* input to determine how many elements should be processed.
        - **Memory Addressing:** The Vector_Processor module defines a *mem_source_address* and *mem_dest_address* input for reading from or writing to memory respectively. Combined with the *length*, *dim0*, and *dim1* signals, the vector processor determines the exact number of elements to read or write as well as their order.
        - **Source/Destination Address:** The Vector_Processor module defines a *vect_source_address* and *vect_dest_address* input that specify where the vector processor should source data from and where it should send it to. Valid sources and destinations are listed below.
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
    - **State Machine Flow:**
        - Wait for the Controller to assert *vector_start* HIGH.
        - When the Controller asserts *vector_start* HIGH, assert *vector_idle* LOW.
        - Combinationally decode control signals from the controller.
        - In a loop until all elements have been processed:
            - Set data path and memory addresses.
            - Assert *element valid* HIGH for one clock cycle when the data path is valid (accounting for RAM latency in the process if necessary).
        - When the loop finishes, assert *vector_idle* HIGH and repeat.
    - **Error State:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior.
    - **Requantization:** When reading from the systolic array outputs, the Vector_Processor module requantizes the output value before writing to its destination.

### Activator
- Path to File: `/TPU/PROCESSING/Activator.v`
- Description: The Activator module implements the ACT instruction functions from the `Functional_TPU_ISA_v0.1.md`. 
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***enable***: Enables the activator (active HIGH)
            - ***clear***: Clears residual values from the activators input and intermediate registers when asserted HIGH.
    - **ALU-Like Behavior:** The Activator module is effectively a specialized arithmetic Logic Unit (ALU) that implements operations specified by the ISA. 
    - **Control:** The Controller module sets the Activator function based on instruction. If the current instruction does not use the Activator, the Controller sets the Activator function to NO OP.
    - **Enable Behavior:** While *enable* is asserted HIGH, every clock cycle the activator takes an input value, applies a function to it based on a control signal, and writes the result to the vector buffer. As a result, the activator can process a single value by asserting *enable* HIGH for one clock cycle.
    
### ALU
- Path to File: `/TPU/PROCESSING/ALU.v`
- Description: The ALU module implements ELEM instructions from the `Functional_TPU_ISA_v0.1.md`.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***enable***: Enables the ALU (active HIGH)
            - ***clear***: Clears residual values from the ALUs input and intermediate registers when asserted HIGH.
    - **Standard ALU:** The ALU module functions as a standard ALU with two input values and one output value. 
    - **Control:** The Controller module sets the ALU function based on instruction. If the current instruction does not use the ALU, the Controller sets the ALU function to NO OP.
    - **Enable Behavior:** While *enable* is asserted HIGH, every clock cycle the ALU takes two input values, performs a arithmetic or logical operation on them, and writes the result to the vector buffer. As a result, the ALU can process a single value by asserting *enable* HIGH for one clock cycle.
    - **Requantization:** For arithmetic operations, the ALU module requantizes the output before writing to the vector buffer.
 
### Vector_Buffer
- Path to File: `/TPU/MEMORY/VECTOR_BUFFER/Vector_Buffer.v`
- Description: The Vector_Buffer module is a Quartus Prime IP FIFO memory module. It temporarily stores outputs from the ALU or activator.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***rdreq***: Read request. Asserting *rdreq* HIGH will pop data from the buffer every clock cycle to the *q* module output. As such, *rdreq* should only be asserted HIGH for a single clock cycle to pop a single value.
            - ***wrreq***: Write request. Asserting *wrreq* HIGH will push the data from the *data* input to the buffer every clock cycle. As such, *wrreq* should only be asserted HIGH for a single clock cycle to push a single value.
            - ***sclr***: Flushes the buffer of its contents on the next positive clock edge when asserted HIGH.
        - **Outputs**
            - ***full***: HIGH when the buffer is full.
            - ***empty***: HIGH when the buffer is empty.
    - **Word Width:** 8 bits
    - **Depth:** 65536 words

### Multiply_Accumulate_Unit
- Path to File: `/TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v`
- Description: The Multiply_Accumulate_Unit module is the basic building block of the systolic array.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***clear***: Clears values from all registers in the MAC.
    - **Data Signals:** The Multiply_Accumulate_Unit module has two data inputs *a_in* and *b_in* and three data outputs *a_out*, *b_out* and *c*.
    - **Operation:** The Multiply_Accumulate_Unit module implements a single operation: multiply-accumulate. Every clock cycle the module performs *c* = *c* + *a_in* * *b_in*, sets *a_out* = *a_in*, and sets *b_out* = *b_in*.

### Systolic_Array_Input_Buffer
- Path to File: `/TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v`
- Description: The Systolic_Array_Input_Buffer is a variable-depth FIFO buffer.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clock***: Module input clock.
            - ***rdreq***: Read request. Asserting *rdreq* HIGH will pop data from the buffer every clock cycle to the *q* module output. As such, *rdreq* should only be asserted HIGH for a single clock cycle to pop a single value.
            - ***wrreq***: Write request. Asserting *wrreq* HIGH will push the data from the *data* input to the buffer every clock cycle. As such, *wrreq* should only be asserted HIGH for a single clock cycle to push a single value.
            - ***sclr***: Flushes the buffer of its contents on the next positive clock edge when asserted HIGH.
        - **Outputs**
            - ***full***: HIGH when the buffer is full.
            - ***empty***: HIGH when the buffer is empty.
    - **Word Width:** 8 bits
    - **Depth:** Variable by input parameter (Depth=N words).

### Systolic_Array
- Path to File: `/TPU/SYSTOLIC_ARRAY/Systolic_Array.v`
- Description: The Systolic_Array module instantiates the multiply accumulate units (MACs) and systolic array input buffers and wires them together, implementing an output-stationary systolic array. It also defines a state machine that loads data from the systolic array input buffers to the multiply accumulate units in the correct order.
    - **Synchronous Control Signals:**
        - **Inputs:**
            - ***clk***: Module input clock.
            - ***trst***: Module reset signal (active LOW).
            - ***clear***: Clears all values from the MACs and flushes all systolic input buffers when asserted HIGH.
            - ***systolic_array_start***: Indicates that the systolic array should begin operations when asserted HIGH for one clock cycle.
        - **Outputs:**
            - ***systolic_array_idle***: Indicates that the systolic array is not performing an operation (active HIGH).
    - **Parameterization:** The module uses an input parameter N to determine the size of the systolic array and systolic array input buffers.
    - **Systolic Array Definition:** The module instantiates an NxN grid of multiply accumulate units using Verilog generate statements. For the purpose of tiling, the Systolic_Array module treats each multiply accumulate unit as follows: *a_in* enters from the top, *b_in* enters from the left, *a_out* leaves from the bottom, *b_out* leaves from the right, and *c* is an externally accessible register.
    - **Input Buffers:** The module also instantiates two sets of N systolic array input buffers of depth N on the left and top sides of the systolic array (again using generate statements). The outputs of the top side buffers are connected to the *a_in* inputs of the multiply accumulate units on the top edge of the systolic array. Likewise, the left side buffers are connected to the *b_in* inputs of the MACs on the left edge of the systolic array. The buffers on the left side are numbered from 0 to N-1, starting from the top and the buffers on the top side are numbered from 0 to N-1 starting from the left.
    - **Clear Behavior:** The module passes the *clear* input signal directly to the *clear* input signal of every MAC. The module ORs *clear* and inverted *trst* together and passes it to the *sclr* input of every systolic array input buffer so that either *trst* going LOW or *clear* going HIGH will flush the buffers of their contents.
    - **State Machine Flow:**
        - Wait for *systolic_array_start* to be asserted HIGH.
        - Assert *systolic_array_idle* LOW.
        - Loop through each input buffer on the top and left side from 0 to N-1, asserting the *rdreq* signal of the Mth buffers HIGH every clock cycle until all buffers are inserting values into the systolic array (implements the required offset for an output-stationary systolic array).
        - Once every input buffer is empty, assert all *rdreq* signals LOW and assert *systolic_array_idle* HIGH.
        - Repeat.
    - **Error State:** The state machine defaults to a STATE_ERROR state in the event of undefined state machine behavior.
    








