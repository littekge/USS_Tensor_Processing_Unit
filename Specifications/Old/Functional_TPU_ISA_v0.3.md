# Functional TPU Instruction Set Architecture (ISA)

> **Purpose:** This document contains the *Functional TPU* instruction set
>architecture specification.
>
> **Version: 0.3.0**

## Memory

This section describes memory expectations of the *Functional TPU instruction
set architecture*. The ISA has an address space of 2^16=65536 words of length
XLEN. The memory address space is non-cyclic.

### Tensors in Memory

The ISA expects tensors to be formatted in memory as a contiguous flattened
array, with the address of the first element being used to dereference the
entire tensor. For example, if a tensor is stored at memory address 0x2 with
dimensions 2x2, the next unused address will be at 0x2 + 4 = 0x6. All matrix
operands referenced by memory addresses are logically interpreted by the ISA as
Row-Major arrays. Tensors of an order greater than 2 are not supported.

### Reserved Addresses

The ISA defines a number of reserved memory addresses that serve special
functions. Regardless of memory structure, reserved addresses are still treated
as valid locations in the main address space. Descriptions of reserved addresses
and their functionality are shown below.

**Address 0x0:**
Address 0x0 acts as a register with all bits hardwired to 0. 0x0 may only be
used as a source address; attempting to store data at 0x0 will raise an error,
discarding the data in the process.

**Address 0x1:**
The ISA defines address 0x1 as a unique, architecturally isolated memory
designation used to store temporary data. Any tensor in a format expected by the
ISA may be stored at address 0x1. The symbol 0x1 may be targeted as a
destination or source address by any instruction. Any write or store operation
targeting 0x1 inherently invalidates and completely overwrites all elements
previously residing within it. The ISA does not currently support partial overwrites.

--------------------------------------------------

## Instruction Format

This section describes the general format of *Functional TPU* instructions. In
the *Functional TPU* ISA, there are a variety of instruction formats (MUL,
SHAPE, ACT, ELEM, SYSTEM) which all have a fixed length of 128 bits.
Instructions are described with the most significant bit (127) on the left side
of the instruction, and the least significant bit (0) on the right side. In
common between all instruction sets are the following properties:

- **Opcode:** Bits 127-124 are reserved for the OPCODE.
- **Variable Instruction:** Bits 123-0 are variable based on instruction format.
Each instruction format is described below:

### MUL Format

**Description by Bit:**

| 127-124 | 123-108 | 107-104 | 103-100 | 99-84 | 83-80 | 79-76 | 75-60 | 59-52 | 51-44 | 43-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 4 | 4 | 16 | 4 | 4 | 16 | 8 | 8 | 41 | 3 |
| opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | M0 | n | reserved | funct3 |

### SHAPE Format

> *Not implemented in current revision of ISA*

### ACT Format

**Description by Bit:**

| 127-124 | 123-108 | 107-92 | 91-76 | 75-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 16 | 16 | 16 | 73 | 3 |
| opcode | rs1 | rd | len | reserved | funct3 |

### ELEM Format

**Description by Bit:**

| 127-124 | 123-108 | 107-92 | 91-84 | 83-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 16 | 8 | 8 | 16 | 57 | 3 |
| opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3 |

### SYSTEM Format

**Description by Bit:**

| 127-124 | 123-7 | 6-0 |
| --- | --- | --- |
| 4 | 117 | 7 |
| opcode | reserved | funct7 |

--------------------------------------------------

## Instructions

This section describes individual instructions available in the *Functional TPU*
ISA.

### Multiplication Instructions

**Description by Bit:**

| 127-124 | 123-108 | 107-104 | 103-100 | 99-84 | 83-80 | 79-76 | 75-60 | 59-52 | 51-44 | 43-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 4 | 4 | 16 | 4 | 4 | 16 | 8 | 8 | 41 | 3 |
| opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | M0 | n | reserved | funct3 |
| MUL | src1 | dim11 | dim12 | src2 | dim21 | dim22 | dest | scale | shift | 0 | MULT |

#### mult

*mult* takes a matrix of dimensions *dim11* x *dim12* located at address *src1*
and multiplies it by a matrix of dimensions *dim21* x *dim22* located at address
*src2*, storing the result contiguously starting from address *dest* in
Row-Major order. Before storage, each resultant value is requantized according to
the following equation: result = clamp((*scale* * x + (1 << (*shift* - 1))) >>
*shift*). *reserved* bits are set to 0. Note that the current revision of the
ISA does not support matrix dimensions greater than 15x15 for this instruction.

### Activation Instructions

**Description by Bit:**

| 127-124 | 123-108 | 107-92 | 91-76 | 75-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 16 | 16 | 16 | 73 | 3 |
| opcode | rs1 | rd | len | reserved | funct3 |
| ACT | src1 | dest | size | 0 | RELU |

#### relu

*relu* applies the rectified linear unit activation function (*mathematically
f(x) = max(0, x)*) to *size* contiguous elements beginning at address *rs1* and
storing them contiguously at address *dest*.

### Elementwise Instructions

**Description by Bit:**

| 127-124 | 123-108 | 107-92 | 91-84 | 83-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 16 | 8 | 8 | 16 | 57 | 3 |
| opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3 |
| ELEM | src1 | src2 | dim1 | dim2 | dest | 0 | ADD |

#### add

*add* performs elementwise addition between two matrices of dimensions *dim1* x
*dim2* located at addresses *src1* and *src2* respectively, storing the result
contiguously starting at address *dest* in Row-Major order. *reserved* bits are
set to 0. Note that the current revision of the ISA does not support matrix
dimensions greater than 255x255 for this instruction.

### System Instructions

**Description by Bit:**

| 127-124 | 123-7 | 6-0 |
| --- | --- | --- |
| 4 | 117 | 7 |
| opcode | reserved | funct7 |
| SYSTEM | 0 | END |
| SYSTEM | PRIVATE | SYSCALL |

#### end

The *end* instruction terminates the current program.

#### syscall

The *syscall* instruction is used to make a service request to the supporting
system hardware. The system will define how parameters for the service request
are passed.

--------------------------------------------------

## Instruction Set Listing

This section defines the concrete binary layouts for all instructions in the
*Functional TPU* ISA.

| Instruction | opcode | funct3 |
| --- | --- | --- |
| mult | 1000 | 0x0 |
| relu | 1001 | 0x0 |
| add | 1010 | 0x0 |

| Instruction | opcode | funct7 |
| --- | --- | --- |
| end | 0000 | 0x0 |
| syscall | 0000 | 0x1 |
