# Functional TPU Instruction Set Architecture (ISA)

> **Purpose:** This document contains the *Functional TPU* instruction set
> architecture specification.
>
> **Version: 0.7.0**

## Memory

This section describes memory expectations of the *Functional TPU instruction
set architecture*. The ISA has an address space of XWIDTH words of length
XLEN. The memory address space is non-cyclic. While XWIDTH and XLEN are both
variable, the ISA does not support XWIDTH > 2^24 = 16777216.

### Tensors in Memory

The ISA expects tensors to be formatted in memory as a contiguous flattened
array, with the address of the first element being used to dereference the
entire tensor. For example, if a tensor is stored at memory address 0x2 with
dimensions 2x2, the next unused address will be at 0x2 + 4 = 0x6. All matrix
operands referenced by memory addresses are logically interpreted by the ISA as
Row-Major arrays.

Tensors of any order are supported and are stored in Row-Major order: the last
dimension varies fastest, and each dimension's stride is the product of all
dimensions that follow it. For example, in a *C* x *H* x *W* tensor the channel
stride is *H* x *W*, the row stride is *W*, and the element stride is 1; a
2 x 3 x 3 tensor stored at 0x2 spans 18 elements, so the next unused address is
0x2 + 18 = 0x14 and its second channel begins at 0x2 + 9 = 0xB. MUL-, ELEM-,
and ACT-type instructions operate on operands of at most two dimensions;
higher-order tensors are addressed only by the windowed instructions (*im2col*
and *max*), per the window descriptor. Pure data movement operations (e.g.
*move* instruction) are dimension-agnostic.

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
previously residing within it. The ISA does not currently support partial
overwrites.

## Registers

This section describes the registers defined by the *Functional TPU instruction
set architecture*. **Note** that the ISA does not define any general purpose
registers; each register has a reserved function and all generic data is stored
in main memory.

### Leading Dimension Registers

| Name | Description |
| --- | --- |
| ld1 | Leading dimension of *rs1* |
| ld2 | Leading dimension of *rs2* |

The ISA defines two *leading dimension registers*, *ld1* and *ld2*. A leading
dimension is the physical row stride of a larger Row-Major matrix, may exceed
the operand's logical width, and lets an instruction address a sub-block in
place. Leading dimension registers are set by the *stride* instruction, persist
until the next *stride* instruction or system reset, and default to contiguous
(no stride, *ld1*=*ld2*=0). *ld1* and *ld2* define the leading dimensions of
*rs1* and *rs2* for all MUL-type instructions. For writeback, the
leading dimension of *rd* is always equal to *ld2*.

### Window Descriptor Registers

| Name | Description |
| --- | --- |
| chans | Input feature map channels |
| inh | Input feature map height |
| inw | Input feature map width |
| outh | Output feature map height |
| outw | Output feature map width |
| winh | Window height |
| winw | Window width |
| strh | Vertical window stride |
| strw | Horizontal window stride |
| padh | Vertical zero-padding |
| padw | Horizontal zero-padding |

The ISA defines a set of *window descriptor* registers holding the
parameters for windowed instructions (*im2col*, *max*): the input feature map
dimensions (*chans*, *inh*, *inw*), the output spatial dimensions (*outh*,
*outw*), the window dimensions (*winh*, *winw*), the window strides (*strh*,
*strw*), and the padding (*padh*, *padw*). The window descriptor is set by the
*window* instruction and persists until the next *window* instruction or
system reset. It has no default value; a *window* instruction must set the
descriptor before any windowed instruction is issued.

## Instruction Formats

This section describes the general format of *Functional TPU* instructions. In
the *Functional TPU* ISA, there are a variety of instruction formats which all
have a fixed length of 128 bits. Instructions are described with the most
significant bit (127) on the left side of the instruction, and the least
significant bit (0) on the right side. In common between all instruction formats
are the following properties:

- **Opcode:** Bits 127-124 are reserved for the OPCODE.
- **Variable Instruction:** Bits 123-0 are variable based on instruction format.

Instructions in the *Functional TPU ISA* follow a three-level hierarchy:

- **Format:** The physical bit layout. One *format* may be shared by more than
  one type.
- **Type:** Selected by the *opcode*. The *type* determines the operation class
  and the data path it dispatches to.
- **Function:** Selected by a *funct* (or otherwise similar) field. The
  *function* selects a specific operation within a class.

Each instruction format is described below:

### M-Format

| 127-124 | 123-100 | 99-96 | 95-92 | 91-68 | 67-64 | 63-60 | 59-36 | 35-28 | 27-20 | 19-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 24 | 4 | 4 | 24 | 4 | 4 | 24 | 8 | 8 | 17 | 3 |
| opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | M0 | n | reserved | funct3 |

### A-Format

| 127-124 | 123-100 | 99-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 16 | 57 | 3 |
| opcode | rs1 | rd | aux | reserved | funct3 |

### E-Format

| 127-124 | 123-100 | 99-76 | 75-68 | 67-60 | 59-36 | 35-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 8 | 8 | 24 | 33 | 3 |
| opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3 |

### C-Format

| 127-124 | 123-100 | 99-76 | 75-3 | 2-0 |
| --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 73 | 3 |
| opcode | im1 | im2 | reserved | funct3 |

### W-Format

| 127-124 | 123-112 | 111-100 | 99-88 | 87-76 | 75-64 | 63-60 | 59-56 | 55-52 | 51-48 | 47-44 | 43-40 | 39-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 12 | 12 | 12 | 12 | 12 | 4 | 4 | 4 | 4 | 4 | 4 | 37 | 3 |
| opcode | inh | inw | chans | outh | outw | winh | winw | strh | strw | padh | padw | reserved | funct3 |

### S-Format

| 127-124 | 123-7 | 6-0 |
| --- | --- | --- |
| 4 | 117 | 7 |
| opcode | reserved | funct7 |

## Instructions

This section describes individual instructions available in the *Functional TPU*
ISA. Instructions are grouped by type.

### Multiplication Instructions

**M-Format Multiplication Instructions:**

| 127-124 | 123-100 | 99-96 | 95-92 | 91-68 | 67-64 | 63-60 | 59-36 | 35-28 | 27-20 | 19-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 24 | 4 | 4 | 24 | 4 | 4 | 24 | 8 | 8 | 17 | 3 |
| opcode | rs1 | sz11 | sz12 | rs2 | sz21 | sz22 | rd | M0 | n | reserved | funct3 |
| MUL | src1 | dim11 | dim12 | src2 | dim21 | dim22 | dest | scale | shift | 0 | MULT |
| MUL | src1 | dim11 | dim12 | src2 | dim21 | dim22 | dest | scale | shift | 0 | MULTIP |

The Functional TPU maintains an internal matrix accumulator that holds multiply
results at full internal precision, prior to requantization. Both multiplication
instructions compute the matrix product of their operands, add it into the
accumulator, requantize the accumulator's contents, and store the result to
dest; they differ only in their treatment of the accumulator after writeback,
selected by *funct3*. The maximum dimensions of the accumulator are an
implementation-defined hardware parameter, and a sequence of accumulating
multiplies must target identical *dest* dimensions that do not exceed it.

Each operand (src1, src2) and the destination (dest) addresses the top-left
element of a matrix that may be a sub-block of a larger Row-Major matrix. The
dim fields give each operand's logical dimensions, while the leading dimension
registers give the physical row stride used to walk it: successive rows of src1
are spaced ld1 elements apart, rows of src2 are spaced ld2 apart, and the result
is written to dest with rows spaced ld2 apart. When ld1=ld2=0 (the default),
rows are spaced by the operand's own width and every matrix is read and written
contiguously. A non-zero leading dimension lets a single MUL instruction operate
on a tile of a larger matrix in place, without rearranging memory (see Leading
Dimension Registers).

#### mult

*mult* takes a matrix of dimensions *dim11* x *dim12* located at address *src1*
and multiplies it by a matrix of dimensions *dim21* x *dim22* located at address
*src2*, storing the result starting from address *dest* in Row-Major order
(addressed per the leading dimensions described above). Before storage, each
resultant value is requantized according to the following equation: result =
clamp((*scale* * x + ((1 << *shift*) >> 1)) >> *shift*) (Note: clamp() denotes
saturation to XLEN min/max). After writeback, the accumulator is cleared.
Reserved bits are set to 0. Note that the dim fields limit each operand to at
most 15x15 elements per instruction; larger matrices are computed by issuing
MUL instructions over array-sized tiles addressed within their parent matrices
via *ld1* and *ld2*.

#### multip

*multip* performs the identical multiply → accumulate → requantize → writeback
sequence as *mult*, but does not clear the accumulator afterward, so a
subsequent multiplication accumulates into the same accumulator. A run of
*multip* instructions terminated by a *mult* computes the requantized sum of
their individual products, enabling matrix multiplications whose contraction
dimension exceeds a single instruction. Every instruction in such a run must
reference identical *dest* dimensions; intermediate writebacks are overwritten
by later ones, so only the final *mult* writeback is meaningful. The leading
dimensions in effect (ld1, ld2) are shared by every instruction in the run,
since they are set once by a preceding stride instruction and persist across the
whole accumulation.

### Activation Instructions

**A-Format Activation Instructions:**

| 127-124 | 123-100 | 99-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 16 | 57 | 3 |
| opcode | rs1 | rd | aux | reserved | funct3 |
| ACT | src1 | dest | len | 0 | RELU |

#### relu

*relu* applies the rectified linear unit activation function (*mathematically
f(x) = max(0, x)*) to *len* contiguous elements beginning at address *src1* and
stores them contiguously at address *dest*.

### Element-wise Instructions

**E-Format Element-wise Instructions:**

| 127-124 | 123-100 | 99-76 | 75-68 | 67-60 | 59-36 | 35-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 8 | 8 | 24 | 33 | 3 |
| opcode | rs1 | rs2 | sz1 | sz2 | rd | reserved | funct3 |
| ELEM | src1 | src2 | dim1 | dim2 | dest | 0 | ADD |

#### add

*add* performs element-wise addition between two matrices of dimensions *dim1* x
*dim2* located at addresses *src1* and *src2* respectively, storing the result
contiguously starting at address *dest* in Row-Major order. *reserved* bits are
set to 0. Note that the current revision of the ISA does not support matrix
dimensions greater than 255x255 for this instruction.

### Configuration Instructions

**C-Format Configuration Instructions:**

| 127-124 | 123-100 | 99-76 | 75-3 | 2-0 |
| --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 73 | 3 |
| opcode | im1 | im2 | reserved | funct3 |
| LDCONFIG | ld1 | ld2 | 0 | STRIDE |

**W-Format Configuration Instructions:**

| 127-124 | 123-112 | 111-100 | 99-88 | 87-76 | 75-64 | 63-60 | 59-56 | 55-52 | 51-48 | 47-44 | 43-40 | 39-3 | 2-0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 12 | 12 | 12 | 12 | 12 | 4 | 4 | 4 | 4 | 4 | 4 | 37 | 3 |
| opcode | inh | inw | chans | outh | outw | winh | winw | strh | strw | padh | padw | reserved | funct3 |
| WINCONFIG | inh | inw | chans | outh | outw | winh | winw | strh | strw | padh | padw | 0 | WINDOW |

#### stride

The *stride* instruction loads *im1* and *im2* into the *ld1* and *ld2*
registers respectively. *ld1* and *ld2* are effective for all subsequent
MUL-type instructions until the next *stride* instruction or system reset. A
value of 0 for *ld1* and *ld2* indicates no stride; matrices will be read
contiguously. *ld1* and *ld2* are set to 0 by default.

#### window

The *window* instruction loads the window descriptor used by all
subsequent windowed instructions until the next *window* instruction or
system reset. The descriptor specifies the input feature map (*chans* channels,
*inh* rows, *inw* columns), the window dimensions (*winh* x *winw*), the window
stride (*strh*, *strw*), the zero-padding (*padh*, *padw*), and the output
spatial dimensions (*outh* x *outw*). *outh* and *outw* are supplied explicitly
and must equal floor((*inh* + 2*padh* - *winh*) / *strh*) + 1 and floor((*inw* +
2*padw* - *winw*) / *strw*) + 1 respectively; the hardware does not derive them.
Reserved bits are set to 0.

### Shape Instructions

**A-Format Shape Instructions:**

| 127-124 | 123-100 | 99-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 16 | 57 | 3 |
| opcode | rs1 | rd | aux | reserved | funct3 |
| SHAPE | src1 | dest | 0 | 0 | IM2COL |
| SHAPE | src1 | dest | len | 0 | MOVE |

#### im2col

The *im2col* instruction expands the input feature map at *src1* into the
column matrix used to compute a convolution as a matrix multiply, writing the
result to *dest*. It uses the window descriptor set by the most recent *window*
instruction and interprets the feature map as a *chans* x *inh* x *inw* tensor
in Row-Major order. For each of the *outh* x *outw* output positions, *im2col*
gathers the *winh* x *winw* window across all *chans* channels into one column
of length *chans* x *winh* x *winw*, ordered channel-major (channel outermost,
then window row, then window column). The columns are written to *dest* as a
(*chans* x *winh* x *winw*) x (*outh* x *outw*) matrix in Row-Major order.
Window elements that fall outside the feature map (due to *padh* or *padw*)
contribute a value of 0. *aux* and reserved bits are set to 0.

#### move

The *move* instruction moves *len* contiguous values starting from address
*src1* and stores them contiguously at address *dest*. *move* can relocate up to
65535 elements. Move operations that exceed 65535 elements require multiple
*move* instructions. Note that *move* is dimension-agnostic; it moves raw data
between memory locations. Reserved bits are set to 0.

### Pooling Instructions

**A-Format Pooling Instructions:**

| 127-124 | 123-100 | 99-76 | 75-60 | 59-3 | 2-0 |
| --- | --- | --- | --- | --- | --- |
| 4 | 24 | 24 | 16 | 57 | 3 |
| opcode | rs1 | rd | aux | reserved | funct3 |
| POOL | src1 | dest | 0 | 0 | MAX |

#### max

The *max* instruction performs max pooling on the input feature map at *src1*,
writing the pooled result to *dest*. It uses the window descriptor set by the
most recent *window* instruction and interprets the feature map as a *chans* x
*inh* x *inw* tensor in Row-Major order. For each channel independently and each
of the *outh* x *outw* output positions, *max* outputs the greatest value within
the *winh* x *winw* window, positioned by the strides *strh* and *strw*. The
result is written to *dest* as a *chans* x *outh* x *outw* tensor in Row-Major
order. Window elements that fall outside the feature map (due to *padh* or
*padw*) take the minimum representable value and therefore never affect the
result. *aux* and reserved bits are set to 0.

### System Instructions

**S-Format System Instructions:**

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

## Instruction Set Listing

This section defines the concrete binary layouts for all instructions in the
*Functional TPU* ISA.

| Instruction | Type | Format | opcode | funct3 |
| --- | --- | --- | --- | --- |
| mult | MUL | M-Format | 1000 | 0x0 |
| multip | MUL | M-Format | 1000 | 0x1 |
| relu | ACT | A-Format | 1001 | 0x0 |
| add | ELEM | E-Format | 1010 | 0x0 |
| stride | LDCONFIG | C-Format | 1111 | 0x0 |
| window | WINCONFIG | W-Format | 1110 | 0x0 |
| im2col | SHAPE | A-Format | 1101 | 0x0 |
| move | SHAPE | A-Format | 1101 | 0x1 |
| max | POOL | A-Format | 1100 | 0x0 |
| Instruction | Type | Format | opcode | funct7 |
| --- | --- | --- | --- | --- |
| end | SYSTEM | S-Format | 0000 | 0x0 |
| syscall | SYSTEM | S-Format | 0000 | 0x1 |
