# Functional TPU ISA — Change Log

> Append a new entry every time a change is made. Newest entries at the top.
>

## 2026-07-16 — Functional TPU ISA v0.6 (Convolution & Pooling)

- **Added on-device convolution (im2col) and max-pooling instructions.**
- **Generalized the Row-Major memory model to tensors of any order.** Replaced
the "order greater than 2 not supported" rule: each dimension's stride is the
product of the dimensions that follow it, so a *C* x *H* x *W* tensor stores
channels outermost. MUL-, ELEM-, and ACT-type instructions still operate on
operands of at most two dimensions; higher-order tensors are addressed only by
the windowed instructions.
- **Added the W-Format**, a dedicated layout for the window descriptor.
- **Removed the reserved SHAPE format** (v0.5, unimplemented); SHAPE is now a
*type* carried by the A-Format.
- **Added the WINCONFIG type and the *window* instruction** (opcode 1110),
which loads the window descriptor until the next *window* or system reset, with
*outh*/*outw* supplied explicitly.
- **Defined the eleven Window Descriptor Registers** (*chans*, *inh*, *inw*,
*outh*, *outw*, *winh*, *winw*, *strh*, *strw*, *padh*, *padw*) — set by
*window*, read by *im2col*/*max*, mirroring how *stride* sets *ld1*/*ld2*.
- **Renamed the CONFIG type to LDCONFIG** (C-Format, *stride*); "Configuration"
now groups LDCONFIG and WINCONFIG.
- **Added the SHAPE type with the *im2col* function** (A-Format, opcode 1101):
expands a feature map into the (*chans* x *winh* x *winw*) x (*outh* x *outw*)
column matrix in channel-major order, zero-filling out-of-bounds elements; the
convolution is completed by the existing MUL/GEMM path.
- **Added the POOL type with the *max* function** (A-Format, opcode 1100): the
per-channel maximum over each window; out-of-bounds elements take the minimum
representable value.
- Added Instruction Set Listing rows: *window* (WINCONFIG, 1110/0x0), *im2col*
(SHAPE, 1101/0x0), *max* (POOL, 1100/0x0).

## 2026-07-15 — Functional TPU ISA v0.6 (Format Update)

- **Reorganized the instruction taxonomy into a three-level hierarchy:**
*Format* (bit layout) → *Type* (opcode) → *Function* (funct field). No new
instructions and no bit-layout changes — this stage is naming/structure only.
- **Renamed the instruction formats** to generic letter codes: *M-Format* (was
MUL), *A-Format* (was ACT), *E-Format* (was ELEM), *C-Format* (was CONFIG),
*S-Format* (was SYSTEM). Letters are labels only and carry no binding meaning.
- **Generalized the A-Format** operand field from *len* to a generic *aux*
field, so the format can host multiple Types (a single unary src→dst layout).
- Clarified that *Function* is selected by *funct3* for all formats except
*S-Format*, which uses *funct7*.
- Updated the Instruction Set Listing with *Type* and *Format* columns; opcode
and funct values are unchanged (mult/multip 1000, relu 1001, add 1010, stride
1111, end/syscall 0000).

## 2026-07-13 — Functional TPU ISA v0.5 (leading-dimension addressing)

- **Added strided (leading-dimension) matrix addressing** so a single MUL
instruction can operate on a sub-block of a larger Row-Major matrix in place —
enabling matrix multiplications larger than the systolic array via tiling.
- Added a **Registers** section defining two leading-dimension registers *ld1*
and *ld2* (row strides of *rs1* and *rs2*; *rd* always uses *ld2*). The ISA
otherwise defines no general-purpose registers.
- Added the **CONFIG instruction format** (*opcode* | *im1* | *im2* | reserved
| *funct3*).
- Added the ***stride*** instruction (CONFIG format) that loads *im1*/*im2* into
*ld1*/*ld2*; effective for all subsequent MUL-type instructions until the next
*stride* or system reset. *ld1*=*ld2*=0 (the default) means contiguous.
- Updated the *mult*/*multip* descriptions for sub-block addressing: rows of
*src1*/*src2*/*dest* are spaced *ld1*/*ld2*/*ld2* apart, and the "15x15" limit is
now a per-instruction tile bound (larger matrices reached by tiling).
- Added CONFIG to the instruction-format list, and a STRIDE row (opcode 1111,
funct3 0x0) to the Instruction Set Listing.

## 2026-07-08 — Functional TPU ISA v0.5 (multiply-in-place)

- **Added multiply-in-place functionality**
- Explicitly defined the internal matrix accumulator.
- Modified the description of *mult* to include an accumulator clear after the
operation is complete.
- Added the ***multip*** instruction for multiply-in-place -> same exact
function as *mult*, but does not clear the accumulator.

## 2026-07-08 — Functional TPU ISA v0.4

- **Changed address length in instructions**
- Addresses encoded in instructions are now 3 bytes (upped from 2)
- Made memory word count variable with parameter XWIDTH (max is 2^24).

## 2026-07-06 — Functional TPU ISA v0.3

- **Added precise requantization functionality** for MUL instructions.
- MUL instructions now encode an *M0* and *n* field, both 8-bit unsigned integers.
- The *mult* instruction now utilizes the *M0* and *n* fields as *scale* and
*shift* values respectively.
- Updated the *mult* instruction description to include the necessary
requantization equation: result = clamp((*scale* * x + (1 << (*shift* - 1))) >>
*shift*).

## 2026-07-01 — Specification Structure Update

- **Changed specification naming scheme:**
- Changed name of `Functional_TPU_ISA_v0.2.md` to
`Functional_TPU_ISA.md` — most recent version of spec will
always be referenced by this filename. Old versions will still have a version
number specified in the filename.
- Updated top level header of spec to include a version number.

## 2026-06-22 — Functional TPU ISA v0.2

- Changed the SYSTEM instruction format to use a 7-bit function code *funct7*
instead of a 3-bit code, allowing for greater extension of SYSTEM instructions
in future revisions.
- Changed the *syscall* instruction (was previously used implicitly to terminate
programs) to *end*, maintaining compatibility with program termination for older
programs while defining a clearer termination command.
- Created a new *syscall* instruction with *funct7 = 0x1* to replace the old one.

## 2026-06-22 — Log File

- Created `Functional_TPU_ISA_Changelog.md` to record changes made to the
*Functional TPU ISA*.
