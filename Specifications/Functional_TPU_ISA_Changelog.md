# Functional TPU ISA — Change Log

> Append a new entry every time a change is made. Newest entries at the top.
>
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
