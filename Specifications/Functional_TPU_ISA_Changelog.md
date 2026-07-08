# Functional TPU ISA — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

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
