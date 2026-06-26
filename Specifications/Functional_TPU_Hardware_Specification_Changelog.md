# Functional TPU Hardware Specification — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-26 — Functional TPU Hardware Specification v0.2

- **Control Signal Description:** Added a section that clarifies how the spec
defines control signals.
- Updated *Feeder* module to output an *end_reached* signal when an end
instruction is retrieved from program memory.

## 2026-06-22 — Functional TPU Hardware Specification v0.1.1

- Updated the *Feeder* module description to use the new *end* instruction
instead of *syscall* to terminate the program.

## 2026-06-22 — Log File

- Created `Functional_TPU_ISA_Changelog.md` to record changes made to the
*Functional TPU ISA*.
