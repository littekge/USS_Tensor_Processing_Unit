# Functional TPU ISA — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-22 — Functional TPU ISA v0.2
- Changed the SYSTEM instruction format to use a 7-bit function code *funct7* instead of a 3-bit code, allowing for greater extension of SYSTEM instructions in future revisions.
- Changed the *syscall* instruction (was previously used implicitly to terminate programs) to *end*, maintaining compatibility with program termination for older programs while defining a clearer termination command.
- Created a new *syscall* instruction with *funct7 = 0x1* to replace the old one.

## 2026-06-22 — Log File
- Created `Functional_TPU_ISA_Changelog.md` to record changes made to the *Functional TPU ISA*.