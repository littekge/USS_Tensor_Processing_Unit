# Functional TPU Hardware Specification — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-29 — Functional TPU Hardware Specification v0.2

- Updated Programmer module to include multiple methods of IO in addition to programming.
- Defined two modes of operation: device mode and external mode.
- defined multiple **meta-states** (**IDLE, PROGRAM, DEVICE_INPUT, DEVICE_OUTPUT,
EXTERNAL_INPUT, EXTERNAL_OUTPUT**) that perform new IO functions and improve
readability.
- Rewrote state machine flow to include meta-states and describe their implementation.
- defined *mode_select*, *device_ready*, and *input_data_valid* input signals.
- defined *tpu_rst* and *output_data_valid* output signals.
- defined *input_data* and *output_data* data signals.
- Clarified behavior of programmer when writing to program memory (old programs
are now overwritten rather than appended to when programming occurs).

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
