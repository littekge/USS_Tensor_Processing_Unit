# Functional TPU Hardware Specification — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-30 — Functional TPU Hardware Specification v0.2 — Design Fixes

- **Addressed pending design-level fixes.**
- Updated ALU and activator to suppress vector buffer write behavior when their
  operation/function is NO OP.
- Clarified that programmer error states are unrecoverable by any means other
than a full system reset (conforms with error behavior required by
`Functional_TPU_Message_Protocol_v0.2.md`)
- Updated Systolic Array module description to pass *trst* to the MACs in
addition to *clear*.

## 2026-06-30 — Functional TPU Hardware Specification v0.2 — Mechanical Fixes

- **Whole-spec cleanup pass** (mechanical fixes only; design-level issues
  deferred).
- Removed a stray change-log note accidentally left in the *Programmer*
  description.
- Added *mode_select* to the *Programmer IO Passthrough* inputs in the *TPU*
  module.
- Added the missing *data* input and *q* output to the
  *Systolic_Array_Input_Buffer* signal lists.
- Grammar/formatting: removed stray backslash before "loading" in
  *Vector_Processor*; added missing colon after *vector_start*; "a arithmetic"
  → "an arithmetic" in *ALU*; reflowed the *clear* bullet comma in *Controller*;
  "asserted low" → "asserted LOW" in *Feeder* and *SPI_Interface*; fixed missing
  possessive apostrophes throughout (module's, controller's, array's, vector
  processors', activator's, ALU's, buffer's; "its" → "it's" in
  *Vector_Processor*).
- Files modified: `Functional_TPU_Hardware_Specification_v0.2.md`.

## 2026-06-30 — Functional TPU Hardware Specification v0.2 — TPU

- **Small updates to TPU module**
- Updated the *Separate TPU Reset* section to AND *rst* and *tpu_rst* instead of
  *rst* and *tpu_rst* (allows programmer to access memory and reset the
internals separately).
- Added the *Programmer IO Passthrough* section to expose Programmer IO to
external sources.

## 2026-06-30 — Functional TPU Hardware Specification v0.2 — Programmer Revision

- **General revision and clarification to Programmer module description** (grammar,
wording, etc.)
- Clarified that *output_data* is used in both modes of the Programmer.
- Updated definition of IDLE meta-state to discard INPUT headers while in device
mode.
- Added explicit priority to meta-state changes from IDLE.

## 2026-06-29 — Functional TPU Hardware Specification v0.2 — Programmer

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
