# Functional TPU Hardware Specification — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-07-16 — Functional TPU Hardware Specification v0.6.0 — Convolution and Pooling

- **Added the hardware for convolution (via *im2col*) and max pooling**, in
tandem with ISA v0.6.
- **Pooler (new module, peer of the ALU and Activator; feeds the Vector_Buffer):**
- Reduces a stream of input values to a single output value per pooling window
using one accumulator and one reduction operator (a comparator for *max*).
- Folds one value per clock while *enable* is HIGH; on *window_end* it writes the
result to the vector buffer and resets the accumulator to the reduction identity
(the minimum representable value for *max*).
- The Controller selects the function, fixing both the operator and its identity;
NO OP suppresses buffer writes. Performs no requantization and cannot overflow.
- **Vector_Processor:**
- Added windowed addressing for *im2col* and *max*: derives input feature map
coordinates from the window descriptor, checks them against the feature map
bounds, and substitutes the out-of-bounds fill (0 for *im2col*, minimum
representable value for *max*) in place of a memory read.
- Added the *window_end* output, asserted with *element_valid* on the final
element of each pooling window.
- Added the Pooler input as a destination. The element count for windowed
instructions is derived from the window descriptor rather than the *length* input.
- **Controller:**
- Added the eleven window descriptor registers (*chans*, *inh*, *inw*, *outh*,
*outw*, *winh*, *winw*, *strh*, *strw*, *padh*, *padw*), loaded by the *window*
instruction and retained until the next *window* instruction or *trst*; they have
no default value.
- Generalized the *stride* decode-and-latch path into a single
register-configuration path covering LDCONFIG and WINCONFIG (no Execute or
Writeback phase).
- Skips the Writeback phase for instructions that write their result directly to
memory during Execute (*im2col*).
- Added the Pooler to the inter-operation *clear* list.
- **TPU:**
- Wired vector processor *a* to the Pooler (*element_valid* → *enable*,
*window_end* → *window_end*).
- Added the Pooler to the *trst* reset list.
- **Programmer:** Added the Pooler to the *tpu_rst* reset-module list.

## 2026-07-14 — Functional TPU Hardware Specification v0.5.1 — SPI_Slave Hardening

- **Rewrote the SPI_Slave module description** to reflect a synchronous
oversampling receiver, replacing the previous externally-sourced (nandland)
description. The module interface and byte-level behavior are unchanged.
- **SPI_Slave:**
- Documented the *clk*/*rst*, SPI (*SPI_Clk*, *MOSI*, *CS_n*, *MISO*), and byte
interface (*RX_DV*, *RX_Byte*, *TX_DV*, *TX_Byte*) signals, plus the SPI mode
parameter (CPOL/CPHA, modes 0–3; deployed at mode 0).
- Specified byte reception (MSB-first, one byte per eight *SPI_Clk* sampling
edges, with a one-clock-cycle *RX_DV* pulse), back-to-back multi-byte
transactions while *CS_n* is held LOW, and the tri-stated MISO transmit path.
- Added a *Synchronous Recovery* section: the SPI inputs are synchronized into
the *clk* domain, debounced, and *SPI_Clk* edges are recovered by oversampling —
the module never uses *SPI_Clk* as a register clock or *CS_n* as an asynchronous
reset — so glitches and slow (signal-integrity-limited) edges cannot produce
spurious clock edges or transaction boundaries.

## 2026-07-13 — Functional TPU Hardware Specification v0.5 — Leading Dimension Update

- **Implemented strided (leading-dimension) matrix addressing** in tandem with
ISA v0.5.
- **Controller:**
- Added two 24-bit leading-dimension registers *ld1* and *ld2*, set by the
*stride* instruction and cleared to 0 (contiguous) on *trst*.
- Added a decode-and-latch path for the *stride* instruction (loads *im1*/*im2*
into *ld1*/*ld2*; no Execute or Writeback phase).
- Supplies the active leading dimension to the vector processors: *ld1* when
loading *src1*, *ld2* when loading *src2*, and *ld2* during writeback to *dest*.
- **Vector_Processor:**
- Added a *leading_dimension* combinational input.
- Reads and writes matrices with successive rows spaced *leading_dimension*
elements apart, allowing in-place access to a sub-block of a larger Row-Major
matrix; a value of 0 selects contiguous access (rows spaced by *dim1*).

## 2026-07-08 — Functional TPU Hardware Specification v0.5 — *multip* Update

- **Implemented multiply-in-place functionality.**
- Separated *clear* in Systolic_Array/Multiply_Accumulate_Unit modules into two
signals, *clear* and *accumulator_clear*.
  - *clear* flushes residuals.
  - *accumulator_clear* clears the *c* accumulators in the MACs.
- Updated the Controller module to assert *accumulator_clear* after writeback
based on instruction.

## 2026-07-08 — Functional TPU Hardware Specification v0.4 — Memory Update

- **Abstracted memory into a unified Data_Memory module**
- Weight_Memory renamed to Mem_Unit -> stores general program data instead of
only weights.
- Data_Memory now instantiates the TPU_0x1_Buffer module and multiple Mem_Unit
modules, abstracting them into a unified memory space.
- The 0x1 buffer is now accessed using an *offset* signal.
- **Syntax change:** The ISA and Message protocol are now referenced by filename
at the top of the spec and previous filename references have been replaced by
italicized text.
- Clarified WRITE_ERROR vs MEM_ERROR -> WRITE_ERROR is raised when writing to a
read-only address (0x0) and MEM_ERROR is raised when accessing an invalid memory
location (e.g. an out of range address).

## 2026-07-06 — Functional TPU Hardware Specification v0.3 — Quantization

- **Improved requantization method for mult instructions** in tandem with ISA
v0.3.
- **Vector_Processor:**
- Added explicit quantization instructions in the *requantization* section.
- Added *scale* and *shift* combinational inputs — requantization constants from
  MUL-type instructions.
- **ALU** — Renamed the *requantization* section to *overflow handling* ->
requantization no longer occurs in the ALU, only the vector processors.

## 2026-07-01 — Specification Structure Update

- **Changed specification naming scheme:**
- Changed name of `Functional_TPU_Hardware_Specification_v0.2.md` to
`Functional_TPU_Hardware_Specification.md` — most recent version of spec will
always be referenced by this filename. Old versions will still have a version
number specified in the filename.
- Updated top level header of spec to include a version number.

## 2026-07-01 — Functional TPU Hardware Specification v0.2 — Minor change

- Exposed *end_reached* output from Feeder module as an output of the TPU
module.
- Updated description of *device_ready* signal in programmer to note that
it's asserted HIGH as long as the device is ready, not for one clock cycle.
- Updated description of **EXTERNAL_OUTPUT** meta-state to wait for
*device_ready* before asserting *output_data_valid*.

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
  *rst* and *program* (allows programmer to access memory and reset the
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

- Created `Functional_TPU_Hardware_Specification_Changelog.md` to record changes
made to the *Functional TPU Hardware Specification*.
