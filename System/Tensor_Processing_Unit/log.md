# Functional TPU — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-22 — Build Step 7: Systolic Array completion, integration, and tests

- **Fixed `TPU/SYSTOLIC_ARRAY/Systolic_Array.v`** — corrected the errors in the
  partially-built module:
  - Next-state logic referenced an undeclared `LOAD_WHILE_CHECK` state and left
    `NS` unassigned; replaced the unused `LOAD_FOR_*` parameters with a clean
    `STATE_ERROR/START/IDLE/LOAD` FSM.
  - Every input buffer drove the same `all_empty` wire (2N drivers contending on
    one net). Split into per-buffer `top_empty[N-1:0]` / `left_empty[N-1:0]`
    vectors and derived `all_empty = (&top_empty) & (&left_empty)`.
  - `top_rdreq` / `left_rdreq` were undriven `wire` arrays, so no buffer ever
    read. Changed to packed `reg [N-1:0]` driven by combinational decode.
  - Interconnect nets were declared `[0:N-1][0:N-1]` but indexed out of bounds
    at `v_interconnect[i][j+1]` and `h_interconnect[i+1][j]`. Resized to
    `v_interconnect[0:N-1][0:N]` and `h_interconnect[0:N][0:N-1]` to hold the
    bottom/right edge outputs.
- **Implemented the FSM per the hardware spec:** on `systolic_array_start` the
  array enters LOAD, asserts `systolic_array_idle` LOW, and ramps a `load_cnt`
  counter so buffer M begins reading on the Mth cycle (the diagonal offset
  required by an output-stationary array). `rdreq` for each buffer is gated by
  its `empty` flag. Once `all_empty` is HIGH the FSM returns to IDLE and reasserts
  `systolic_array_idle`.
- **Tiling convention documented and verified:** first net index = column
  (a/top stream), second = row (b/left stream); `o_c = unit_out[i_col][i_row]`,
  matching the Vector_Processor readback (`o_sa_row=col_cnt`, `o_sa_col=row_cnt`).
- **Updated `TPU/TPU.v`:** instantiated `Systolic_Array #(.N(8))`; connected
  `ctrl_clear`, `ctrl_systolic_array_start`, VP_A top buffers, VP_B left buffers,
  and VP_A SA read interface (`o_sa_row/o_sa_col`/`i_sa_c`). Replaced the
  Controller `i_systolic_array_idle` stub (was `1'b1`) and the VP_A `i_sa_c` stub
  (was `32'd0`) with the real `sa_systolic_array_idle` and `sa_c` nets.
- **Created `tests/TB_Step7_SystolicArray.v`** — 7-test testbench. Tests: reset
  idle state; clear flushes all input buffers; start lowers idle; read-request
  skew (buffer M enabled on cycle M); rdreq gated by empty; idle returns after
  drain; MAC tile accumulates a non-zero product with correct `o_c` output-mux
  decode (untouched tiles read zero).
- **Verification:** full TPU hierarchy compiles with 0 errors/0 warnings under
  Questa; top-level `TPU` elaborates cleanly; all 7 testbench cases PASS
  (`vsim -L altera_mf_ver -voptargs=+acc`).

## 2026-06-18 — Build Step 7: Systolic Array module
- Began building `TPU/SYSTOLIC_ARRAY/Systolic_Array.v` — top level module for the systolic array subsystem.
- Generates a systolic array from `TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Buffer.v` modules by instantiating a grid of instances connected by wire nets.
- Generates input buffers from `TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v` and connects them to the top and left edges of the systolic array.
- Combinationally decodes control signals from the two vector processors to write to the input buffers and read from the systolic array outputs.
- Partially defined state machine to handle systolic array insertion timings and synchronous control signals from the controller.

## 2026-06-18 — Build Step 7: Systolic Array Input Buffer
- Built `TPU/SYSTOLIC_ARRAY/TEMPLATE/BUFFER_TEMPLATE.v` — template for directly instantiating a scfifo megafunction (unused in actual project).
- Built `TPU/SYSTOLIC_ARRAY/Systolic_Array_Input_Buffer.v` — wrapper module for a scfifo megafunction FIFO buffer. 
- Standard buffer control signals `wrreq`, `rdreq`, `empty`, `full`, and `sclr`.
- Data input `data` and data output `q`.
- Variable depth via the `DEPTH` parameter during module instantiation.

## 2026-06-17 — Build Step 7: Multiply Accumulate Unit
- Built `TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v` — single multiply accumulate unit that serves as one "tile" in the systolic array.
- Performs one single multiply accumulate operation (c=c+a*b) every clock cycle.
- Forwards `i_a` and `i_b` to `o_a` amd `o_b` respectively every clock cycle.

## 2026-06-17 — Testbench: ALU Module
- Created `tests/TB_Step6_ALU.v` — 6-test testbench. Tests: o_data is forced to 45 when i_trst LOW, i_enable LOW, or i_clear HIGH, i_enable routes to o_write, NOOP operation functions correctly, ADD operation functions correctly, positive clamping functions correctly, negative clamping functions correctly.

## 2026-06-16 — Debug: Tensor_Processing_Unit and TPU modules
- Modified TPU to output a debug signal `o_debug_val` that concatenates relevant output signals to prevent removal of memory and logic during synthesis.
- Instantiated `debug` module in `Tensor_Processing_Unit.v` to output `o_debug_val` to a monitor via VGA.
- Created basic logic to connect FPGA external input `KEY[3]` to `i_write_next` in the debug module to trigger writing to the screen when `KEY[3]` is pressed.

## 2026-06-16 — Build Step 6: ALU Module
- Built `/TPU/PROCESSING/ALU.v` — combinational ALU that implements ADD and NOOP function codes.
- Module input `i_clk` intentionally not connected.
- `i_trst` and `i_clear` signals hold the module in a default state.
- `i_enable` assigned to `o_write` to adhere to hardware spec while maintaining pure combinational logic.
- Defined `clamp()` function to clamp output from the ADD operation to fit within an 8 bit signed integer.
- Updated `TPU/TPU.v` to connect relevant control signals...
- `ctrl_clear` and `ctrl_alu_funct` connected to ALU.
- `vpa_data` and `vpb_data` connected to `i_data_a` and `i_data_b` respectively.
- `alu_enable` stub from build step 4 connected to `i_enable`.
- `alu_write` and `alu_data` from build step 5 connected to `o_write` and `o_data` respectively.

## 2026-06-16 — Testbench: Activator Module

- **Activator Debug Values:** added debug values to the activator module to indicate a default state (45), control error (29), or NOOP  function (98).
- Created `tests/TB_Step5_Activator.v` — 6-test testbench. Tests: o_data is forced to 45 when i_trst LOW, o_data is forced to 45 when i_clear HIGH, o_data is forced to 45 when i_enable LOW, i_enable routes to o_write, NOOP operation functions correctly, ReLu operation functions correctly.

## 2026-06-15 — Build Step 5: Activator Module

- Built `/TPU/PROCESSING/Activator.v` — combinational activation function module that implements ReLu and NOOP function codes.
- Module input `i_clk` intentionally not connected.
- `i_trst` and `i_clear` signals hold the module in a default state.
- `i_enable` assigned to `o_write` to adhere to hardware spec while maintaining pure combinational logic.
- Updated `TPU/TPU.v` to connect relevant control signals...
- `ctrl_clear` and `ctrl_activator_funct` stubs connected to activator.
- `vpa_data` stub connected to `i_data` in activator module.
- Defined `alu_write`, `act_write`, `alu_data`, and `act_data` to connect to write and data outputs (`alu_write` and `alu_data` are stubs to be connected in build step 6). 
- `vb_wrreq` and `vb_data` updated to be driven by `act_write` and `alu_write`. 
- `vb_data` is set combinationally based on `alu_write`.

## 2026-06-15 — Testbench: Added Tests 8–10 (0x1 Buffer and zero source coverage)

- **Test 8** (`VSRC_MEM src=0x0001 → VDST_ACT`): Exercises the 0x1 Buffer
  as a *read* source. Loads `buf_mem[0..1]` with 0xD1/0xD2, expects
  `element_valid` to pulse twice with `o_data` = 0xD1 then 0xD2.
  Verifies `buf_rd_addr = cur_rd_isa − mem_source_lat = elem_cnt` path.
- **Test 9** (`VSRC_VEC_BUF → VDST_MEM dest=0x0001`): Exercises the 0x1
  Buffer as a *write* destination. Pushes 0xE1/0xE2 into the VB, expects
  them written to `buf_mem[0..1]`.
  Verifies `buf_wr_addr = cur_wr_isa − mem_dest_lat = elem_cnt` path and
  the `dst_is_buf` branch in `VB_WRITE`.
- **Test 10** (`VSRC_MEM src=0x0000 → VDST_ACT`): Exercises the hardwired-
  zero source. `mem_rd_data` is combinationally forced to 0x00 when
  `mem_source_lat == 0x0000`; the state machine proceeds normally.
  Expects `element_valid` twice with `o_data = 0x00`.
- Updated header comment: "All 10 tests should PASS".

## 2026-06-15 — Testbench Fix: vb_q implicit 1-bit wire (Test 5)

- **Root cause (testbench bug):** `vb_q` was never explicitly declared. In
  Verilog 2001, using an undeclared name in a continuous assignment
  (`assign vb_q = vb_q_reg`) causes an implicit 1-bit `wire` declaration.
  Only bit [0] was driven; bits [7:1] floated to Z, making `i_vb_q` appear
  as `zZ` in the waveform. The VP always read 0x00 or 0xZZ from the FIFO.
- **Fix (`tests/TB_Step4_VectorProcessor.v`):** Added explicit
  `wire [7:0] vb_q;` declaration alongside `vb_q_reg`.

## 2026-06-15 — VP Bug Fix: o_element_valid registered to align with o_data

- **Root cause (VP bug, not testbench):** `o_element_valid` was a combinational
  wire (`assign o_element_valid = (S == MEM_FORWARD) && ...`), while `o_data`
  is a registered output updated via non-blocking assignment (NBA) in the
  MEM_FORWARD case. Because NBA updates take effect *after* the posedge active
  region, the correct data did not appear on `o_data` until the cycle *after*
  `o_element_valid` had already gone LOW. The two signals were never
  simultaneously valid.
- **Fix (`TPU/PROCESSING/Vector_Processor.v`):**
  - Changed `output wire o_element_valid` → `output reg o_element_valid`.
  - Added `o_element_valid <= 1'b0` to the reset block and the default
    deassert section (alongside `o_wm_wren`, `o_vb_rdreq`, etc.).
  - In the MEM_FORWARD/VDST_ACT case: `o_element_valid <= 1'b1` alongside the
    existing `o_data <= mem_rd_data`. Both NBAs commit at the same posedge, so
    on the *following* cycle both `o_element_valid` and `o_data` are
    simultaneously correct for the downstream consumer.
  - Removed the stale `assign o_element_valid = ...` statement.
- **Testbench (`tests/TB_Step4_VectorProcessor.v`):** No changes required.
  The capture logic (`always @(posedge clk) if (vp_element_valid) ev_data[ev_count] <= vp_data`)
  was already correct in its expectation; the VP was not fulfilling the
  contract it implied.

## 2026-06-15 — Testbench Fix: TB_Step4_VectorProcessor tests 2–4

- **Root cause:** `MEM_ADDR` state checks `dst_is_zero` for all operations,
  not just memory-write ones. Tests 2, 3, and 4 passed `mem_dest_address =
  16'h0000`, which triggered `MEM_ERROR` even for `VDST_ACT`, `VDST_SA_A`,
  and `VDST_SA_B` where the dest address is unused. With VP stuck in
  `MEM_ERROR`, tests 5 and 6 also failed because `vp_start_and_wait` never
  saw `vp_vector_idle` go HIGH.
- **Fix (`tests/TB_Step4_VectorProcessor.v`):** Changed `mem_dest_address`
  from `16'd0` to `16'h0002` in the `vp_start_and_wait` calls for tests 2,
  3, and 4. The value is architecturally unused for those destination codes
  but must be non-zero to avoid `MEM_ERROR`.

## 2026-06-15 — Build Step 4: Vector_Processor Module

- Built `TPU/PROCESSING/Vector_Processor.v` — 12-state FSM (START, IDLE,
  MEM_ADDR, MEM_WAIT, MEM_FORWARD, VB_RDREQ, VB_WAIT, VB_WRITE, SA_OUT_SEL,
  SA_OUT_WRITE, DONE_STATE, MEM_ERROR, STATE_ERROR) implementing all required
  data routing for the current ISA.
- **Memory abstraction:** ISA address 0x0 always returns 0x00; 0x1 maps to
  TPU_0x1_Buffer at internal offset `elem_cnt`; addresses ≥ 0x2 map to
  Weight_Memory at `(ISA_addr − 2)`.
- **SA loading convention:** VP_A (VDST_SA_A) loads rows of rs1 sequentially
  into top SA input buffers (top_buf[j] = row j); VP_B (VDST_SA_B) loads
  columns of rs2 via strided memory access into left SA input buffers
  (left_buf[i] = col i). The stride multiply (row_cnt × dim1_lat) uses
  zero-extended 4-bit operands to avoid Verilog truncation.
- **SA output readback (VSRC_SA_OUT):** VP_A reads MAC(sa_row=col_cnt,
  sa_col=row_cnt) to reconstruct result[row_cnt][col_cnt] row-major, then
  requantizes (arithmetic right-shift by REQUANT_SHIFT bits, saturate to
  signed 8-bit) and writes linearly to destination memory.
- **Done detection:** uses lookahead comparisons (`row_cnt == dim0_lat − 1 &&
  col_cnt == dim1_lat − 1`) so next-state transitions evaluate the last
  element correctly without needing an extra state.
- **MEM_ERROR:** entered when `i_mem_dest_address == 16'h0000` (write to
  reserved zero register).
- Updated `TPU/TPU.v`:
  - Declared VP_A/VP_B memory, handshake, and SA interface wires.
  - Updated memory MUX: when program=HIGH, wm_address_a/data_a/wren_a and
    buf_address_a/data_a/wren_a are now driven by VP_A (not stubbed to 0).
  - Connected VP_B to WM port b and 0x1 Buffer port b (previously stubbed).
  - Replaced Controller's `i_vector_idle_a/b` stubs with real VP idle signals.
  - Instantiated `Vector_Buffer` with VP_A rdreq and `vb_sclr` connected;
    wrreq and data stubs remain until steps 5/6.
  - Declared `vpa_data`, `vpb_data`, `alu_enable` wires for steps 5/6.
- Created `tests/TB_Step4_VectorProcessor.v` — 7-test testbench. Tests:
  reset idle state; VSRC_MEM→VDST_ACT streaming with element_valid pulses;
  VSRC_MEM→VDST_SA_A correct row-major top buffer loading; VSRC_MEM→VDST_SA_B
  stride column-major left buffer loading; VSRC_VEC_BUF→VDST_MEM VB read/WM
  write; VSRC_SA_OUT→VDST_MEM requantization with saturation; MEM_ERROR on
  write to address 0x0.

## 2026-06-15 — Build Step 3: Controller Module

- Built `TPU/CONTROL/Controller.v` — 11-state FSM (START, IDLE, CLEAR,
  EXEC_VP_START, EXEC_VP_WAIT, SA_START, SA_WAIT, WB_VP_START, WB_VP_WAIT,
  STATE_ERROR, DECODE_ERROR) implementing the DECODE-EXECUTE-WRITEBACK steps
  of the CPU instruction cycle.
- State machine validates the opcode in CLEAR state (entering DECODE_ERROR for
  invalid opcodes), latches the full 128-bit instruction register, then
  proceeds to the execute phase.
- Execute phase: pulses `o_vector_start_a` and (for mult/add) `o_vector_start_b`
  HIGH during EXEC_VP_START; waits in EXEC_VP_WAIT for the required vector
  processors to assert `vector_idle` HIGH. For mult, proceeds to SA_START/
  SA_WAIT before writeback; for relu and add, skips directly to writeback.
- Writeback phase: pulses `o_vector_start_a` in WB_VP_START and waits in
  WB_VP_WAIT for VP_A idle, then reasserts `o_controller_idle` HIGH.
- All VP control signals (`o_vect_source_a/b`, `o_vect_dest_a/b`,
  `o_mem_source_address_a/b`, `o_mem_dest_address_a/b`, `o_length_a/b`,
  `o_dim0_a/b`, `o_dim1_a/b`) are combinationally decoded from the latched
  instruction and current state (execute vs. writeback phase).
- `o_clear` is asserted HIGH for exactly one cycle (CLEAR state) to flush the
  vector buffer, ALU, Activator, and Systolic_Array.
- `o_activator_funct` and `o_alu_funct` are combinationally set based on opcode
  (relu → FUNCT_RELU=3'd0 to Activator; add → FUNCT_ADD=3'd0 to ALU; all others
  → FUNCT_NOOP=3'd7).
- `elem_length` for add is computed as sz1 * sz2 (8x8 product, 16-bit result).
- Updated `TPU/TPU.v` — declared all Controller interface wires (`ctrl_*`),
  declared `vb_sclr = ctrl_clear | ~trst` for future vector buffer connection,
  replaced Feeder's `i_controller_idle` stub with `ctrl_controller_idle`,
  instantiated Controller with VP/SA idle inputs stubbed to 1'b1.
- Created `tests/TB_Step3_Controller.v` — 7-test testbench. Tests: reset state
  (idle HIGH, clear LOW), clear asserted for exactly one cycle, invalid opcode
  → DECODE_ERROR, mult execute-phase VP_A/VP_B decode signals, relu execute-
  phase VP_A signals and VP_B never started, add execute-phase signals with
  correct elem_length=sz1*sz2, controller_idle returns HIGH after writeback.

## 2026-06-15 — Build Step 2: Feeder Module

- Built `TPU/PROGRAMMER/Feeder.v` — 9-state FSM (START, FETCH, WAIT_PM, CHECK,
  FORWARD, WAIT_CTRL, DONE, FEED_ERROR, STATE_ERROR) implementing the FETCH step
  of the CPU instruction cycle.
- State machine retrieves 128-bit instructions from Program_Memory using a program
  counter register (*pc*), handles 1-cycle registered RAM output latency via a
  dedicated WAIT_PM state, detects the syscall termination opcode (4'b0000) and
  enters DONE, and synchronizes with the Controller via a controller_start/
  controller_idle handshake.
- FEED_ERROR is entered when controller acknowledges the instruction at PM_MAX_ADDRESS
  (default 1023) without a prior syscall being found.
- `o_pm_wren` is hardwired LOW (Feeder only reads Program_Memory).
- PM_MAX_ADDRESS is a Verilog parameter (default 10'd1023) to allow small-memory
  override in testbenches.
- Updated `TPU/TPU.v` — declared `feeder_pm_address`, `feeder_pm_wren`,
  `feeder_instruction`, and `feeder_controller_start` wires; updated `pm_address`
  MUX to drive feeder_pm_address when program=HIGH; instantiated Feeder with
  `i_controller_idle` stubbed to 1'b1 (connected to Controller in step 3).
  Wire declarations moved before assign statements to avoid forward-reference warnings.
- Created `tests/TB_Step2_Feeder.v` — 7-test testbench using a behavioral 4-word
  Program_Memory (1-cycle registered output) and a behavioral Controller stub.
  Tests: reset state, syscall DONE (no forward), non-syscall forward, one-cycle
  controller_start pulse, PC increment after ack, correct instruction opcode
  forwarded, FEED_ERROR after all addresses exhausted.

## 2026-06-12 — Programmer.v: SPI Buffer Latency Fix (WAIT/EXEC Refactor)

- **Bug:** The SPI input buffer output (`spi_q`) has 1 cycle of registered latency
  after `rdreq` is asserted. The original state machine read `spi_q` in the same
  state that asserted `rdreq` (old WAIT states), so it always captured the
  *previous* byte. This worked for all bytes except the final STOP code, which had
  no subsequent byte to displace it — causing Test 2 to fail.
- **Fix:** Added a dedicated WAIT state between every POP and EXEC state. The WAIT
  state holds for `BUFFER_LATENCY` clock cycles (default 1) before transitioning to
  EXEC, giving the buffer output time to settle. Added a `wait_count` register to
  implement the parameterizable delay.
- **Renamed:** All previous WAIT states → EXEC states (reflect their actual role).
- **Added parameter:** `BUFFER_LATENCY` (integer, default 1) — number of cycles
  spent in each WAIT state. Increase if buffer latency exceeds 1 cycle.
- **State register width:** Expanded from 5 to 6 bits to accommodate 39 states.
- **Files modified:** `TPU/PROGRAMMER/Programmer.v`



- Built `TPU/PROGRAMMER/Programmer.v` — 30-state FSM decoding the Functional TPU
  Messaging Protocol (START/MEM/PROGRAM/STOP function codes) from the SPI input
  buffer and writing to Program_Memory, Weight_Memory port a, and TPU_0x1_Buffer
  port a. Instantiates `SPI_Interface`. Defines `o_program` (active LOW during
  a programming session) and four error states: STATE_ERROR, COM_ERROR,
  WRITE_ERROR, PROG_OVERFLOW_ERROR.
- Built `TPU/TPU.v` — top-level TPU module. Instantiates Programmer,
  Program_Memory, Weight_Memory, and TPU_0x1_Buffer. Defines `trst = rst & program`.
  Implements memory MUX: Programmer drives all memory inputs when `program` LOW;
  stub zeros drive inputs when `program` HIGH (Feeder and Vector_Processor A stubs
  to be connected in steps 2 and 4).
- Updated `Tensor_Processing_Unit.v` — replaced previous debug SPI_Interface test
  with a TPU instantiation. Added default assignments for VGA and LEDR outputs.
  SPI GPIO pin mapping unchanged (GPIO_0[7/5/3/1]).
- Created `tests/TB_Step1_Programmer.v` — 7-test testbench for the Programmer
  module. Tests: program signal after reset, START/STOP toggling, MEM writes to
  Weight_Memory and 0x1 Buffer, PROGRAM instruction assembly and write,
  COM_ERROR on invalid function code, WRITE_ERROR on address 0x0000.


