# Functional TPU — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

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


