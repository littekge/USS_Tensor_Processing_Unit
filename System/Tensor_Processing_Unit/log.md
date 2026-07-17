# Functional TPU — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-07-17 — Fix: windowed gather (im2col/max) 0x1-buffer addressing

- **Context / root cause:** A LeNet-5 program produced wrong logits on hardware
  while a bit-exact ISA interpreter of the same `PROGRAM.bin` produced the
  correct answer, so the divergence was in the RTL. Diagnosed to the windowed
  (im2col / max) address generation in `TPU/PROCESSING/Vector_Processor.v`: the
  windowed path baked the running element index into `o_dm_address`
  (`win_rd_addr = mem_source_lat + idx`) and forced `o_dm_offset = 0`. For a
  feature map staged in the architecturally isolated `0x1` buffer (the network
  input; `INPUT_ADDRESS = 0x1`), `Data_Memory` decodes only the *exact* address
  `0x1` to the buffer, so only the first element (offset 0) read the buffer and
  every other element decoded to main memory. conv1's `im2col` (`rs1=0x1`, the
  only windowed instruction in the program sourcing `0x1`) therefore read garbage
  for all but the first pixel, corrupting the whole network and every logit. The
  coordinator confirmed the fault bit-exact: modeling this addressing bug in the
  software interpreter reproduces the observed hardware output
  `[-6,2,2,7,-4,3,-6,0,...]`.
- **Fix — `TPU/PROCESSING/Vector_Processor.v`:** In the windowed-address
  combinational block, compute the pure element index `win_idx` and select
  address/offset the same way the non-windowed `src_is_buf` path does:
  out-of-bounds -> address `0x0` (offset 0); source `0x1` -> address held at
  `0x1` with `win_idx[9:0]` on the offset (1024-word buffer / 10-bit offset,
  conv1 max index 783 fits); any other source (addr >= `0x2`) -> flat
  `win_in_addr_full` address with offset 0 (unchanged behavior). The `WIN_ADDR`
  state now drives `o_dm_offset <= win_rd_offset` instead of a hardwired 0. Only
  the windowed path changed; the systolic-array/linear paths and requant are
  untouched. No new `.v` files; no debug sections touched.
- **Tests — `tests/TB_Step4_VectorProcessor.v` (19 -> 21):** added Test 20
  (im2col sourced from `rs1=0x1`: a 3x3 feature map staged at buffer offsets 0..8,
  2x2 window stride 1, dense output checked element-by-element against the
  channel-major expectation `[1,2,4,5, 2,3,5,6, 4,5,7,8, 5,6,8,9]`) and Test 21
  (max stream sourced from `rs1=0x1`, same geometry, stream + `window_end`
  checked). The behavioral `Data_Memory` model already mirrors the real decode
  (only exact `0x1` -> buffer indexed by the offset port), so the pre-fix RTL
  fails both new tests (only element 0 correct) and the post-fix RTL passes.
- **`tests/run_regression.sh`:** TB_Step4 block unchanged — its only RTL
  dependency is `Vector_Processor.v` (still in sync with the TB "How to run"
  header); the fix added no new dependency.
- **Verification: PASS** (Questa PC, 2026-07-17). `./tests/run_regression.sh 4`
  → `TB_Step4_VectorProcessor` 21/21, and the full regression passed with no
  cross-cutting failures (`TB_Step8_FullSystem` included). The fixed RTL was
  re-synthesized and flashed, and the LeNet-5 draw-and-send demo classifies
  correctly on hardware. Bug fix within the completed v0.6 scope, so no `main.md`
  step changes.

## 2026-07-16 — v0.6 Verification: Questa Regression PASS

- **Verification: PASS.** The full regression was run on the Questa PC and passed
  with no errors, promoting the v0.6 build (Steps 1-5) from PENDING to verified.
- v0.6 testbenches pass at the counts documented in the Step 5 entry below:
  `TB_Pooler` 9/9, `TB_Step3_Controller` 17/17, `TB_Step4_VectorProcessor` 19/19,
  `TB_Step8_FullSystem` 15/15; the unchanged Step1/2/5/6/7 and SPI benches pass
  with no regressions.
- `main.md` v0.6 Steps 1-5 marked complete.
- Verification-only entry — no RTL changes.

## 2026-07-16 — v0.6 Step 5: Regression runner synced; full v0.6 verification PENDING

- **Context:** v0.6 Step 5 — sync `tests/run_regression.sh` with the v0.6
  testbench changes and record the deferred verification state. No RTL changes.
- **`tests/run_regression.sh` (already updated across Steps 1/4, confirmed in
  sync):**
  - New `run_tb pool TB_Pooler` block (sources `TPU/PROCESSING/Pooler.v`,
    `tests/TB_Pooler.v`).
  - `TB_Step8_FullSystem` source list gained `TPU/PROCESSING/Pooler.v` (TPU now
    instantiates the Pooler). `TB_Step3_Controller` and
    `TB_Step4_VectorProcessor` blocks unchanged (their RTL dependency lists —
    `Controller.v` and `Vector_Processor.v` respectively — did not change, only
    the testbench contents). Each block's source list matches its testbench
    "How to run" header.
  - Ran `./tests/run_regression.sh` here: it self-gated (Questa absent) and
    exited with the deferral notice, as expected — no simulation attempted.
- **Verification: PENDING (Linux laptop — no `vsim.exe` at
  `C:\intelFPGA_lite\23.1std\questa_fse\win64\`).** On the Questa PC run
  `./tests/run_regression.sh`; expected new/changed counts:
  - `TB_Pooler` 9/9 (new)
  - `TB_Step3_Controller` 17/17 (13 -> 17: WINCONFIG latch, im2col/max decode,
    destination select, pooler funct, im2col skips writeback)
  - `TB_Step4_VectorProcessor` 19/19 (16 -> 19: im2col zero-fill gather, max
    window stream + window_end, max min-fill)
  - `TB_Step8_FullSystem` 15/15 (13 -> 15: end-to-end convolution im2col+matmul,
    max pooling)
  - plus the unchanged Step1/2/5/6/7 and SPI benches for regressions
  (pass = each `Results:` line with 0 FAIL and no `Test N: FAIL`).
  The v0.6 `main.md` Steps 1-5 are left UNMARKED until that run confirms.
- **Open item surfaced (see report):** the user-provided `Pooler.v` was an empty
  0-byte file rather than the port-list skeleton the v0.6 prerequisite describes;
  its header/ports were authored from the Hardware Spec + ALU/Activator
  convention. The user should confirm the authored interface matches intent.

## 2026-07-16 — v0.6 Step 4: TPU top-level integration (Pooler + windowed routing)

- **Context:** v0.6 Step 4 wires the Pooler into `TPU/TPU.v` and routes the
  Controller window-descriptor/pooler-function nets to the vector processors and
  Pooler, per the Hardware Spec v0.6.0 (TPU Pooler / Vector Buffer bullets) and
  `main.md` v0.6 Step 4. RTL + `tests/TB_Step8_FullSystem.v` updated. No new `.v`
  files; no debug sections touched.
- **`TPU/TPU.v`:**
  - Instantiated the Pooler: `i_data` = vector processor a data, `i_enable` =
    a's `element_valid`, `i_window_end` = a's new `o_window_end`, `i_clear` =
    `ctrl_clear` (inter-operation flush), `i_pooler_op` = `ctrl_pooler_funct`,
    `i_trst` = `trst` (Pooler added to the trst reset list).
  - Added the Pooler as a third vector-buffer writer: `vb_wrreq = act_write |
    alu_write | pool_write`, `vb_data_in` three-way mux (only one is non-NO-OP
    per instruction).
  - Routed the Controller's eleven `o_win_*` descriptor nets to VP_A's window
    inputs (VP_B's tied to 0 — windowed instructions use VP_A only) and
    `o_pooler_funct` to the Pooler.
  - Added the `vpa_window_end` net; VP_B's `o_window_end` left unconnected.
- **Tests — `tests/TB_Step8_FullSystem.v` (13 -> 15):** new `mk_window`/
  `mk_im2col`/`mk_max` builders. Test 14 = convolution: a 3x3 feature map is
  expanded by im2col (2x2 window, stride 1) to a 4x4 dense matrix and multiplied
  by a 1x4 all-ones kernel, giving per-window sums [12,16,24,28] end to end
  (FLASH -> im2col -> matmul -> requant -> memory). Test 15 = max pooling: a 4x4
  feature map (1..16), 2x2 window stride 2, pooled to [6,8,14,16] through the
  windowed VP -> Pooler -> vector buffer -> writeback path.
- **`tests/run_regression.sh`:** added `TPU/PROCESSING/Pooler.v` to the
  `TB_Step8_FullSystem` source list (TPU now instantiates the Pooler), in sync
  with the TB "How to run" header.
- **Verification: PENDING** (Linux laptop — no `vsim.exe`; `run_regression.sh`
  self-gates). On the Questa PC run `TB_Step8_FullSystem` (expect 15/15).
  `main.md` v0.6 Step 4 left UNMARKED until the regression passes there.

## 2026-07-16 — v0.6 Step 3: Vector_Processor windowed addressing (im2col / max)

- **Context:** v0.6 Step 3 adds windowed address generation to
  `TPU/PROCESSING/Vector_Processor.v` per the Hardware Spec v0.6.0
  (Vector_Processor Windowed Addressing) and ISA v0.6 (im2col / max gather
  order, fill values). RTL + `tests/TB_Step4_VectorProcessor.v` updated. No new
  `.v` files; no debug sections touched.
- **`TPU/PROCESSING/Vector_Processor.v`:**
  - Added the eleven window-descriptor inputs (`i_win_chans`/`inh`/`inw`/`outh`/
    `outw` [12-bit]; `winh`/`winw`/`strh`/`strw`/`padh`/`padw` [4-bit]), latched
    in IDLE at `vector_start`; added the `o_window_end` output.
  - Added `VSRC_MEM_WIN = 3'd3` and `VDST_POOL = 3'd6`; widened the state
    register to 5 bits; added windowed states `WIN_ADDR`/`WIN_WAIT`/`WIN_WAIT2`/
    `WIN_FORWARD`/`WIN_WRITE`.
  - Windowed gather: for each window element, derives the input coordinate
    ih = oh*strh + wr - padh, iw = ow*strw + wc - padw (via non-negative bases
    with a pad-region compare), bounds-checks against inh x inw, and substitutes
    the fill when out of bounds — read steered to 0x0 (returns 0) for im2col,
    minimum representable value (0x80) for max. Element count is descriptor-
    derived (chans*winh*winw*outh*outw), not from length.
  - `im2col` (dest VDST_MEM): iterates ch,wr,wc,oh,ow (output positions inner)
    and writes every gathered element contiguously to dest, yielding the dense
    (chans*winh*winw) x (outh*outw) Row-major matrix. `max` (dest VDST_POOL):
    iterates ch,oh,ow,wr,wc (window inner), streams each element to the Pooler
    via o_data/o_element_valid, and asserts `o_window_end` coincident with
    `element_valid` on each window's final element.
  - v0.5 contiguous/strided paths (MEM/SA/VB) and the requant datapath are
    unchanged.
- **Tests — `tests/TB_Step4_VectorProcessor.v` (16 -> 19):** new window-descriptor
  drive registers + `o_window_end` connection, and a windowed-stream capture
  block (element data + window_end). Test 17 = padded im2col zero-fill, dense
  matrix written Row-major to dest ([0,0,0,4,0,0,3,0,0,2,0,0,1,0,0,0]). Test 18
  = max window stream (4x4 map, 2x2/stride2) to the Pooler dest with window_end
  on each window's 4th element; descriptor-derived 16-element count. Test 19 =
  padded max min-fill (0x80) on out-of-bounds elements. Tests 3/4/15/16 (v0.5
  contiguous/strided) unchanged.
- **`tests/run_regression.sh`:** unchanged for Step 4 (RTL dependency list is
  still just `Vector_Processor.v` + the testbench).
- **Verification: PENDING** (Linux laptop — no `vsim.exe`; `run_regression.sh`
  self-gates). On the Questa PC run `TB_Step4_VectorProcessor` (expect 19/19).
  `main.md` v0.6 Step 3 left UNMARKED until the regression passes there.

## 2026-07-16 — v0.6 Step 2: Controller window descriptors + windowed decode

- **Context:** v0.6 Step 2 adds the window descriptor registers and windowed-
  instruction handling to `TPU/CONTROL/Controller.v` per the Hardware Spec v0.6.0
  (Controller Window Descriptor Registers) and ISA v0.6 (W/A-format field
  positions). RTL + `tests/TB_Step3_Controller.v` updated. No new `.v` files; no
  debug sections touched.
- **`TPU/CONTROL/Controller.v`:**
  - Opcodes added: `OP_MAX = 1100` (POOL), `OP_IM2COL = 1101` (SHAPE),
    `OP_WINDOW = 1110` (WINCONFIG). Source/dest encodings added:
    `VSRC_MEM_WIN = 3'd3` (windowed gather), `VDST_POOL = 3'd6` (Pooler input).
    Function code `FUNCT_MAX = 3'd0`.
  - Eleven window descriptor registers (`chans`, `inh`, `inw`, `outh`, `outw`
    [12-bit]; `winh`, `winw`, `strh`, `strw`, `padh`, `padw` [4-bit]), latched
    from the W-format fields of a `window` instruction in the CLEAR state (same
    timing as `stride`'s ld1/ld2 latch, from `i_instruction`), retained until the
    next `window`; cleared to 0 on `trst`.
  - Generalized the register-configuration decode path: CLEAR returns to IDLE
    (no execute/writeback) for both `stride` (LDCONFIG) and `window` (WINCONFIG).
  - `im2col`/`max` decode added to the VP control block: both set VP_A source
    `VSRC_MEM_WIN` from src1 (bits 123-100). `im2col` dest `VDST_MEM` at rd
    (bits 99-76); `max` execute dest `VDST_POOL`, writeback `VSRC_VEC_BUF` ->
    `VDST_MEM` at rd with `length = chans*outh*outw`. `EXEC_VP_WAIT` routes
    `im2col` straight to IDLE (writes to memory during Execute — skips Writeback).
    `single_vp_op` (relu/im2col/max) gates `all_exec_vps_done` to VP_A only.
  - Outputs added: `o_pooler_funct` (MAX during a `max`, NO-OP otherwise) and the
    eleven `o_win_*` descriptor outputs (mirror the persistent registers).
- **Tests — `tests/TB_Step3_Controller.v` (13 -> 17):** new `make_window`/
  `make_im2col`/`make_max` builders (W/A-format), new port connections, a
  `vector_start_a` pulse counter. Test 14 = WINCONFIG latches all eleven
  registers, no vector_start. Test 15 = im2col routes VSRC_MEM_WIN->VDST_MEM at
  dest, pooler NO-OP, skips writeback (exactly one vector_start_a pulse, VP_A
  never reads SA/vector-buffer). Test 16 = max execute VSRC_MEM_WIN->VDST_POOL
  from src1, pooler MAX. Test 17 = max writeback VSRC_VEC_BUF->VDST_MEM at dest,
  length = chans*outh*outw (24 for the preceding window).
- **`tests/run_regression.sh`:** unchanged for Step 3 (its RTL dependency list is
  still just `Controller.v` + the testbench).
- **Verification: PENDING** (Linux laptop — no `vsim.exe`; `run_regression.sh`
  self-gates). On the Questa PC run `TB_Step3_Controller` (expect 17/17).
  `main.md` v0.6 Step 2 left UNMARKED until the regression passes there.

## 2026-07-16 — v0.6 Step 1: Pooler module (streaming max-pooling reducer)

- **Context:** v0.6 Step 1 implements `TPU/PROCESSING/Pooler.v`, the streaming
  reducer peer to the ALU/Activator, per the Hardware Specification v0.6.0
  (Pooler description) and `main.md` v0.6 Step 1.
- **Skeleton was EMPTY (surfaced):** the user-provided `Pooler.v` was a 0-byte
  file, not the "module declaration + port list + parameter/code/debug sections"
  skeleton the v0.6 prerequisite describes. The module header and port list were
  authored from the Hardware Specification (Pooler control-signal list) plus the
  established `ALU.v`/`Activator.v` convention (i_/o_ naming, `i_pooler_op`
  mirroring `i_alu_op`/`i_activator_op`, `o_write` to the vector buffer). The
  DEBUG section is present but empty for the user to fill. **User should confirm
  the authored interface matches intent.**
- **`TPU/PROCESSING/Pooler.v` (implemented):**
  - Ports: `i_clk`, `i_trst`, `i_enable`, `i_clear`, `i_window_end`,
    `i_pooler_op[2:0]`, `i_data[7:0]`; `o_write`, `o_data[7:0]`. Unlike ALU/
    Activator, `i_clk` is connected (the module is sequential — it holds an
    accumulator across a window's stream).
  - Function params `MAX = 3'h0` (POOL funct3 0x0) and `NOOP = 3'h7`; identity
    `MIN_VALUE = 8'h80` (-128, min representable signed 8-bit).
  - `acc` register: reset to `MIN_VALUE` on `i_trst` LOW or `i_clear`; while
    `i_enable` HIGH, folds each input via a signed `max` comparator, and on
    `i_window_end` resets to identity for the next window.
  - `o_data` combinational = `max(acc, i_data)` so on the `window_end` cycle it
    holds the fully reduced window at the same time `o_write` pulses (matches the
    vector buffer's single-cycle capture). `o_write = i_enable & i_window_end &
    (op != NOOP)` — exactly one buffer write per window, suppressed for NO OP.
  - No requantization (pooled value is an input value already in the datatype).
- **Tests:** `tests/TB_Pooler.v` (NEW, 9 tests): single-element window (proves
  identity is min not 0), clear resets accumulator, multi-element window max,
  two back-to-back windows reset independently, min-fill (-128) never wins,
  all-fill window reduces to -128, NO-OP write suppression, `o_write` LOW
  mid-window, `o_write` LOW when disabled. Follows the post-v0.1 dedicated-unit-
  testbench convention (`TB_SPI_Slave.v`).
- **`tests/run_regression.sh`:** added a `run_tb pool TB_Pooler` block (sources
  `Pooler.v`, `tests/TB_Pooler.v`), in sync with the TB header dependency list.
- **Verification: PENDING** (Linux laptop — no `vsim.exe` at
  `C:\intelFPGA_lite\23.1std\questa_fse\win64\`; `run_regression.sh` self-gates).
  On the Questa PC run `TB_Pooler` (expect 9/9). `main.md` v0.6 Step 1 left
  UNMARKED until the regression passes there.

## 2026-07-14 — v0.5.1 SPI_Slave rewrite verified (Questa PASS + hardware)

- **Verification: PASS.** Full Questa regression passed on the Questa PC —
  the new `TB_SPI_Slave` plus `TB_Step1_Programmer` and `TB_Step8_FullSystem`
  (system-level functional equivalence of the rewritten module). Supersedes the
  `Verification: PENDING` recorded in the rewrite entry below.
- **Hardware-validated:** with the resynthesized bitstream, Bigger_NN flashed
  successfully **3 times in a row with zero errors** over the real Arduino→FPGA
  SPI link (previously unreliable at that size). The synchronous oversampling
  receiver tolerates the marginal ~100 ns level-shifter edges; the planned
  logic-level converter is now an optional backup rather than a requirement.
- **Steps:** `main.md` v0.5.1 Steps 1–3 marked complete.
- **Deferred (user):** the Hardware-Spec `SPI_Slave` description ("pulled from
  nandland") is knowingly stale; the user will update it before the v0.5.1 tag
  (`Specifications/` is user-managed / read-only for Claude).

## 2026-07-14 — v0.5.1 SPI_Slave synchronous-oversampling rewrite (Verification: PENDING)

- **What was done:** rewrote the internals of
  `TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v` as a fully synchronous oversampling
  receiver in the `i_Clk` (50 MHz) domain, replacing the nandland design that
  used the raw SPI clock as a real clock and raw CS as an asynchronous reset
  with a cross-domain "byte done" flag. Eliminates the glitch-triggered
  spurious clocking, the last-byte/CS-deassert race, and the CDC exposure that
  made large-model SPI flashing unreliable on the passive 5V→3.3V path (see the
  2026-07-14 v0.5 known-issue entry).
- **New architecture (internals only):**
  - 2-stage FF metastability synchronizers on `i_SPI_Clk`, `i_SPI_MOSI`,
    `i_SPI_CS_n` (reset to idle SCK level / deasserted CS).
  - Debounce of SCK and CS via `localparam DEBOUNCE_N = 3`: a level change is
    accepted only after N consecutive equal synchronized samples (~60 ns),
    rejecting slow-edge bounce and short glitches. Edge-detect on the debounced
    level.
  - CPOL/CPHA derived from `SPI_MODE` (localparams `CPOL`/`CPHA`) select the
    sample edge (capture MOSI) and shift edge (advance MISO), covering modes
    0–3.
  - RX: on the sample edge, shift synchronized MOSI MSB-first into an 8-bit
    register and advance a 0..7 counter; on the 8th bit latch `o_RX_Byte` and
    pulse `o_RX_DV` for exactly one `i_Clk` cycle.
  - Synchronous CS framing: CS-assert resets the RX bit counter; CS-deassert
    NEVER asynchronously clears received-byte/DV state (this fixes the
    last-byte race). Partial byte discarded at the next frame start.
  - TX/MISO driven synchronously off the shift edge: MSB preload on CS-assert,
    MSB-first shift, tri-state when synchronized CS is high, `i_TX_Byte`
    registered on `i_TX_DV`.
- **Contract preserved (byte-identical interface):** module ports, the
  `#(parameter SPI_MODE = 0)` declaration, and functional behavior are
  unchanged — verified against `SPI_Interface.v` (which ties `i_TX_DV`=0 and
  reads `o_RX_DV`/`o_RX_Byte` into the FIFO `wrreq`/`data`). `o_RX_DV` remains a
  single-cycle pulse (critical: it drives `wrreq`); MSB-first assembly;
  multi-byte per CS window; active-low reset; full mode 0–3 semantics. DEBUG
  section untouched; no other `.v` files changed.
- **Files created/modified:**
  - `TPU/PROGRAMMER/SPI_LINK/SPI_Slave.v` — internals rewritten, header comment
    updated to describe the synchronous-oversampling architecture and state the
    interface/behavior are unchanged.
  - `tests/TB_SPI_Slave.v` — NEW dedicated unit testbench (14 tests): MSB-first
    assembly, single-cycle DV, multi-byte and back-to-back within one CS window,
    SCK-glitch rejection, CS-glitch rejection, noisy/bouncing edges, last-byte
    CS-deassert race, DV-exactly-one-cycle over a burst, reset clearing, TX/MISO
    behavior, and mode 1/2/3 correct-edge smoke checks. Four DUTs (modes 0–3)
    share the SPI stimulus nets.
  - `tests/run_regression.sh` — added a `run_tb spi TB_SPI_Slave` block
    (sources: `SPI_Slave.v`, `tests/TB_SPI_Slave.v`), in sync with the TB
    header dependency list.
- **Verification: PENDING (non-Questa machine — Linux laptop, no simulation
  run).** On the Questa PC run `TB_SPI_Slave` (new), plus `TB_Step1_Programmer`
  and `TB_Step8_FullSystem` (system-level functional-equivalence, expected to
  still pass unchanged since the interface is preserved). v0.5.1 `main.md` steps
  left UNMARKED until the regression passes on the Questa PC.
- **Spec note:** the `SPI_Slave` description in the Hardware Specification
  (currently "pulled from nandland") is knowingly left stale; per the user this
  spec update is deferred to hardware bring-up.

## 2026-07-14 — v0.5 verified on Questa; build steps marked complete (release-ready)

- **Regression PASS (user-verified on the Questa PC, 2026-07-14):** the full v0.5
  regression — `TB_Step3` (Controller), `TB_Step4` (Vector_Processor), `TB_Step7`
  (Systolic_Array), `TB_Step8` (FullSystem) — all pass. Resolves the
  `Verification: PENDING` state carried since the 2026-07-13 implementation.
- **Release hygiene:** `TPU.v` DEBUG section is empty (no `o_debug_val`/debug
  ports); no new `.v` files; debug sections untouched. `main.md` v0.5 Steps 1–5
  marked ✅ Complete.
- **Functional confirmation:** on the flash attempts that completed end-to-end,
  every neural network computed correctly with the new leading-dimension
  partitioning.
- **Known issue (NOT a v0.5 defect — tracked separately):** large-model SPI
  flashing is unreliable from a signal-integrity limitation on the 5V→3.3V path:
  the passive 2k/3.3k resistive divider drives ~100 ns edges into the FPGA's
  asynchronous SPI clock (SCK) / reset (CS) pins. Orthogonal to the v0.5
  partitioning logic (verified by Questa and by successful flashes). Fix pending
  a proper buffered level converter (+ optional `SPI_Slave.v` deglitch/CS
  synchronize); see the comms flashing notes.
- **Files modified:** `main.md` (Steps 1–5 marked complete), `log.md`.

## 2026-07-13 — v0.5 review fixes: TB_Step8 debug port removed; multip carry-path test added

- **Context:** two review fixes on the v0.5 partitioning worktree (branch
  `worktree-wf_7feca0ff-b9b-1`, RTL committed at fc4b325). Test-only changes;
  no RTL edited, no debug sections touched, no new `.v` files.

- **Fix 1 — `tests/TB_Step8_FullSystem.v` debug port (test-only):** the user
  removed the `o_debug_val` output from `TPU/TPU.v` (its DEBUG section is now
  empty), so the TB's `.o_debug_val(debug_val)` connection referenced a
  nonexistent port and would fail elaboration. Removed the port connection from
  the TPU instantiation, dropped the now-trailing comma on `.o_end_reached()`
  so it is the last port, and removed the unused `wire [31:0] debug_val;`
  declaration (a note replaces it). No other reference to `debug_val` existed.
  This resolves the debug-section discrepancy flagged in the prior entry.

- **Fix 2 — multip accumulation carry-path verification:**
  - **Mechanism (why stale products are NOT accumulated across the
    inter-operation gap), cited:**
    - `Multiply_Accumulate_Unit.v:75` accumulates `o_c <= o_c +
      $signed(i_a)*$signed(i_b)` every cycle whenever `i_accumulator_clear` is
      LOW — so protection must come from the operands `i_a`/`i_b` being 0 during
      the gap, making the addend `0*0 = 0`.
    - `Systolic_Array.v:129` and `:148` gate the array edges:
      `h_interconnect[0][j] = left_rdreq_d[j] ? left_q[j] : 8'd0` and
      `v_interconnect[i][0] = top_rdreq_d[i] ? top_q[i] : 8'd0`. Genuine popped
      data reaches an edge only while its delayed read-valid is HIGH; otherwise
      the edge is fed 8'd0.
    - `Systolic_Array.v:257-272` decode `top_rdreq`/`left_rdreq` = 0 unless
      `S == LOAD`; during the idle gap (`S == IDLE`) no reads occur, so
      `top_rdreq_d`/`left_rdreq_d` (`:234-251`) fall to 0 and both edges feed 0.
      Those 0s propagate through the MAC operand registers, so within N cycles
      every MAC sees a 0 operand and adds nothing.
    - `Controller.v:281` (`o_clear = (S == CLEAR)`) pulses clear at the START of
      EVERY operation (`IDLE -> CLEAR`, `:200`), including `multip`. Clear
      drives `buf_sclr` (`Systolic_Array.v:71`, flushes input buffers), resets
      `top_rdreq_d`/`left_rdreq_d` (`:241-245`), and zeroes each MAC's operand
      registers `o_a`/`o_b` (`Multiply_Accumulate_Unit.v:52-56`) — immediately
      forcing the whole datapath feeding the accumulator to 0.
    - **Holds with the persistent (multip) accumulator:** v0.5 split clear from
      accumulator-clear. `o_accumulator_clear` pulses only in `ACC_CLEAR` and
      only for `mult`, not `multip` (`Controller.v:298`), so a `multip` keeps
      `o_c`. But `o_clear` still flushes operands/buffers every op, so during the
      gap `o_c <= o_c + 0`: the accumulator is preserved intact and no stale
      product is added. Real tile operands streamed on the next op accumulate on
      top — the intended tiled sum.
  - **Test coverage added:** `tests/TB_Step7_SystolicArray.v` new Test 10
    (9 -> 10 tests). Tile 1 (top_buf[0]=2 x2, left_buf[0]=3 x2) accumulates P1
    into MAC(0,0); then a one-cycle `clear` (accumulator_clear held LOW, as for
    multip) plus a 20-cycle idle gap; asserts the accumulator still reads P1
    after the gap (no stale leak). Tile 2 uses identical operands, so the final
    accumulator must be exactly `2*P1`. FAILS if residual products leaked during
    the gap (after_gap would exceed P1 and final would exceed 2*P1). Existing
    Test 8 only checks clear-preserves / accumulator_clear-zeroes over 2 cycles;
    Test 10 is the directed multip run (persistent accumulator + long gap +
    second tile) that Test 8 does not cover.

- **`tests/run_regression.sh`:** no change — neither TB_Step7 nor TB_Step8
  changed its RTL dependency file LIST (only testbench contents changed); no
  pass-count is hardcoded in the runner.

- **Verification: PENDING.** Not the Questa PC (no `vsim.exe` at
  `C:\intelFPGA_lite\23.1std\questa_fse\win64\`); nothing simulated,
  `run_regression.sh` self-gates. Affected testbenches to run on the Questa PC:
  `TB_Step7_SystolicArray` (now 10/10 expected, Test 10 = multip carry-path) and
  `TB_Step8_FullSystem` (13/13 expected; elaborates now that the dead debug port
  is gone). v0.5 `main.md` steps left UNMARKED until that run confirms.

## 2026-07-13 — v0.5 Partitioning Update (Leading-Dimension Addressing): RTL + testbenches

- **Context:** v0.5 enables matmuls larger than the 8x8 systolic array by tiling,
  with sub-block operands addressed in place via leading dimensions. Implements
  the `multip` instruction (opcode 1000, funct3 0x1) with
  accumulator-clear-after-writeback, the `stride` instruction (CONFIG format,
  opcode 1111) loading `ld1`/`ld2`, and strided operand addressing in the
  Vector_Processor. Field positions per `Functional_TPU_ISA.md` v0.5 (MUL M0
  [35:28] / n [27:20] / funct3 [2:0]; CONFIG im1 [123:100] / im2 [99:76] /
  funct3 [2:0]). RTL for all four build steps done; testbenches updated. No new
  `.v` files; no debug sections touched.

- **`TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v`** (Step 1) — split the clear
  semantics. `i_clear` now resets only the operand pipeline registers (`o_a`,
  `o_b`); a new `i_accumulator_clear` input zeroes only the accumulator `o_c`;
  `trst` still resets all three. The accumulate `o_c <= o_c + $signed(i_a)*
  $signed(i_b)` runs whenever accumulator_clear is deasserted, so the
  accumulator survives an inter-operation clear and can carry tiled partial
  products across a `multip` run until a terminating `mult` clears it.
- **`TPU/SYSTOLIC_ARRAY/Systolic_Array.v`** (Step 1) — added `i_accumulator_clear`,
  routed to every MAC. `i_clear` still drives `buf_sclr` (input-buffer flush),
  the `top_rdreq_d`/`left_rdreq_d` pipeline reset, and each MAC's operand-clear.
- **`TPU/CONTROL/Controller.v`** (Step 2) — added `funct3` decode on opcode 1000
  (`FUNCT3_MULT` 0x0 / `FUNCT3_MULTIP` 0x1). Added an `ACC_CLEAR` state after
  `WB_VP_WAIT`; new output `o_accumulator_clear` pulses HIGH for one cycle in
  `ACC_CLEAR` only when the MUL funct3 selects clearing (`mult`, not `multip`).
  Added `ld1`/`ld2` 24-bit registers, reset to 0 on `trst`, latched from
  `im1`/`im2` when a `stride` (opcode 1111) is decoded in `CLEAR`; `stride`
  returns straight to `IDLE` (no execute/writeback). New outputs
  `o_leading_dimension_a`/`_b` supply the active stride to the VPs: `ld1` for
  src1 load, `ld2` for src2 load, `ld2` at writeback; 0 for non-MUL.
- **`TPU/PROCESSING/Vector_Processor.v`** (Step 3) — added `i_leading_dimension`
  (latched to `ld_lat`). Generalized operand-load and writeback addressing:
  `row_stride = (ld_lat==0)? dim1_lat : ld_lat`, `block_offset = row_cnt*
  row_stride + col_cnt`. SA-load reads (`VDST_SA_A`/`VDST_SA_B`) and SA-output
  writeback (`VSRC_SA_OUT`) use `block_offset`; linear (relu/add, VB writeback)
  ops still walk `elem_cnt` contiguously. With ld=0 the stride equals the matrix
  width, reproducing v0.4 addressing exactly (replaces the old `sa_b_stride`).
- **`TPU/TPU.v`** (Step 4) — routed `ctrl_accumulator_clear` to
  `Systolic_Array.i_accumulator_clear` (independent of the shared `ctrl_clear`),
  and `ctrl_leading_dimension_a`/`_b` to the two VPs' `i_leading_dimension`.

- **Tests (written/updated; NOT simulated — see Verification):**
  - `tests/TB_Step7_SystolicArray.v` (Test 8 reworked): drives new `i_accumulator_
    clear`; verifies the MAC accumulator SURVIVES a `clear` pulse (value
    unchanged) and is zeroed only by `accumulator_clear`. 9 tests.
  - `tests/TB_Step3_Controller.v` (10 -> 13): added `make_mul_f3`/`make_multip`/
    `make_stride` builders and connected `o_accumulator_clear`/
    `o_leading_dimension_a`/`_b`. Test 11 = accumulator_clear pulses once for
    `mult`, zero for `multip` (both decode/run). Test 12 = `stride` latches
    ld1/ld2 with no vector_start. Test 13 = after a stride, a MUL presents ld1
    on leading_dimension_a and ld2 on leading_dimension_b at load, ld2 at
    writeback.
  - `tests/TB_Step4_VectorProcessor.v` (14 -> 16): added `i_leading_dimension`
    port and a `stride_mode` SA stub. Test 15 = strided 2x2 sub-block read from
    a 4-wide parent (ld=4). Test 16 = strided 2x2 writeback into a 4-wide parent
    (row 1 lands at base+4, mem[4]/mem[5] untouched). Tests 3/4 (ld=0) confirm
    contiguous == v0.4.
  - `tests/TB_Step8_FullSystem.v` (12 -> 13): added `mk_mul_f3`/`mk_multip`/
    `mk_stride` and a burst `send_mem_block` helper. Test 13 = tiled matmul
    C(2x2)=A(2x16)*B(16x2) with K=16 in two K=8 passes (a `multip` accumulating
    the first tile terminated by a `mult`), `stride` ld1=16 so each 2x8 A
    sub-block is read in place; requant (M0=1,n=4) reference [[1,2],[2,4]].
  - `tests/run_regression.sh`: no change — no testbench's RTL dependency file
    LIST changed (TB_Step3 -> Controller.v; TB_Step4 -> Vector_Processor.v;
    TB_Step7 -> MAC + SAIB + Systolic_Array; TB_Step8 -> full hierarchy), only
    the contents of already-listed files.

- **Verification: PENDING.** This is not the Questa PC (no `vsim.exe` at
  `C:\intelFPGA_lite\23.1std\questa_fse\win64\`), so nothing was simulated;
  `run_regression.sh` self-gates. Expected on the Questa PC via
  `./tests/run_regression.sh`: TB_Step7 9/9, TB_Step3 13/13, TB_Step4 16/16,
  TB_Step8 13/13, plus the unchanged Step1/2/5/6 for regressions (pass = each
  `Results:` line with 0 FAIL and no `Test N: FAIL`). The v0.5 `main.md` steps
  are left UNMARKED until that run confirms.
- **Notes / risks for the Questa run:**
  - Test 13 (tiled accumulation) depends on the systolic array not
    over-accumulating stale products across the inter-operation idle gap between
    the `multip` and the terminating `mult` (the accumulator must hold exactly
    the first tile's result). The v0.4 Test 10 (two consecutive mults, no
    doubling) is evidence the array zeroes cleanly after drain, but the
    multip->mult carry path is exercised for the first time here.
  - Pre-existing: `TB_Step8_FullSystem.v` connects `.o_debug_val(debug_val)` on
    the TPU, but the committed `TPU/TPU.v` has an empty DEBUG section with no
    `o_debug_val` port. This is a debug-section (user-owned, machine-local)
    discrepancy, not introduced by this change and not editable here; the
    Questa-PC copy presumably defines it. Surfaced for the user.

## 2026-07-09 — v0.4 Memory Expansion: full Questa regression PASS

- **Ran the pending v0.4 regression** on the Questa PC (`vsim.exe` confirmed at
  `C:\intelFPGA_lite\23.1std\questa_fse\win64\`) via `./tests/run_regression.sh`
  — the step the prior sessions could not run (they were on non-Questa machines
  and recorded `Verification: PENDING`).
- **Results — all PASS, no regressions:** TB_Step1_Programmer 19/19,
  **TB_Step1_DataMemory 10/10** (new), TB_Step2_Feeder 9/9, TB_Step3_Controller
  10/10, TB_Step4_VectorProcessor 14/14, TB_Step5_Activator 7/7, TB_Step6_ALU
  7/7, TB_Step7_SystolicArray 9/9, TB_Step8_FullSystem 12/12. Runner reported
  `ALL PASS`. No compile/elaboration errors or fatals across the suite.
- This confirms the merged v0.4 RTL (flat 24-bit dual-port `Data_Memory`
  wrapper, unified addressing threaded through TPU/Vector_Processor/Controller/
  Programmer/Feeder, 8192-word program memory, 3-byte MEM address) against the
  updated testbenches. The `TB_Step1_DataMemory` decode/latency cases and the
  flat-addressing rework in Step1_Programmer/Step4/Step8 all hold on hardware-IP
  simulation.
- **`main.md`:** all six v0.4 build steps (Data_Memory wrapper, TPU rewire,
  Vector_Processor widening, Programmer 3-byte MEM, Feeder/PM depth, Regression)
  marked ✅ Complete. v0.4 build done.

## 2026-07-09 — v0.4 Memory Expansion: TB_Step3_Controller ISA-layout update

- **Context:** follow-up to the entry below (coordinator-approved). The prior
  session flagged `tests/TB_Step3_Controller.v` as still using the pre-v0.4
  16-bit ISA field layout while the merged `Controller.v` decodes the v0.4
  24-bit-field layout. Updated it to match; same mechanical treatment applied to
  TB_Step8. Confirmed this is a pure testbench-layout mismatch, NOT an RTL bug:
  every expected decode value in Step 3 lines up with the v0.4 field positions
  (e.g. add `elem_length = sz1*sz2 = 12`; mult dims sz11/sz12/sz21/sz22 at
  99-96/95-92/67-64/63-60; M0/n at 35-28/27-20), so no RTL was touched.
- **`tests/TB_Step3_Controller.v`.** Rewrote `make_mult_rq`/`make_mult`/
  `make_relu`/`make_add` to the ISA v0.4 128-bit layout: rs1[123:100], MUL
  rs2[91:68]/sz11[99:96]/sz12[95:92]/sz21[67:64]/sz22[63:60]/rd[59:36]/
  M0[35:28]/n[27:20], ELEM rs2[99:76]/sz1[75:68]/sz2[67:60]/rd[59:36], ACT
  rd[99:76]/len[75:60]; opcode[127:124], reserved bits (incl. funct3) 0.
  Widened the DUT address wires (`mem_source_address_a`/`mem_dest_address_a`/
  `mem_source_address_b`) and the captured-address regs 16 -> 24 bits, and the
  instruction/expected address literals to 24-bit. Length stays 16-bit; sz/M0/n
  stay as-is. All 10 cases preserved.
- **`tests/run_regression.sh`.** Step 3 block unchanged — its only RTL
  dependency is `Controller.v` (plus the testbench), which did not change; the
  block already matches the "How to run" header.
- **Verification: PENDING.** No Questa on this machine. Expected on the Questa PC:
  `TB_Step3_Controller` 10/10 (pass = `Results:` line with 0 FAIL and no
  `Test N: FAIL`). v0.4 `main.md` steps stay UNMARKED until the full regression
  passes there.

## 2026-07-09 — v0.4 Memory Expansion: testbenches (unblocked)

- **Context:** the prior v0.4 entry left testbench work BLOCKED by a permission
  glob that made the section `tests/` unwritable. That is fixed; this session
  places/updates the v0.4 testbenches against the merged RTL (read as ground
  truth: `Data_Memory.v`, `TPU.v`, `Vector_Processor.v`, `Programmer.v`,
  `Feeder.v`, `Controller.v`). Linux laptop — no `vsim.exe`/`altera_mf` — so
  **nothing was simulated**; `run_regression.sh` self-gates.

- **`tests/TB_Step1_DataMemory.v` (NEW).** 10 cases for the `Data_Memory`
  wrapper: zero-register read (0x0), 0x1-buffer routing via `i_offset`, offset
  independence, block A select (addr[16]=0), block B select (addr[16]=1),
  block A/B isolation, 0x0 write error, out-of-range error (bit >[16]), dual-
  port independence, and 2-cycle read latency. DUT instantiation verified
  against the merged `Data_Memory` port list (i_data_a/i_address_a[23:0]/
  i_wren_a/i_offset_a[9:0]/o_q_a + port b + four error flags).
- **`tests/run_regression.sh`.** Added the `TB_Step1_DataMemory` block
  (TPU_0x1_Buffer.v + Mem_Unit.v + Data_Memory.v + tb). Fixed the Step 8 source
  list: `MEMORY/WEIGHT_MEMORY/Weight_Memory.v` -> `MEMORY/MEM_UNIT/Mem_Unit.v`
  + `MEMORY/Data_Memory.v`. Both lists mirror the "How to run" headers.
- **`tests/TB_Step2_Feeder.v`.** `pm_address`/`PM_MAX` widened 10 -> 13 bits
  (`13'd3`); `pm_address == 13'd0` / `13'd1` comparisons updated for the 8192-
  word program memory.
- **`tests/TB_Step4_VectorProcessor.v`.** Replaced the separate `o_wm_*`/
  `o_buf_*`/`i_wm_q`/`i_buf_q` ports with the unified `o_dm_address[23:0]`/
  `o_dm_wren`/`o_dm_data`/`o_dm_offset[9:0]`/`i_dm_q`. Added a Data_Memory-style
  read model (flat main memory by address, 0x1 buffer by offset, 2-cycle
  registered read latency) and widened source/dest addresses to 24 bits. Test
  data moved to flat addresses (ISA 0x2 -> main_mem[2], etc.); 0x1 tests drive
  the offset port. All 14 cases preserved.
- **`tests/TB_Step1_Programmer.v`.** DUT ports -> `o_dm_data_a`/
  `o_dm_address_a[23:0]`/`o_dm_wren_a`/`o_dm_offset_a[9:0]`/`i_dm_q_a`. MEM
  address sends are now 3-byte little-endian (LADD/MADD/UADD). Flat unified
  addressing (dropped the ISA-2 weight mapping): main-memory writes land at the
  flat address; 0x1 writes hold address 0x1 and walk the offset. Added a
  Data_Memory shadow (main + 0x1 buffer) and reworked the write-capture into a
  main-memory log (by flat address) and a 0x1-buffer log (by offset). 12 cases
  preserved; `pm_address` widened to 13 bits.
- **`tests/TB_Step8_FullSystem.v`.** Shadow now snoops the unified port
  (`dut.dm_wren_a`/`dm_address_a`/`dm_offset_a`/`dm_data_a`) into a main-memory
  shadow (flat) + 0x1-buffer shadow; `send_mem4` sends the 3-byte address; all
  MEM/INPUT sends carry MADD. Flat addressing: mult results read at flat
  addresses (rd 0x0A -> mem[10..13], rd 0x0E -> mem[14..17]) instead of the old
  WM-phys 8..11/12..15. **Instruction builders `mk_relu`/`mk_mul_rq`/`mk_mul`
  rewritten to the ISA v0.4 128-bit layout** (24-bit rs1[123:100], MUL
  rs2[91:68]/rd[59:36]/M0[35:28]/n[27:20], ACT rd[99:76]/len[75:60]) — required
  for the Controller decode, otherwise Tests 9-12 decode garbage. Compile
  source list header updated (Weight_Memory -> Mem_Unit + Data_Memory).

- **Verification: PENDING.** Deferred to the Questa PC per the machine gate.
  Expected once `./tests/run_regression.sh` runs there: Step1 (TB_Step1_Programmer
  12/12 and TB_Step1_DataMemory 10/10), Step2 (9/9), Step3 (10/10), Step4 (14/14),
  Step8 (12/12), plus the unchanged Step5/6/7 for regressions. A pass = every
  testbench prints its `Results:` line with 0 FAIL and no `Test N: FAIL`. v0.4
  `main.md` steps left UNMARKED until that run confirms.
- **Out-of-scope finding (surfaced): `tests/TB_Step3_Controller.v`.** It still
  builds instructions with the pre-v0.4 16-bit ISA field layout while the merged
  `Controller.v` decodes the v0.4 24-bit-field layout, so Step 3 would FAIL on
  the Questa PC. Not in this task's update list; left for user direction.
  **RESOLVED** in the follow-up entry above (coordinator-approved) — builders and
  address widths updated to the v0.4 layout.

## 2026-07-09 — v0.4 Memory Expansion: RTL (Steps 1-5) + address-width threading

- **Context:** v0.4 grows data memory to 2^17 = 131072 words behind a new
  `Data_Memory` wrapper, unifies the data address space to a flat 24-bit dual
  port (a/b), grows program memory to 8192 words, widens ISA address fields
  16 -> 24 bits, and adds the 3-byte MEM address (Message Protocol v0.3). RTL
  for all five build steps is implemented; testbench work is BLOCKED (see the
  "Verification / blocker" note below).

- **`TPU/MEMORY/Data_Memory.v`** (Step 1) — implemented the user-provided
  skeleton. Instantiates `TPU_0x1_Buffer` (buff) and two `Mem_Unit` blocks
  (mem_a, mem_b) by name. Per-port 24-bit decode: `0x0` -> hardwired-zero read
  and `o_write_error_*` on a write; `0x1` -> buffer addressed by `i_offset_*`;
  `0x2..0x1FFFF` -> Mem_Unit with `address[16]` selecting block B(1)/A(0) and
  `address[15:0]` the within-block offset; any bit above [16] -> `o_range_error_*`
  and a 0 read. Write enables are gated by the region decode (writes to `0x0`/
  out-of-range never reach a RAM). The output data mux selects among
  {zero, buffer, block A, block B} using the region decode delayed by two clock
  cycles so it aligns with the RAMs' 2-cycle registered read latency — total
  latency stays 2 cycles.
- **`TPU/PROCESSING/Vector_Processor.v`** (Step 3) — widened
  `i_mem_source_address`/`i_mem_dest_address` and all internal address registers
  to 24 bits. Replaced the separate weight-memory + 0x1-buffer port groups
  (`o_wm_*`/`o_buf_*`/`i_wm_q`/`i_buf_q`, and the `-2` weight offset / `mem_rd_data`
  mux) with one flat Data_Memory port (`o_dm_address[23:0]`, `o_dm_wren`,
  `o_dm_data`, `o_dm_offset[9:0]`, `i_dm_q`). The VP now issues flat ISA
  addresses; only the reserved 0x0 (address held at 0 so the wrapper returns 0
  for every element) and the isolated 0x1 buffer (address held at 0x1, element
  index carried on `o_dm_offset`) are special-cased. Flatten<->matrix ordering,
  length/dim walking, and the dyadic requantization datapath are unchanged.
- **`TPU/CONTROL/Controller.v`** (address-width threading, per scope note) —
  updated all instruction field aliases to the ISA v0.4 layout and widened the
  address outputs to 24 bits: rs1 [123:100], MUL rs2 [91:68] / rd [59:36] /
  M0 [35:28] / n [27:20], ACT rd [99:76] / len [75:60], ELEM rs2 [99:76] /
  sz1 [75:68] / sz2 [67:60] / rd [59:36]. No control-flow / requant changes.
- **`TPU/PROGRAMMER/Programmer.v`** (Step 4) — widened `mem_addr` to 24 bits;
  added the 3-byte little-endian MEM address decode (LADD/MADD/UADD via new
  `M_MADD_*` states). Replaced the `o_wm_*`/`o_buf_*` ports with the unified
  Data_Memory port a (`o_dm_data_a`, `o_dm_address_a[23:0]`, `o_dm_wren_a`,
  `o_dm_offset_a[9:0]`, `i_dm_q_a`). MEM writes to the 0x1 buffer hold the flat
  address at 0x1 and walk the offset; main-memory writes issue the flat address
  directly (no `-2`, the wrapper owns block decode). Device/external I/O still
  staged at 0x1. Widened the program-memory write path for the deeper memory:
  `o_pm_address` -> [12:0], `pc` -> 14 bits, overflow at `PM_DEPTH = 8192`.
- **`TPU/PROGRAMMER/Feeder.v`** (Step 5) — `pc` and `o_pm_address` widened to
  13 bits; `PM_MAX_ADDRESS = 13'd8191`; FEED_ERROR/overflow now keyed to the
  8192-word program memory. Confirmed the `Program_Memory` IP is 8192 words
  (13-bit address).
- **`TPU/TPU.v`** (Step 2) — replaced the direct `Weight_Memory` +
  `TPU_0x1_Buffer` instantiations with a single `Data_Memory` instance. The
  port-a memory MUX now routes Programmer (program LOW) / VP_A (program HIGH)
  data/address/wren/offset into the wrapper; VP_B drives port b. Widened all
  data-memory address nets to 24 bits, program-memory nets to 13 bits, and the
  Controller<->VP address nets to 24 bits. The user-owned DEBUG section is
  unchanged; because the weight memory and 0x1 buffer are now unified, the
  removed nets it references (`wm_q_a/b`, `buf_q_a/b`) are preserved as
  code-section aliases of `dm_q_a/b` so the debug section still elaborates.

- **Verification / blocker (Verification: PENDING).** This is not the Questa PC
  (Linux; no `vsim.exe`), so simulation is deferred per the machine gate.
  Additionally, the section `tests/` directory is currently **inaccessible to
  the agent**: the committed `.claude/settings.json` deny glob `Read/Edit/Write
  (Tests/**)` — intended for the repo-root `Tests/` scratchpad — matches
  `System/Tensor_Processing_Unit/tests/` case-insensitively, so no testbench or
  `run_regression.sh` change could be written in place. The RTL above breaks the
  existing testbenches (old interfaces) until they are updated. Ready-to-drop-in
  artifacts were produced in the session scratchpad:
  `tests_v0.4/TB_Step1_DataMemory.v` (new, 10 decode/latency cases) and
  `tests_v0.4/run_regression.sh` (adds a `TB_Step1_DataMemory` block; fixes the
  Step 8 source list: `WEIGHT_MEMORY/Weight_Memory.v` -> `MEM_UNIT/Mem_Unit.v`
  + `MEMORY/Data_Memory.v`). Still to update once `tests/` is writable:
  `TB_Step2_Feeder.v` (pc/pm_address -> 13 bits, `PM_MAX` -> `13'd3`),
  `TB_Step4_VectorProcessor.v` (DUT ports -> `o_dm_*`/`i_dm_q`, 24-bit
  addresses, drive `o_dm_offset` for 0x1), `TB_Step1_Programmer.v` (3-byte MEM
  address, `o_dm_*_a`/`i_dm_q_a` ports, flat addresses instead of ISA-2),
  `TB_Step8_FullSystem.v` (unified flat addressing — physical addr == ISA addr,
  3-byte MEM sends, Data_Memory shadow instead of weight-memory shadow).
  Expected once run on the Questa PC: Step1 (Programmer + DataMemory), Step2,
  Step3, Step4, Step7, Step8 pass; v0.4 `main.md` steps left UNMARKED until then.

## 2026-07-07 — Regression runner + machine-gated test workflow

- **Added `tests/run_regression.sh`** — one-command Questa regression driver.
  Compiles each testbench into its own fresh work library (behavioral stubs
  collide with real RTL otherwise), simulates batch-mode
  (`vsim -c -L altera_mf_ver -voptargs=+acc ... -do "run -all; quit -f"`), and
  prints a per-testbench pass/fail summary. Runs all benches by default or a
  subset by step number (`./tests/run_regression.sh 3 4 8`). Sim scratch is
  written off the OneDrive-synced Desktop (`$LOCALAPPDATA/tpu_sim`, override via
  `SIM_WORK`) to avoid the transient vopt file-lock seen there.
  - Failure detection keys off the per-test `Test N: FAIL` line (printed by every
    testbench) rather than the summary wording, since Step2/Step4 report
    `X / Y tests passed` while the others report `X PASS, Y FAIL`.
  - Each testbench is a single `run_tb <step> <top> <sources...>` block; the
    script header documents how to add/edit a block when testbenches change.
- **Machine gate:** the script exits with a deferral notice (no simulator run) if
  `vsim.exe` is absent at `C:\intelFPGA_lite\23.1std\questa_fse\win64\`. This is
  the only project machine with Questa (`COMPUTERNAME = CEC-EGB267-05`).
- **`CLAUDE.md`** — added a "Test Execution Environment" section: Questa install
  location + `altera_mf_ver` invocation, how to run the regression, how to update
  the runner for new/changed testbenches, and the run-here-defer-elsewhere rule
  (off the Questa PC, still write testbenches but mark `Verification: PENDING`).
- **`.claude/settings.json`** (tracked, travels across machines) — added
  `permissions.allow` prefix rules so testbench runs are auto-approved:
  `./tests/run_regression.sh` and the Questa binaries (`vlib`/`vlog`/`vsim`/
  `vmap`). Safe to check in because the runner self-gates off the Questa PC.
  Reading results uses the Read/Grep tools (no Bash permission needed). The
  brittle per-command rules in the gitignored `settings.local.json` are now
  redundant but left untouched.
- **Verification:** ran `./tests/run_regression.sh` on the Questa PC — full suite
  ALL PASS (Step1 19/19, Step2 9/9, Step3 10/10, Step4 14/14, Step5 7/7,
  Step6 7/7, Step7 9/9, Step8 12/12), plus a subset smoke test (`5 6`). No RTL or
  testbench source changed; tooling/workflow only.

## 2026-07-07 — v0.3 Step 3 verification: full Questa regression PASS

- **Ran the pending v0.3 regression** in Questa Intel FPGA Edition 2023.3
  (`intelFPGA_lite/23.1std/questa_fse`) on the Windows dev machine — the step the
  prior (2026-07-06) session could not run. Each testbench compiled into its own
  fresh work library (behavioral stubs collide with real RTL otherwise) and
  simulated batch-mode with `-c -L altera_mf_ver -voptargs=+acc ... -do "run -all;
  quit -f"`. `altera_mf_ver` resolved from the global `modelsim.ini` mapping.
- **Results — all PASS, no regressions:** Step1 19/19, Step2 9/9, **Step3 10/10**,
  **Step4 14/14**, Step5 7/7, Step6 7/7, Step7 9/9, **Step8 12/12**. Matches the
  expected counts recorded in the 2026-07-06 entry. No compile errors, elaboration
  errors, or fatals across the suite.
  - `TB_Step3_Controller` **Test 10**: M0/n decoded from the MUL instruction and
    asserted to VP_A during the SA-output writeback decode, held 0 during the
    SA-load execute decode.
  - `TB_Step4_VectorProcessor` **Test 13** (round-to-nearest M0=1,n=2: 62→16 vs.
    truncation's 15) and **Test 14** (M0 scale multiply M0=25,n=3: 10→31) confirm
    the runtime dyadic requant datapath `clamp((M0*x + (1<<(n-1))) >> n)`.
  - `TB_Step8_FullSystem` **Test 12**: identity × [[100,50],[25,10]] with M0=3,n=4
    requantizes to [[19,9],[5,2]] end to end through SPI → weight flash → systolic
    array → runtime M0/n requant → memory.
- **`main.md`:** v0.3 Step 3 marked ✅ Complete. v0.3 build done.

## 2026-07-06 — v0.3: Per-layer dyadic requantization (Vector_Processor + Controller)

- **Context:** v0.3 replaces the uniform, compile-time `REQUANT_SHIFT`
  requantization with per-layer dyadic requantization. Each MUL instruction now
  carries an 8-bit multiplier `M0` (bits 59-52) and an 8-bit right-shift `n`
  (bits 51-44); the Vector_Processor applies
  `result = clamp((M0 * x + (1 << (n - 1))) >> n)` to systolic array outputs
  before writeback. RTL-only change confined to the Vector_Processor (requant
  datapath) and Controller (decode + routing) per the main.md scope note.
- **`TPU/PROCESSING/Vector_Processor.v`** (Step 1):
  - Added `i_scale` (M0, 8-bit unsigned) and `i_shift` (n, 8-bit unsigned)
    inputs; latched into `scale_lat`/`shift_lat` in IDLE alongside the other
    control signals (reset to 0).
  - Removed the `REQUANT_SHIFT` parameter and its fixed `x >>> REQUANT_SHIFT`
    requant. New datapath: `sa_scaled = $signed({1'b0,scale_lat}) *
    $signed(i_sa_c)` (unsigned scale zero-extended to a 9-bit positive signed
    value so the 41-bit product stays signed and cannot overflow the 32-bit
    accumulator × 8-bit scale), `sa_round_bias = 1 << (shift-1)` added **before**
    the arithmetic right shift `>>> shift` (round-to-nearest), then saturated to
    signed 8-bit [-128,127]. `shift == 0` is guarded so `(shift-1)` never
    underflows the 8-bit width.
- **`TPU/CONTROL/Controller.v`** (Step 2):
  - Decoded `mul_scale = instr_latch[59:52]` and `mul_shift = instr_latch[51:44]`.
  - Added `o_scale_a`/`o_shift_a` outputs (VP_A only — only VP_A reads SA outputs
    and requantizes). Driven with `mul_scale`/`mul_shift` in the MUL **writeback**
    decode branch; default 0 otherwise (don't-care for non-MUL / execute phase).
- **`TPU/TPU.v`:** added `ctrl_scale_a`/`ctrl_shift_a` wires; connected Controller
  `o_scale_a`/`o_shift_a` → VP_A `i_scale`/`i_shift`. VP_B `i_scale`/`i_shift`
  tied to 0 (VP_B never requantizes). Debug section untouched.
- **Tests:**
  - `tests/TB_Step4_VectorProcessor.v` (12 → 14): drives `i_scale`/`i_shift` at
    runtime (removed the obsolete `defparam REQUANT_SHIFT`). Tests 6/12 pinned to
    M0=1,n=8 (reproduce the prior >>8 expected values with rounding). Added
    Test 13 (round-to-nearest: M0=1,n=2, 62→16 vs. truncation's 15) and Test 14
    (M0 scale multiply: M0=25,n=3, 10→31). Generalized the SA stub with a
    `force_mode`/`sa_c_force` to drive arbitrary MAC(0,0) values.
  - `tests/TB_Step3_Controller.v` (9 → 10): added `make_mult_rq` (M0/n operands);
    Test 10 verifies M0/n are decoded and asserted to VP_A during the writeback
    decode (`vect_source_a == VSRC_SA_OUT`) and held at 0 during the SA-load
    execute decode.
  - `tests/TB_Step8_FullSystem.v` (11 → 12): parameterized `mk_mul` with M0/n
    (default M0=1,n=8 keeps Tests 9-11 expectations; added `mk_mul_rq`); removed
    the `defparam REQUANT_SHIFT` lines. Added Test 12: identity × [[100,50],
    [25,10]] with M0=3,n=4 requantizes to [[19,9],[5,2]] via
    `clamp((3x+8)>>4)`, exercising the runtime M0/n datapath end to end through
    SPI → weight flash → systolic array → requant → memory.
- **Verification:** **PENDING.** This session's environment has no Questa/vsim
  (the systolic array needs Altera `altera_mf`/scfifo), so the regression could
  not be run here. Must be run in Questa on the Windows dev machine: expect
  Step3 10/10, Step4 14/14, Step8 12/12, plus the full suite for regressions.
  `main.md` v0.3 steps left unmarked until that run confirms.

## 2026-07-02 — v0.2.1 verification: full Questa regression PASS

- **Ran the pending v0.2.1 regression** in Questa Intel FPGA Edition 2023.3
  (`intelFPGA_lite/23.1std/questa_fse`) on the Windows dev machine — the step
  the prior session could not run. Each testbench compiled into its own fresh
  work library (behavioral stubs collide with real RTL otherwise) and simulated
  batch-mode with `-L altera_mf_ver -voptargs=+acc ... -do "run -all; quit -f"`.
- **Results — all PASS, no regressions:** Step1 19/19, Step2 9/9, Step3 9/9,
  Step4 12/12, Step5 7/7, Step6 7/7, **Step7 9/9, Step8 11/11**.
  - `TB_Step7_SystolicArray` **Test 9** (negative operand 0xFE = −2 × 3 through
    MAC(0,0)) reads back a negative multiple of 6 — passes only with the signed
    multiply; confirms the fix.
  - `TB_Step8_FullSystem` **Test 11** (mixed-sign weights rs2 =
    [[−16,32],[48,−64]] × identity·16) requantizes to **[−1,2,3,−4]** end to end
    through SPI → weight flash → systolic array → requant → memory.
- **Requantization-impact note (0x7F saturation):** the signed multiply removes
  the positive bias that unsigned MACs injected — a −1 operand (0xFF) no longer
  accumulates as +255. Dot products containing negatives now carry their true
  sign into requant instead of piling up large positives that clamped to 0x7F,
  so the constant-0x7F saturation is expected to be resolved for any program with
  negative weights/activations. Test 11 demonstrates negative requant outputs
  (−1, −4) surviving end to end, which the old RTL could not produce.
- **`main.md`:** v0.2.1 Step 1 and Step 2 marked ✅ Complete. v0.2.1 build done.

## 2026-07-02 — v0.2.1: Systolic array signed-multiply fix

- **Context:** The Multiply_Accumulate_Unit performed an *unsigned* multiply
  while the ISA Memory Datatype is 8-bit *signed*. A negative operand (e.g. a
  weight of −1 = 0xFF) accumulated as a large positive value (255), corrupting
  any dot product containing negatives and biasing results strongly positive —
  a suspected contributor to the constant-0x7F requantization saturation.
- **`TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Unit.v`** (Step 1):
  - Changed the accumulation to `o_c <= o_c + ($signed(i_a) * $signed(i_b))` so
    the operands multiply as signed two's-complement values.
  - Declared the accumulator `output reg signed [31:0] o_c`. This is required,
    not cosmetic: with an *unsigned* `o_c` the mixed-sign addition would coerce
    the signed product to unsigned and **zero-extend** it into the 32-bit sum,
    re-introducing the bug. A signed accumulator keeps the expression fully
    signed so the product is sign-extended. Bit pattern at the port is
    unchanged, so the unsigned `unit_out` net in Systolic_Array and the
    `$signed(i_sa_c)` requant in Vector_Processor are unaffected.
  - No specification change required — signed operands were already implied by
    the Memory Datatype; only the RTL was out of conformance.
- **Tests (Step 1 + Step 2):**
  - `tests/TB_Step7_SystolicArray.v` — added **Test 9**: streams a negative
    operand (top_buf[0] = −2 = 0xFE, left_buf[0] = 3) through MAC(0,0); asserts
    the accumulator reads back **negative** and a multiple of 6. The old
    unsigned RTL would read 0xFE as +254 and accumulate a large positive, so
    this test passes only with the signed multiply. (8 → 9 tests.)
  - `tests/TB_Step8_FullSystem.v` — added **Test 11**: a full-system mult with
    mixed-sign weights rs2 = [[−16,32],[48,−64]] against rs1 = identity*16,
    expecting requantized result [[−1,2],[3,−4]]. Exercises signed operands end
    to end through SPI → weight flash → systolic array → requant → memory.
    (10 → 11 tests.)
- **Verification:** **PENDING.** This session's environment has no Questa/vsim
  (and the array needs Altera `altera_mf`/scfifo), so the regression could not
  be run here. Must be run in Questa on the Windows dev machine:
  `TB_Step7_SystolicArray` (expect 9/9) and `TB_Step8_FullSystem` (expect 11/11),
  plus the full suite for regressions. `main.md` v0.2.1 steps left unmarked until
  that run confirms.

## 2026-07-01 — Display Logic Update part 3

- Pulled over 7-segment display code from another project — located in
`TPU/VGA_DEBUG/`.
- instantiated `three_decimal_vals_w_neg` module to display *output_val* in
device mode.
  - Output to 7-segment displays is disabled in external mode to give an
  additional visual indicator of mode.
- modified `Tensor_Processing_Unit.sdc` to properly constrain VGA clock and
SPI clock.

## 2026-07-01 — Display Logic Update part 2

- **Updated `TPU/VGA_DEBUG/debug.v`**
- Added *i_clear* input signal — clears ASCII buffer when asserted HIGH.
- Added CLEAR state to FSM to implement clear behavior (transitions to 
WAIT_WREN when complete). 
- *write_ready* held LOW, *col_num* and *row_num* held at 0 in CLEAR state.
- Modified *ascii_wren* behavior — now held LOW in WAIT_WREN state and 
HIGH in CLEAR state.
- Implemented a *clear_count* variable to loop through all ASCII buffer
addresses while in the CLEAR state.
- Updated combinational ASCII address decoding to set *ascii_addr* to 
*clear_count* while the current state or next state is CLEAR.
- Updated combinational ASCII data decoding to force *ascii_data* to 0 when
in the CLEAR or WAIT_WREN states.
- **Added clear behavior to `Tensor_Processing_Unit.v`**
- Wired *end_reached* from `TPU/TPU.v` to *i_clear* in `TPU/VGA_DEBUG/debug.v`
so that the screen is cleared on program end.

## 2026-07-01 — Hardware Spec minor update (device_ready level, EXTERNAL_OUTPUT order, end_reached exposed)

- **Context:** Hardware Spec changelog 2026-07-01 made three changes affecting
  the Programmer, Feeder, and TPU modules. Wrapper `Tensor_Processing_Unit.v`
  intentionally not modified (per instruction; user manages it).
- **`TPU/PROGRAMMER/Programmer.v`** — EXTERNAL_OUTPUT reordered to match the spec:
  for each of the 10 values it now **forwards the datum, waits for
  `i_device_ready` HIGH, then pulses `output_data_valid`** (previously it pulsed
  valid before waiting). State flow changed to `X_OUT_FWD → X_OUT_WAIT_RDY →
  X_OUT_VALID → X_OUT_INC`; the valid strobe moved from `X_OUT_FWD` to
  `X_OUT_VALID`. Updated the `i_device_ready` comment to reflect that it is a
  level (HIGH while the consumer is ready), not a one-cycle pulse. The existing
  `X_OUT_WAIT_RDY` logic already treated it as a level, so no other change.
- **`TPU/TPU.v`** — exposed the Feeder's `end_reached` at the TPU boundary.
  Corrected the port declaration `O_end_reached` → `o_end_reached` so the port,
  the internal `assign o_end_reached = feeder_end_reached`, and the wrapper's
  `.o_end_reached(...)` connection all agree (and conform to the `o_` naming
  convention). Updated the `i_device_ready` passthrough comment.
- **`TPU/PROGRAMMER/Feeder.v`** — no change required; it already outputs
  `o_end_reached` (one-cycle strobe), which the spec change only re-exposes at
  the TPU level.
- **Tests:**
  - `tests/TB_Step8_FullSystem.v` — connect the new `o_end_reached` port
    (unused); the EXTERNAL_OUTPUT tests still pass under the reordered flow.
    Pinned `REQUANT_SHIFT = 8` via `defparam` on both VPs so the mult compute
    tests (9/10) verify the datapath deterministically regardless of the file's
    requant default (now 11, tuned for the NN under bring-up).
  - `tests/TB_Step4_VectorProcessor.v` — pinned `REQUANT_SHIFT = 8` via
    `defparam` for the same reason (Tests 6/12 check SA-output requant and
    saturation, which are scale-dependent).
- **Verification (Questa 2023.3):** full regression with the file default
  `REQUANT_SHIFT = 11` — Step1 19/19, Step2 9/9, Step3 9/9, Step4 12/12,
  Step5 7/7, Step6 7/7, Step7 8/8, Step8 10/10. All PASS.

## 2026-07-01 — Display Logic Update

- **Updated display logic in `Tensor_Processing_Unit.v`.**
- Moved debug logic to main code block — display output is an intended feature 
now.
- Renamed variables for clarity.
- Added proper MUX logic for programmer *mode_select*.
- **In TPU module:** exposed *end_reached* from Feeder module as an output.

## 2026-06-30 — Full-system mult tests (HW debug: constant 0x7F output)

- **Context:** on hardware, a device-mode neural-network program (mult → add →
  mult → add) outputs 0x7F (int8 positive saturation) regardless of input, while
  a single-`end` program correctly passes the input through. Added two
  full-system tests to localize the fault between the candidate causes.
- **`tests/TB_Step8_FullSystem.v`** (8 → 10 cases): added a weight-memory shadow
  (snoops the muxed weight-memory port-a writes) plus `mk_mul` / `send_mem4`
  helpers, and:
  - **Test 9 (coverage gap):** FLASH weights via MEM, then a single mult against
    them (rs1 [[16,0],[0,16]] × rs2 [[16,32],[48,64]] → requantized [[1,2],[3,4]]).
    Exercises the weight-flash + systolic-array mult path end to end — a path the
    v0.2 full-system test had stopped covering (the rewrite used a relu-over-
    0x1-buffer program with no weight matrix).
  - **Test 10 (consecutive mults):** two mult instructions with identical
    operands in one program (mult1 → WM 8..11, mult2 → WM 12..15). Both must
    yield [[1,2],[3,4]]; if the MAC accumulators were not cleared between mults,
    mult2 would double to [[2,4],[6,8]] or saturate.
- **Result:** both tests PASS (Test 9 = [1,2,3,4]; Test 10 mult2 = [1,2,3,4]).
  Full-system suite now 10/10. This exonerates the two RTL hypotheses in
  simulation: the weight-flash+mult path is correct, and the MACs *are* cleared
  between consecutive mults. The remaining likely cause is the requantization
  scale (fixed `REQUANT_SHIFT = 8`) vs. how the network weights were quantized —
  being investigated separately.
- **Note:** ran in a local scratch work library; the Desktop `work/` dir hit a
  transient file lock (OneDrive/Defender) during `vopt`. Testbench source
  unaffected; no RTL changed.

## 2026-06-30 — Testbench coverage expansion (all 8 testbenches)

- **Context:** lifted the prior 7-test-per-bench convention to add coverage of
  behaviors that were previously untested. No RTL changed; tests only.
- **`tests/TB_Step1_Programmer.v`** (7 → 12 cases / 19 sub-checks): added
  COM_ERROR on an invalid command code; WRITE_ERROR on a MEM command to ISA
  address 0x0000; EXTERNAL_INPUT unified routing to *weight* memory (not just the
  0x1 buffer); PROGRAM command inside an INPUT transmission → COM_ERROR (the
  EXTERNAL_INPUT clarification); and IDLE discarding stray non-header bytes
  before a valid FLASH. Error tests reset the DUT (error states are terminal).
- **`tests/TB_Step2_Feeder.v`** (7 → 9): re-added the FEED_ERROR exhaustion test
  (previously displaced by the end_reached test) and added a check that
  end_reached never pulses while forwarding non-terminating instructions.
- **`tests/TB_Step3_Controller.v`** (7 → 9): systolic_array_start is asserted for
  mult but not for relu/add; function-code decode (relu → activator RELU, add →
  ALU ADD, mult → both NOOP).
- **`tests/TB_Step4_VectorProcessor.v`** (10 → 12): VSRC_MEM → VDST_ALU_A
  element_valid streaming; VSRC_SA_OUT negative requantization saturating to
  0x80 (added a `neg_mode` to the SA accumulator stub).
- **`tests/TB_Step5_Activator.v`, `tests/TB_Step6_ALU.v`** (6 → 7 each): invalid
  function/operation code drives the control-error debug value (29).
- **`tests/TB_Step7_SystolicArray.v`** (7 → 8): `clear` zeroes a MAC accumulator
  (residual data flushed between operations).
- **`tests/TB_Step8_FullSystem.v`** (7 → 8): external re-input without re-FLASH —
  a second INPUT transmission re-runs the resident program and emits 10 new relu
  outputs [1..10], exercising the steady-state operating loop.
- **Verification (Questa 2023.3):** full regression — Step1 19/19, Step2 9/9,
  Step3 9/9, Step4 12/12, Step5 7/7, Step6 7/7, Step7 8/8, Step8 8/8. All PASS.

## 2026-06-30 — v0.2 Step 5: Full-system test rewrite (both IO modes) + final regression

- **`tests/TB_Step8_FullSystem.v` rewritten** to drive the top-level `TPU`
  through both v0.2 Programmer IO modes end to end, using the real RAM IP for
  the readback path (no internal RAM array referenced):
  - **Device mode** (`mode_select`=0): FLASH a self-contained `relu 0x0001 ->
    0x0001, len 1` program; supply a single input via `i_input_data` /
    `i_input_data_valid`; verify `o_output_data` / `o_output_data_valid` emit the
    activated value. Tests pass-through (42 -> 42) and negative clamp (-9 -> 0).
  - **External mode** (`mode_select`=1): re-FLASH `relu 0x0001 -> 0x0001, len 10`;
    send an INPUT transmission (MEM @ISA 0x0001, len 10) with a mixed-sign
    vector; verify exactly 10 outputs are emitted with the `i_device_ready`
    handshake and that each equals `relu(input)`. A continuous-assign
    `device_ready = (prog.S == X_OUT_WAIT_RDY)` models the consumer handshake.
- **7 test cases:** device FLASH completion; device positive input; device
  negative input (clamp); external re-FLASH completion; external emits exactly 10
  values; external values equal relu(inputs); clean completion (Controller IDLE,
  Feeder DONE, program HIGH, no fault states). **All 7 PASS.**
- **Full regression (Questa 2023.3, `-L altera_mf_ver -voptargs=+acc`):**
  Step1 14/14, Step2 7/7, Step3 7/7, Step4 10/10, Step5 6/6, Step6 6/6,
  Step7 7/7, Step8 7/7 — **no regressions** from the v0.2 changes.
- **`main.md`:** marked all five v0.2 build steps ✅ Complete. v0.2 build done.

## 2026-06-30 — v0.2 Steps 1-4: Programmer IO rewrite, Feeder end_reached, TPU passthrough, NO OP write suppression

- **Clarifications recorded in `main.md`** (per user request before building): (1)
  `EXTERNAL_INPUT` uses unified address routing like FLASH (PROGRAM command
  invalid in an INPUT transmission); (2) new Programmer IO is exposed at the TPU
  module boundary only (FPGA wrapper pin mapping left to the user); (3) the
  revised full-system test exercises both device and external IO modes.
- **Build-step order:** built Programmer (Step 1) → Feeder `end_reached`
  (Step 3) → TPU (Step 2) → minor changes (Step 4). Feeder/TPU were reordered
  because the TPU rewiring consumes the Feeder's new `end_reached` output.

- **`TPU/PROGRAMMER/Programmer.v` — full meta-state rewrite (v0.2).**
  - New IO: `i_mode_select`, `i_input_data_valid`, `i_device_ready`,
    `i_input_data[7:0]`, `o_output_data[7:0]`, `o_output_data_valid`,
    `i_end_reached`, `i_buf_q_a[7:0]` (0x1-buffer readback), and `o_tpu_rst`
    (active LOW) now separate from `o_program`.
  - Single flat FSM organized into six meta-states: IDLE (scan SPI for a
    FLASH/INPUT header; service latched device input / program end by priority
    PROGRAM > DEVICE_INPUT > OUTPUT), PROGRAM (FLASH: overwrite program memory,
    MEM/PROGRAM commands), DEVICE_INPUT (write `i_input_data` to 0x1-buffer addr
    0, pulse `tpu_rst`), DEVICE_OUTPUT (emit 0x1-buffer[0]), EXTERNAL_INPUT
    (INPUT header, unified MEM routing, PROGRAM command → COM_ERROR),
    EXTERNAL_OUTPUT (emit 0x1-buffer[0..9] paced by `i_device_ready`).
  - Protocol v0.2 codes: FLASH `U`, INPUT `I`, MEM `M`, PROGRAM `P`, STOP `S`.
  - One-cycle pulses (`i_end_reached`, `i_input_data_valid`) are latched so they
    are not missed while the FSM is busy; consumed on entering the relevant
    meta-state. 0x1-buffer reads honor the 2-cycle RAM read latency.
  - Retains error states STATE_ERROR / COM_ERROR / WRITE_ERROR /
    PROG_OVERFLOW_ERROR (unrecoverable except by `i_rst`).
- **`TPU/PROGRAMMER/Feeder.v`** — added `o_end_reached` output, pulsed HIGH for
  one clock cycle in CHECK when an `end` instruction is fetched (HIGH during the
  first DONE cycle). Added a default deassert so it is a clean one-cycle strobe.
- **`TPU/TPU.v`** — `trst` is now `i_rst & prog_tpu_rst` (was `i_rst & program`),
  letting the Programmer reset the internals independently of its memory-bus
  access so output meta-states can read memory without disturbing a finished
  program. Added the Programmer IO passthrough ports at the TPU boundary; wired
  the Feeder `o_end_reached → Programmer i_end_reached`, `buf_q_a → Programmer
  i_buf_q_a`, and `Programmer o_tpu_rst → prog_tpu_rst`. Debug section untouched.
- **`TPU/PROCESSING/Activator.v`, `TPU/PROCESSING/ALU.v`** — `o_write` now
  suppressed when the function/operation is NO OP
  (`o_write = i_enable & (op != NOOP)`), per the v0.2 hardware-spec design fix,
  so a non-activation/non-ALU instruction cannot push placeholder data into the
  vector buffer. Verified the Systolic_Array already passes `i_trst` to every MAC
  (the other listed design fix) — no change needed.

- **Tests:**
  - `tests/TB_Step1_Programmer.v` — rewritten: reset defaults; FLASH+MEM with
    program/tpu_rst toggling; PROGRAM instruction assembly + program-memory
    overwrite; DEVICE_INPUT; DEVICE_OUTPUT; EXTERNAL_OUTPUT (10 values with
    device_ready handshake); INPUT-header routing (external writes, device-mode
    discards). 7 cases / 14 sub-checks — **all PASS**.
  - `tests/TB_Step2_Feeder.v` — Test 7 replaced with an `end_reached` one-cycle
    strobe check (count == 1 and Feeder in DONE). **7/7 PASS**.
  - `tests/TB_Step5_Activator.v`, `tests/TB_Step6_ALU.v` — Test 4/Test 2 updated
    to drive a non-NOOP function when checking `o_write` follows `i_enable`;
    NOOP test now also asserts `o_write` is suppressed. **6/6 PASS each**.
- **Verification:** full TPU hierarchy recompiles with **0 errors / 0 warnings**
  under Questa 2023.3. Full-system test rewrite (Step 5) pending.

## 2026-06-25 — Feeder Program_Memory 2-cycle read latency fix

- **Context:** `main.md` Build Parameters now specify a 2-cycle memory read
  latency for all device RAMs. Audited every module that reads a device RAM:
  - `Vector_Processor.v` — already correct (`MEM_WAIT`+`MEM_WAIT2`,
    `VB_WAIT`+`VB_WAIT2` give two wait cycles; fixed 2026-06-22).
  - `Programmer.v` — reads only the SPI input FIFO (own `BUFFER_LATENCY`);
    it writes, never reads, the device RAMs. No change needed.
  - `Feeder.v` — read Program_Memory with only **one** wait cycle (`WAIT_PM`),
    sampling `i_pm_q` in `CHECK` one cycle too early under 2-cycle latency.
- **`TPU/PROGRAMMER/Feeder.v`** — added `WAIT_PM2` state between `WAIT_PM` and
  `CHECK` so the address now settles for two cycles before the RAM output is
  read (`FETCH → WAIT_PM → WAIT_PM2 → CHECK`). State register already 4 bits
  (WAIT_PM2 = 4'd10). Updated the FETCH comment to describe the 2-cycle path.
- **`tests/TB_Step2_Feeder.v`** — upgraded the behavioral Program_Memory model
  from a single registered stage to a two-stage pipeline (`pm_q_stage1` →
  `pm_q_reg`) so it faithfully models the IP's 2-cycle latency and genuinely
  exercises the new wait state. Updated header/inline comments. Existing test
  loop margins already accommodate the extra cycle.
- **Verification:** RUN 2026-06-25 on Questa Intel Starter FPGA Edition 2023.3.
  Full TPU hierarchy + both testbenches recompiled with 0 errors/0 warnings.
  `TB_Step2_Feeder` → **7/7 PASS** (confirms the new `WAIT_PM2` wait state reads
  Program_Memory correctly under the two-stage-pipeline model). `TB_Step8_FullSystem`
  (`-L altera_mf_ver -voptargs=+acc`) → **7/7 PASS** (mult, relu, add, end
  termination, add saturation, clean completion). No regression from the 2-cycle
  latency fix.

## 2026-06-22 — Terminology update for the *end* instruction (docs + identifiers)

- **Context:** ISA v0.2 / Hardware Spec v0.1.1 renamed the program-termination
  instruction from *syscall* (an implicit terminate) to *end*. *end* reuses the
  old *syscall* encoding (opcode `0000`, funct7 `0x0`); *syscall* moved to a new
  encoding (opcode `0000`, funct7 `0x1`). Because the Feeder terminates on opcode
  `0000`, no Verilog logic required modification — only comments/documentation.
- **`TPU/PROGRAMMER/Feeder.v`** — updated the module-header description; renamed
  the terminator parameter `SYSCALL_OPCODE` → `END_OPCODE` (and its use in the
  CHECK next-state logic). Comment explains the shared `0000` encoding. The
  detection value (`4'b0000`) is unchanged, so behavior is identical.
- **`tests/TB_Step2_Feeder.v`** — renamed `OPCODE_SYSCALL` → `OPCODE_END` and the
  `make_syscall` helper → `make_end` (all call sites updated); updated test-case
  descriptions and inline comments from "syscall"/"non-syscall" to "end"/
  "non-terminating"; corrected the ISA version reference to v0.2.
- **`tests/TB_Step8_FullSystem.v`** — updated header, Test 2 description/messages,
  and inline program comments to name the terminating instruction *end*.
- **`main.md`** — Current State Feeder bullet now reads "*end* termination opcode".
- **Verification:** recompiled the full TPU hierarchy + Step 8 testbench under
  Questa (`-L altera_mf_ver -voptargs=+acc`) with 0 errors; `TB_Step8_FullSystem`
  ran to completion with **7 PASS / 0 FAIL** (mult, relu, add, end termination,
  add saturation, clean completion). No behavioral change from the rename.
- Historical log entries below are left as-is as a dated record of prior work.

## 2026-06-22 — Build Step 8: Full-system integration test and bug fixes

- **Created `tests/TB_Step8_FullSystem.v`** — full-system integration test that
  drives the top-level `TPU` module over SPI exactly as real hardware would. A
  single SPI session (Functional TPU Message Protocol) programs test weights via
  MEM packets and a four-instruction program (mult, relu, add, syscall) via
  PROGRAM packets, then STOP releases the TPU (program HIGH → trst HIGH) and the
  Feeder/Controller run the program to completion. Results are verified by
  snooping the muxed weight-memory / 0x1-buffer write ports into shadow memories
  (mirrors exactly what the RAMs store; no vendor-IP internal arrays referenced).
- **Test program (unified address space; a MEM-packet address and an ISA
  instruction address refer to the same location, physical WM addr = ISA − 2):**
  - mult: rs1@0x2 `[[16,0],[0,16]]` × rs2@0x6 `[[16,32],[48,64]]` → rd@0xA;
    products 256/512/768/1024 requantized (>>8) to `[[1,2],[3,4]]`.
  - relu: src@0xE `[-5,7,-1,100]` → rd@0x12 `[0,7,0,100]`.
  - add: rs1@0x16 `[10,20,-100,50]` + rs2@0x1A `[5,-3,-50,80]` → rd@0x1E
    `[15,17,-128,127]` (verifies int8 saturation clamps).
- **7 tests:** programming-over-SPI completes and weights stored; syscall halts
  the Feeder; mult result; relu result; add result; add saturation; clean
  completion (Controller IDLE, Feeder DONE, no fault states). All 7 PASS.
- **Integration bugs found and fixed (each masked by a unit-test stub that
  modelled the wrong behaviour):**
  1. **`Programmer.v` weight-memory address offset** — the Programmer wrote the
     ISA address directly to Weight_Memory while the Vector_Processor reads/writes
     at `ISA − 2`; the two never agreed on a location. The Programmer was also
     internally inconsistent (it already subtracted the base for the 0x1 buffer).
     Fixed `M_WRITE` to use `o_wm_address_a <= mem_addr - 16'd2`, unifying the
     Programmer and processing-pipeline address spaces.
  2. **`Vector_Processor.v` MEM_ERROR over-trigger** — `MEM_ADDR` treated dest
     address 0 as illegal for *every* operation, but the Controller leaves the
     dest address 0 for systolic-array / activator / ALU destinations (where it
     is unused), so mult/relu/add deadlocked. Gated the check to fire only when
     the destination is device memory (`vect_dest_lat == VDST_MEM && dst_is_zero`).
  3. **`Vector_Processor.v` RAM/FIFO read-latency** — the VP allowed one wait
     cycle for the Weight_Memory/0x1-buffer read (`MEM_WAIT`) and one for the
     Vector_Buffer read (`VB_WAIT`), but both IP blocks have a two-cycle
     registered read latency (registered address + registered output / output
     register). This shifted every streamed element by one (relu/add corrupted,
     SA inputs corrupted). Added `MEM_WAIT2` and `VB_WAIT2` states.
  4. **`Feeder.v` controller handshake race** — `WAIT_CTRL` sampled
     `controller_idle` on the same cycle `controller_start` was asserted, while
     idle was still HIGH (the Controller had not yet reacted), so the Feeder
     advanced prematurely and lost instructions issued while the Controller was
     busy (the add never executed). Added a `WAIT_ACK` state that waits for the
     Controller to drop idle (acknowledge) before `WAIT_CTRL` waits for it to
     rise (done).
  5. **`Systolic_Array.v` post-drain over-accumulation** — the MACs accumulate
     `a*b` every cycle unconditionally, but after the input buffers drained the
     scfifo output registers held their last values, so MACs kept accumulating
     stale products before the VP read them (mult row 1 inflated 8×). Added a
     one-cycle-delayed read-request valid (`top_rdreq_d`/`left_rdreq_d`, matching
     the buffer's 1-cycle read latency) that gates each array edge: genuine
     popped data flows in while a buffer streams; 0 is fed otherwise. Zeros
     propagate through the array, so late MACs still capture their real product
     and drained buffers contribute nothing.
- **Updated `tests/TB_Step1_Programmer.v`** — Test 3 now expects the corrected
  unified mapping (MEM ISA address 0x5/0x6/0x7 → WM physical 3/4/5).
- **Verification:** full hierarchy compiles with 0 errors/0 warnings under
  Questa. All eight testbenches pass with the fixes applied and no regressions:
  Step1 14/14, Step2 7/7, Step3 7/7, Step4 10/10, Step5 6/6, Step6 6/6,
  Step7 7/7, Step8 7/7.

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

- Began building `TPU/SYSTOLIC_ARRAY/Systolic_Array.v` — top level module for
  the systolic array subsystem.
- Generates a systolic array from  
  `TPU/SYSTOLIC_ARRAY/Multiply_Accumulate_Buffer.v` modules by instantiating
  a grid of instances connected by wire nets.
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
  sa_col=row_cnt) to reconstruct result\[row_cnt\]\[col_cnt\] row-major, then
  requantizes (arithmetic right-shift bytes REQUANT_SHIFT bits, saturate to
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
