# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-10)

- **v0.4 shipped.** Tagged `v0.4.0` and `v0.4.1` on 2026-07-09; both merged to
  `main` (the `AI-workflows-test` branch is retired). v0.4 = memory expansion:
  unified `Data_Memory` wrapper, 24-bit flat address space, 3-byte MEM
  addressing (Message Protocol v0.3), 13-bit PC / 8192-word program memory.
  Both TPU and Assembler `main.md` build plans are fully complete.
- **Verified:** full Questa regression PASS on the Questa PC, then confirmed on
  real FPGA (0b46dab). Assembler v0.4 = 41 tests pass.
- **v0.4.1** added `System/run.sh` (synth-flow automation) and an `Assemble.py`
  helper to make running easier.
- **`/fan-out` now proven end-to-end.** The v0.4 RTL and assembler work ran in
  isolated per-agent worktrees and merged per-section into `main`
  (`Merge TPU v0.4…`, `Merge assembler v0.4`) — resolves the earlier
  "execute step untested" gap.

## Next Focus — v0.5 "Partitioning Update" (redo)

Run matmuls larger than the 8x8 array by tiling. A first implementation
(`multip` + single-row M-tiling + a partition MLIR pass) was built and merged on
a `v0.5-workspace` branch, then **discarded 2026-07-10** and the repo reverted to
`main`. Reason: tiling makes each operand/output a strided sub-block of a
row-major tensor, but `mult` only addresses contiguous regions; single-row
M-tiling still left the weight (rhs) tile strided whenever K and N are both tiled
(wrong results) and forced ~1/8 array utilization.

- **Specs (Phase 0) — COMPLETE 2026-07-13:** ISA + HW spec (both v0.5.0,
  proofread clean) cover the full v0.5 feature set — `multip` (opcode 1000 /
  funct3 0x1) with funct3-gated accumulator-clear-after-writeback, plus
  leading-dimension addressing (below). ISA committed + pushed; HW-spec commit
  and both changelog entries pending user.
- **Design B — GEMM-style strided addressing:** a `stride` instruction (CONFIG
  format) sets leading-dimension registers `ld1`=K, `ld2`=N once per matmul (dest
  uses `ld2`); `mult`/`multip` walk sub-blocks with the active strides. Default 0
  = contiguous (v0.4-compatible, MUL format unchanged). Enables row-parallel
  tiling (full utilization) and any M/N/K. Chosen over packing strides into MUL
  (which caps matrix dims ~1023).
- **Instruction taxonomy — CONFIG vs SHAPE:** two formats. `CONFIG` = config
  format (funct-selected banks: leading-dims now, conv/im2col params for v0.6),
  no memory operands, read by MUL and later im2col. `SHAPE` = the reserved
  data-movement *execute* format (transpose, im2col), kept light because CONFIG
  holds the heavy params. The v0.5 leading-dim instruction is a CONFIG bank;
  SHAPE stays reserved for transpose/im2col executes (v0.6+).
- **Transpose:** handled offline — physically transpose constant weights in
  `Process_Weights`, drop the op at legalize. **No hardware transpose op** (LeNet
  is pure CNN; verified the array uses stored rs2 verbatim so it isn't free). A
  live-tensor transpose (transformers) would be a future SHAPE-format transpose
  execute (or transB flag).
- **Assembler gains two stages:** transpose analysis (manifest of transposed
  weight args) + a partitioning pass (emit `CONFIG` + strided tiled `mult`/`multip`).
  No blocked weight layout needed — strides handle tiling.
- **Implementation (Phase 1/2) — DONE & merged to `v0.5-workspace` 2026-07-13**
  via `/fan-out` (`tpu-rtl` + `assembler`) plus a review-fix round (all 5 items).
  TPU RTL: `ld1`/`ld2` context + `stride` decode + VP strided addressing +
  multip/accumulator-clear split, + testbenches. Assembler: CONFIG/`stride`
  ops+encoders, transpose-analysis stage, offline weight transpose, partition
  pass — 64/64 pytest and a REAL Bigger_NN e2e (3 strides, one per matmul).
  Assembler `main.md` steps marked complete; TPU steps left unmarked (Questa).
- **Remaining:** TPU `Verification: PENDING` — run the regression on the Questa
  PC (expect TB_Step3 13/13, TB_Step4 16/16, TB_Step7 10/10, TB_Step8 13/13) and
  confirm `TPU.v` has no `o_debug_val`, then mark the TPU `main.md` steps
  complete. Optional: re-export Bigger_NN from the main tree for portable debug
  `loc` paths. `v0.5-workspace` pushed to origin; not yet merged to `main`.
- **Hardware bring-up / flashing (2026-07-13):** PC↔Arduino serial link rewritten
  to a lock-step CRC-framed ACK/NAK handshake (128-B chunks; commit 1b392ed),
  hardware-validated (ACKs correct) — fixes the UART-overrun that dropped bytes on
  large transfers. Remaining blocker: Bigger_NN flash fails on **breadboard SPI
  signal integrity** (probabilistic, scale-dependent — ~1M SPI bits; the single
  9600 pass was luck; ~800-B / 7-chunk model flashes clean). RTL audit found NO
  deterministic FPGA logic bug (multi-CS-window == continuous), but (A) a latent
  last-byte-vs-CS-deassert race in `SPI_Slave.v` (r_RX_Done async-cleared by CS
  before the i_Clk 2-FF captures it → a window's last byte can silently drop;
  re-rolled ~1000x), (B/C) CS/SCK SI sensitivities, and missing SDC constraints.
  Fix plan: solder a proto board (short leads, common ground, series R, wire
  MISO) = primary SI fix; firmware µs CS-hold before CS-high closes (A) from the
  master; RTL harden (stop CS async-clearing r_RX_Done + add SDC constraints) on
  the next Questa session. 125 kHz = AVR SPI floor. Detail in memory
  `comms-flashing-known-issues`.
- Full plan + rationale in Claude memory (`v0-5-partitioning-redo`).

## Roadmap — end goal LeNet-5

- v0.5 partitioning reaches the FC/GEMM path. v0.6 = on-device `im2col` for conv
  (feature maps are runtime, not compile-time reshapes). Pooling later. Shared
  future gap: a gather/scatter data-movement primitive (im2col, pooling,
  row-parallel reassembly). Detail in memory (`lenet5-end-goal-roadmap`).

## Standing Decisions (dated, with reasons)

- **2026-07-07 — Specifications fully read-only, changelogs included.** The
  old changelog-append instructions once in `Specifications/CLAUDE.md` were
  stale (user deleted that file). User appends all changelog entries himself.
- **2026-07-07 — Tests/ excluded from all agent work.** It is the user's
  personal scratchpad, barely used; not a real test suite.
- **2026-07-07 — nn-trainer may execute training/export scripts** (which
  regenerate `*_Recent.*` artifacts) but proposes code edits as diffs only.
- **2026-07-07 — Agents inherit the session model** (no per-agent pinning).
- **2026-07-07 — /fan-out is route → user approval → execute.** Single-
  section tasks still execute through the pipeline. Main session recommends
  fan-out when a task spans 2+ sections but never launches it unasked.
- **2026-07-07 — Repo relocated to `~/Git` (off OneDrive)** to avoid sync
  file-locks; enabled worktree-per-editing-agent isolation.

## Deferred Work

- **Spec fixes from proofreader pilot (user applies by hand):** 5 logic
  errors (self-negating rst/tpu_rst changelog entry; ISA-changelog
  copy-paste in HW changelog; clamp bounds/x unbound in requant equation;
  relu rs1-vs-src1 naming; undefined M in buffer drain loop), 3 cross-doc
  (note: "XLEN never defined" dropped — word-length agnosticism is intentional,
  see v0.4 design) (v0.3 date conflict 07-01 vs 07-06; reserved address
  0x0 unimplemented in HW spec; "instructions" vs equation in changelog),
  ~15 spelling/grammar + MD013 line-length list. Full report in the
  2026-07-07 session on the main Windows PC.
- Open spec questions to resolve: requant clamp bounds ([-128,127] vs
  [-127,127]); legality of shift = 0; legal command codes inside INPUT
  transmissions; IDLE priority vs pop-order ambiguity.
- README "AI Usage" section still describes unilateral main.md/CLAUDE.md
  management for the Assembler — superseded by approval gates; user rewords.
- `tests/run_regression.sh` + TPU CLAUDE.md mention OneDrive-synced Desktop
  scratch — stale since the `~/Git` move; update when next touched.

## Machine Notes

- **University PC (`CEC-EGB267-05`)** — only machine with Questa
  (`C:\intelFPGA_lite\23.1std\questa_fse\win64\`); otherwise restricted.
- **Linux laptop** — for work needing deep OS access; no Questa (agents go
  `Verification: PENDING` automatically).
- **Main Windows PC** — repo at `C:\Users\littekge\Git\...`; Claude memory
  from before 2026-07-07 is keyed to the old Desktop path.
