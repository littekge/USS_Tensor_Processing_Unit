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
- **v0.5 VERIFIED & RELEASING (2026-07-14).** User confirmed the full TPU
  regression passes on the Questa PC (TB_Step3/4/7/8); TPU `main.md` Steps 1–5
  marked ✅ Complete and `log.md` updated with the pass + known-issue note. RTL is
  release-clean (no `o_debug_val`). Specs already committed at v0.5.0 (ISA
  561fd69, HW spec 0506a04). On the flashes that completed, all NNs computed
  correctly with the new partitioning. **User is handling the merge to `main`, the
  `v0.5.0` tag, publish, and the Neural_Networks working-tree changes (incl. the
  Tiny_NN 4→1000 resize for flashing tests) personally.** The flashing SI issue is
  separate/open (see below + memory `comms-flashing-known-issues`) and does NOT
  block the release.
- **Hardware bring-up / flashing:** serial hop (PC↔Arduino) reliable via lock-step
  CRC-framed ACK/NAK handshake (128-B chunks; commit 1b392ed, hardware-validated).
  Remaining blocker is the **SPI hop (Arduino→FPGA)**.
- **Flashing root cause reframed (2026-07-14 diagnostic session).** `COM_ERROR` is
  an FPGA-side `Programmer.v` parse-desync (byte at a command boundary isn't
  `M/P/S`), not a byte-volume/SI-accumulation limit. Both Tiny_NN (7310 B) and
  Bigger_NN (132 KB) binaries parse 100% clean → **no assembler bug**. Failure
  position tracks the **MEM→PROG stream boundary, not byte count**: MEM (weight)
  writes absorb byte corruption silently (FSM counts `mem_len` regardless of
  value); the PROG run is zero-tolerance (command-code checkpoint every 17 B,
  instruction data never aliases `M/P/S`) → any dropped/misframed byte → COM_ERROR
  within ~17 B. Tiny fails ~7000 (PROG starts @3008); Bigger fails ~100000 (PROG
  starts @102214, after its 102 KB MEM). Bigger's weights are likely corrupted
  too, silently. Real cause: 5V→3.3V level shift is a **2k/3.3k resistive divider**
  (Thevenin ≈1.25 kΩ) feeding **~100 ns edges** into pins `SPI_Slave.v` uses as a
  raw async clock (`w_SPI_Clk`) and async reset (`i_SPI_CS_n`) — ~20–50× too slow.
  Scope (Siglent SDS1104X-HD): edges clean/monotonic but slow; **no runts/glitches
  on SCK or CS** → fault is internal to the FPGA, invisible on the wire (on-die
  double-clock/metastability + latent RTL last-byte/CS race). Wire-invisible RTL
  hazards (A last-byte/CS race, SDC gaps) still on the fix list.
- **Flashing next step:** user will **buy a proper logic-level converter** (74LVC
  push-pull, 5V-tolerant, fast 3.3 V edges) before reinvestigating. Buffer-and-
  retest is decisive: errors vanish → soft-edge-into-async-clock confirmed; persist
  → apply the `SPI_Slave.v` deglitch (Questa-gated). Detail in memory
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
