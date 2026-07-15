# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-15)

- **v0.5.1 RELEASED (merged to `main`).** SPI_Slave hardening — complete internal
  rewrite of `SPI_Slave.v` to a synchronous oversampling receiver (interface
  byte-identical). Questa regression PASS + hardware-validated (Bigger_NN flashed
  3×/3×, zero errors); the RTL fix alone restored flashing reliability, so the
  logic-level converter is now a likely-unneeded backup. HW-Spec SPI_Slave
  description + changelog bumped to v0.5.1 (commit 59eb1b3), proofread clean.
  On `main` via 88d299a / 7689ff5 / 59eb1b3. **NOTE: no `v0.5.1` git tag exists
  yet — only `v0.5.0` is tagged; confirm whether v0.5.1 still needs a tag.**
- **v0.5.0 SHIPPED** — matmul partitioning via GEMM-style leading-dimension
  addressing (`stride`/CONFIG + `multip`/accumulator-clear split); merged to
  `main` and tagged `v0.5.0`; Questa + real Bigger_NN e2e. Detail in memory
  `v0-5-partitioning-redo`.
- **v0.4 shipped** earlier (`v0.4.0`/`v0.4.1`, Questa + real FPGA verified).

## Next Focus — v0.6 (im2col for convolution)

A fresh chat is starting for v0.6. Direction (from memory
`lenet5-end-goal-roadmap`): on-device `im2col` for conv — a CONFIG conv-param
bank (bank1) + a SHAPE-format im2col *execute* (`src`/`dst`/`funct`, heavy params
read from CONFIG); see the CONFIG/SHAPE taxonomy in memory
`v0-5-partitioning-redo`. Must run on-device (conv feature maps are runtime
intermediates, not compile-time reshapes). Open forks: a general gather/scatter
data-movement primitive vs. special-purpose ops; confirm target-LeNet
activations (ReLU vs tanh) and pooling (avg vs max).

Carry-over housekeeping: (1) confirm/apply the `v0.5.1` git tag (see Current
State); (2) optional cosmetic — in the HW-Spec `SPI_Slave` §Transmit, *MISO* is
italicized while the signal list uses bare "MISO".

## v0.5.1 — SPI_Slave Hardening & Flashing Resolution

- **Root cause (2026-07-14):** large-model flashing failed with an FPGA-side
  `COM_ERROR` (a `Programmer.v` parse-desync) — not an assembler bug (both
  binaries parse clean) and not byte-count (failure tracked the MEM→PROG stream
  boundary). Real cause: the 5V→3.3V **2k/3.3k resistive divider** fed ~100 ns
  edges into pins `SPI_Slave.v` used as a raw async clock (SCK) and async reset
  (CS); the fault was internal to the FPGA — the scope showed clean-but-slow
  edges, no glitches. Full analysis in memory `comms-flashing-known-issues`.
- **Fix:** rewrote `SPI_Slave.v` internals as a synchronous oversampling receiver
  (2-FF sync + SCK/CS debounce + `i_Clk` edge detection + synchronous CS framing),
  removing the raw-SCK-clock, async-CS-reset, last-byte race, and CDC hazards.
  Ports/`SPI_MODE` byte-identical (user's hard rules: changes confined to
  `SPI_Slave.v`, interface unchanged). **Questa PASS + Bigger_NN 3×/3× clean on
  hardware.** Converter track parked as an optional backup, not a blocker.
  **Released 2026-07-15** — merged to `main`; HW-Spec + changelog updated to
  v0.5.1 (proofread clean). Git tag still pending (see Current State).

## v0.5.0 — Partitioning Update (shipped, for reference)

GEMM-style strided (leading-dimension) addressing: a `stride` (CONFIG-format)
instruction sets `ld1`/`ld2`; `mult`/`multip` walk strided sub-blocks (default
0 = contiguous, v0.4-compatible). Weight transposes handled offline in the
assembler; no hardware transpose op (SHAPE format reserved for v0.6+ transpose/
im2col executes). Design rationale + CONFIG-vs-SHAPE taxonomy in memory
`v0-5-partitioning-redo`.

## Roadmap — end goal LeNet-5

- v0.5 partitioning reaches the FC/GEMM path. v0.6 = on-device `im2col` for conv
  (feature maps are runtime, not compile-time reshapes). Pooling later. Shared
  future gap: a gather/scatter data-movement primitive (im2col, pooling,
  row-parallel reassembly). Detail in memory (`lenet5-end-goal-roadmap`).

## Standing Decisions (dated, with reasons)

- **2026-07-07 — Specifications fully read-only, changelogs included.** User
  appends all changelog entries himself (old `Specifications/CLAUDE.md` deleted).
- **2026-07-07 — Tests/ excluded from all agent work** (user's personal
  scratchpad, not a real suite).
- **2026-07-07 — nn-trainer may execute training/export scripts** (regenerating
  `*_Recent.*` artifacts) but proposes code edits as diffs only.
- **2026-07-07 — Agents inherit the session model** (no per-agent pinning).
- **2026-07-07 — /fan-out is route → user approval → execute.** Main session
  recommends fan-out for 2+ section tasks but never launches it unasked.
- **2026-07-07 — Repo relocated to `~/Git`** (off OneDrive) to avoid sync
  file-locks; enables worktree-per-editing-agent isolation.

## Deferred Work

- **Spec fixes from proofreader pilot (user applies by hand):** 5 logic errors,
  3 cross-doc, ~15 spelling/grammar + MD013 line-length. Full report in the
  2026-07-07 session on the main Windows PC.
- Open spec questions: requant clamp bounds ([-128,127] vs [-127,127]); legality
  of shift = 0; legal command codes inside INPUT transmissions; IDLE priority vs
  pop-order ambiguity.
- README "AI Usage" section describes superseded unilateral main.md/CLAUDE.md
  management for the Assembler — user rewords.
- `tests/run_regression.sh` + TPU CLAUDE.md mention OneDrive-synced Desktop
  scratch — stale since the `~/Git` move; update when next touched.

## Machine Notes

- **University PC (`CEC-EGB267-05`)** — only machine with Questa
  (`C:\intelFPGA_lite\23.1std\questa_fse\win64\`).
- **Linux laptop** — deep OS access; no Questa (agents go `Verification: PENDING`).
- **Main Windows PC** — repo at `C:\Users\littekge\Git\...`; Claude memory from
  before 2026-07-07 keyed to the old Desktop path.
