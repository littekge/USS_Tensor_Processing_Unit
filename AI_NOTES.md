# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-14)

- **v0.5.0 SHIPPED.** Matmul partitioning via GEMM-style leading-dimension
  addressing (`stride`/CONFIG instruction + `multip`/accumulator-clear split).
  Merged to `main` and tagged `v0.5.0` by the user; Questa-verified + real
  Bigger_NN e2e. Full design history in memory `v0-5-partitioning-redo`.
- **v0.5.1 in progress on `v0.5.1-Workspace` — SPI_Slave hardening.** Complete
  internal rewrite of `SPI_Slave.v` to a synchronous oversampling receiver
  (interface byte-identical). **Questa regression PASS + hardware-validated:
  Bigger_NN flashed 3×/3× with zero errors.** This RTL fix alone restored
  flashing reliability — the logic-level converter is now a likely-unneeded
  backup. Committed + pushed (88d299a); not yet released.
- **v0.4 shipped** earlier (`v0.4.0`/`v0.4.1`, Questa + real FPGA verified).

## Tomorrow (in order)

1. **Hardware-Spec update (user's task; `Specifications/` is read-only for
   Claude):** the `SPI_Slave` description still says "pulled from nandland" —
   rewrite it for the new synchronous-oversampling design.
2. **Release v0.5.1:** TPU `main.md` v0.5.1 steps are marked complete and
   `log.md` records the pass (done below); merge `v0.5.1-Workspace` and tag.
3. **v0.6 planning:** on-device `im2col` for conv (see Roadmap).

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
