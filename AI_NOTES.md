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

## Next Focus — v0.5 "Partitioning Update"

- **Goal:** compute matrix multiplications larger than the hardware systolic
  array by unrolling a large matmul into a series of small matmul ops that fit
  the array. Enables matmuls beyond the physical array size.
- Status: kickoff. Build plan(s) not yet drafted — new `main.md` steps require
  user approval before being written (approval gate).

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
