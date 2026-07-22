# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-22)

- **Working on `v0.7-workspace`** (branched off `main` after v0.6.0). Tree clean.
- **v0.6.0 RELEASED & tagged** on `main`/`origin` (GitHub prerelease "Convolution
  and Pooling" published 2026-07-21); `v0.6-workspace` merged to `main` and
  deleted (locally + origin). All prior versions shipped/tagged.
- **LeNet-5 WORKING ON FPGA** — first full conv net on the platform, classifies
  correctly. Bring-up found two bugs:
  - **Bug 1 (RTL) — FIXED & VERIFIED (shipped in v0.6.0):** `Vector_Processor.v`
    windowed gather (im2col/max) never special-cased the isolated `0x1` I/O
    buffer, so conv1 read input pixels out of weight memory → all logits wrong.
    Fixed to mirror the non-windowed `src_is_buf` path (hold addr `0x1`, element
    index on the offset port; OOB→`0x0`). Questa PASS + hardware-confirmed.
  - **Bug 2 (assembler) — fix now SPEC'D (v0.7 `move`), lowering pending:** fc3's
    tiled output writes logits 8,9 to `rd=9` (=`conv1.weight[7,8]`), corrupting
    weights across re-runs (Programmer re-runs without reload). Only ±1; never
    flips argmax. Retired once the assembler lowers `move` into the fc3 tail.
- **v0.7.0 IN PROGRESS** = proper comms link + demo. **Spec phases DONE:**
  - **ISA** (`7a564ac`, ISA v0.7.0): SHAPE-opcode `move` instruction (opcode 1101
    / funct3 0x1, A-Format) defined — dissolves Bug 2 by copying a computed result
    into the `0x1` I/O buffer for readout. Same commit fixed the requant `shift=0`
    UB (rounding bias `(1 << shift) >> 1`; RTL `Vector_Processor.v:256` + golden
    `Interpreter.py`/`Emulator.py` already guarded it — verified, no code change).
  - **HW spec** (`c260587`, HW spec v0.7.0): `move` added to the Controller
    writeback-skip clause (writes to memory in Execute, like `im2col`) + version
    bump. Deliberately NOT described in Vector_Processor: `move` is the already-
    listed device-mem→device-mem route, no new data-movement mechanism, and the VP
    section documents mechanisms not instructions. Requant already correct in HW
    spec since v0.6 (ISA was the lagging doc; v0.7 ISA edit closed the mismatch).
    spec-proofreader ran clean on both docs.
- Remaining v0.7 work: **assembler `move` lowering (retires Bug 2) — NEXT**,
  INPUT-header (`0x49`) code, and the demo.

## v0.6 carry-forwards for v0.7 (specs are the source of truth for the rest)

v0.7 = comms link + demo, so the LeNet-5 bring-up plumbing is what matters here:

- **No input-embed in the assembler:** the network input is a runtime `@main` arg
  mapped to `0x1` but emitted with ZERO bytes. Weights ship in `MEM.bin` (0x2+);
  the image reaches `0x1` ONLY via the Message-Protocol INPUT header (`0x49`).
  The Programmer already supports it (external mode; preserves the resident
  program, runs on STOP). `System/LeNet-5_Test.py` streams a quantized MNIST digit
  via INPUT/MEM(0x1)/STOP. **Only remaining protocol work is *code* for the INPUT
  header** (external mode = external *input*; output shown on VGA, no readback).
- **Golden/verify tooling — `System/Assembler/Interpreter/`:** `Emulator.py`
  (golden layer-level int8 LeNet) + `Interpreter.py` (bit-exact `PROGRAM.bin`
  executor over a modeled ISA memory: `0x0`=zero, `0x1`=isolated 1024-word buffer,
  `>=0x2` flat main). Both reuse the assembler's quant helpers so they can't
  drift; they localized both bugs with no hardware.

## Shipped history (compressed — full detail in memory)

- **v0.6.0** (tagged on `main`/`origin`; GitHub prerelease) — on-device
  convolution + pooling: `window`/`im2col`/`max` instructions (W/A-Format), a
  streaming Pooler, windowed addressing in Controller/Vector_Processor, and
  assembler conv/pool lowering + channel-major conv-weight flatten. Questa PASS;
  LeNet-5 hardware-verified. Memory:
  `lenet5-hardware-bringup-rtl-windowed-0x1-bug`.
- **v0.5.1** (tagged on `origin`) — SPI_Slave rewritten as a synchronous
  oversampling receiver; fixed large-model flashing (root cause: 5V→3.3V
  resistive divider feeding soft edges into raw async SCK-clock/CS-reset pins).
  Questa PASS + Bigger_NN 3×/3× on hardware. Memory: `comms-flashing-known-issues`.
- **v0.5.0** (tagged) — GEMM-style strided (leading-dimension) addressing:
  `stride`/LDCONFIG sets `ld1`/`ld2`; `mult`/`multip` walk strided sub-blocks
  (0 = contiguous, v0.4-compatible). Transposes offline in the assembler; no HW
  transpose op. Memory: `v0-5-partitioning-redo`.

## Roadmap — end goal LeNet-5

- **LeNet-5 end goal REACHED on hardware (v0.6.0).** v0.5 gave the FC/GEMM path;
  v0.6 added on-device `im2col` (conv) + `max`-pool. Remaining polish is v0.7.0:
  proper comms link + demo + the `move` instruction (dissolves Bug 2). Longer-term
  generalization: a unified gather/scatter data-movement primitive. Detail in
  memory (`lenet5-end-goal-roadmap`).

## Standing Decisions (dated, with reasons)

- **2026-07-17 — `0x1` is purely an I/O staging buffer; Bug 2 fixed later via a
  *move* instruction under the SHAPE opcode.** Rather than special-case tiled
  writes to the isolated `0x1` buffer, the plan is a SHAPE-opcode `move` that
  copies a computed result from main memory into `0x1` for readout. ISA/spec
  change — user-owned (Specifications are read-only). Until then LeNet's argmax is
  unaffected (Bug 2 is a ±1 wobble).
- **2026-07-16 — spec-proofreader reframed as an independent fresh skeptic.**
  Was a 3-category checklist (logic/cross-doc/spelling) that missed omissions —
  notably the `window` instruction referencing window-descriptor registers that
  were never defined (unlike `stride`→`ld1`/`ld2`). Root cause: checklist framing
  can't catch novel classes, and feeding it our change-list anchored it to our
  blind spots. Fix (`.claude/agents/spec-proofreader.md`): mindset-first (assume
  something's wrong, read whole doc, stay independent of the caller's framing);
  omissions/completeness promoted to first category with a parallel-construct
  heuristic; error types demoted to non-exhaustive examples; keeps spelling/
  grammar; drops markdown-formatting checks (user runs a linter). When invoking
  it, give minimal framing — not a checklist to confirm.
- **2026-07-15 — Message Protocol complete; results shown via VGA, not read
  back.** No protocol/spec work for v0.6. "External mode" = external *input*;
  output stays on-device and the first 10 outputs (per-digit logit scores, index
  i = digit i) are written to a VGA screen — a human reads the argmax. No wire
  readback, no on-device argmax/softmax needed. Only remaining protocol work is
  *code* to implement the INPUT header (0x49) for feeding inputs without
  reprogramming.
- **2026-07-15 — Bias adds intentionally dropped; NOT a gap.** Assembler's
  `bias_removal.py` drops all bias `add`s. Validated on LeNet-5 (<1% accuracy
  drop). Tiny_NN/Bigger_NN lose more but are functionless test nets. No
  per-channel/broadcast-add instruction is needed; do not re-add to the v0.6
  spec list. Docstring corrected (was misleadingly credited to "target
  networks"). Memory: `bias-removal-validated-lenet5`.
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

- TPU `CLAUDE.md:75` still frames sim-scratch placement around the "OneDrive-synced
  Desktop" (pre-`~/Git`-move rationale) — re-verified present 2026-07-22.
  Low-priority wording cleanup (reword generically: keep scratch off the synced
  tree) when that file is next touched; needs the CLAUDE.md approval gate.
  The `tests/run_regression.sh` half is unverifiable — `Tests/` is agent-excluded.
  `log.md` entries are dated history — leave them.

## Machine Notes

- **University PC (`CEC-EGB267-05`)** — only machine with Questa
  (`C:\intelFPGA_lite\23.1std\questa_fse\win64\`).
- **Linux laptop** — deep OS access; no Questa (agents go `Verification: PENDING`).
- **Main Windows PC** — repo at `C:\Users\littekge\Git\...`; Claude memory from
  before 2026-07-07 keyed to the old Desktop path.
