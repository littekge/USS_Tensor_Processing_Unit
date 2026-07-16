# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-16)

- **v0.5.1 released and tagged** (`main`, tag on `origin`); v0.4/v0.5.0 also
  shipped/tagged. See "Shipped history".
- **v0.6 specs COMPLETE** — ISA + Hardware Specification both at **v0.6.0**,
  committed and pushed on branch **`v0.6-workspace`**. Two full proofreader
  passes clean.
- **v0.6 RTL:** TPU `main.md` build plan added (5 steps — "Convolution and
  Pooling"); user created the `Pooler.v` skeleton → Step 1 unblocked. No v0.6
  RTL written yet.
- **Next:** draft the Assembler v0.6 plan (in progress), then TPU RTL
  implementation (Questa-gated), comms INPUT-header code, and LeNet-5 artifact
  re-export.

## v0.6 — Convolution and Pooling (specs shipped; implementation pending)

The specs are the source of truth now; only downstream-critical carry-forwards
are kept here.

- **What shipped:** `im2col` (Type SHAPE, A-Format, opcode 1101) and `max`-pool
  (Type POOL, A-Format, 1100), both reading a persistent **window descriptor**
  set by `window` (Type WINCONFIG, new W-Format, 1110 — 11 fields
  chans/inh/inw/outh/outw/winh/winw/strh/strw/padh/padw). A-Format now hosts
  ACT/SHAPE/POOL (`opcode|rs1|rd|aux|reserved|funct3`); old CONFIG→LDCONFIG. HW
  Spec adds the **Pooler** (streaming reducer peer to ALU/Activator; folds one
  value/clock, emits on a Vector_Processor `window_end` strobe), windowed
  addressing + fills in the Vector_Processor, and the descriptor registers +
  windowed decode in the Controller.
- **CRITICAL for the assembler — im2col K-axis is channel-major**
  `k = c·(winh·winw) + ki·winw + kj` (in-ch → win-row → win-col), matching
  PyTorch `(out,in,kh,kw)` weight flatten with NO permutation. Assembler
  weight-flatten MUST match — a mismatch is a silent wrong conv, not a crash.
- **Feature-map layout:** generalized Row-Major, channels outermost (CHW-planar);
  the order>2 rule is generalized; arithmetic ops still ≤2-D.
- **Fills:** im2col out-of-bounds → 0 (hardwired 0x0); max out-of-bounds → minimum
  representable (never wins). Windowed ops are int8-preserving (no requant), write
  contiguously (no `ld`), `aux`=reserved=0.
- **SETTLED — not v0.6 items:** Message Protocol complete (only remaining protocol
  work is *code* for the INPUT header 0x49; external mode = external *input*,
  output via VGA); bias adds intentionally dropped (LeNet <1%); ReLU-only
  activation. See Standing Decisions + memory.

### Free wins available now (no decisions; Phase 0)

Assembler: harden legalizer against silent op-drop; add ReLU lowering (dialect +
`encode_relu` + RTL already exist, but the legalizer never emits `ReluOp`); fix
stale test `test_serializer_and_e2e.py:205` (expects (172,19), correct is
(129,19)). NN: re-export stale LeNet artifacts (2026-06-24, missing
`__M__`/`__scales__`).

## Shipped history (compressed — full detail in memory)

- **v0.5.1** (tagged on `origin`) — SPI_Slave rewritten as a synchronous
  oversampling receiver; fixed large-model flashing (root cause: 5V→3.3V
  resistive divider feeding soft edges into raw async SCK-clock/CS-reset pins).
  Questa PASS + Bigger_NN 3×/3× on hardware. Memory: `comms-flashing-known-issues`.
- **v0.5.0** (tagged) — GEMM-style strided (leading-dimension) addressing:
  `stride`/LDCONFIG sets `ld1`/`ld2`; `mult`/`multip` walk strided sub-blocks
  (0 = contiguous, v0.4-compatible). Transposes offline in the assembler; no HW
  transpose op. Memory: `v0-5-partitioning-redo`.

## Roadmap — end goal LeNet-5

- v0.5 partitioning reached the FC/GEMM path. **v0.6 = on-device `im2col` (conv)
  + `max`-pool** — specs done, RTL/assembler implementation pending (feature maps
  are runtime, not compile-time reshapes). Shared future gap: a general
  gather/scatter data-movement primitive (im2col, pooling, row-parallel
  reassembly). Detail in memory (`lenet5-end-goal-roadmap`).

## Standing Decisions (dated, with reasons)

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
