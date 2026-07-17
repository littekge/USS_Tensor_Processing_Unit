# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-17)

- **v0.5.1 released and tagged** (`main`, `origin`); v0.4/v0.5.0 shipped/tagged.
- **v0.6 IMPLEMENTED, on `v0.6-workspace`** (specs at v0.6.0; RTL Questa PASS
  2026-07-16; assembler 79 pass / 1 pre-existing fail). NOT yet merged to `main`.
- **LeNet-5 HARDWARE BRING-UP (2026-07-17) — first end-to-end FPGA run debugged.**
  Symptom: wrong logits. Root-caused **bit-exact** with a new software interpreter
  (see the LeNet bring-up section). Two bugs:
  - **Bug 1 (primary, RTL) — being fixed:** `Vector_Processor.v` windowed-gather
    (im2col/max) never implemented the `0x1`-buffer special-case addressing, so
    conv1 read 783/784 input pixels out of weight memory → all logits wrong. Fix =
    mirror the non-windowed `src_is_buf` logic (hold addr `0x1`, element index on
    the offset port; OOB→`0x0`). Machine-gated: **Verification PENDING** (Questa PC).
  - **Bug 2 (secondary, assembler) — deferred:** fc3's tiled output writes logits
    8,9 to `rd=9` (=`conv1.weight[7,8]`), corrupting weights across re-runs
    (Programmer re-runs without reload). Only ±1; does not flip argmax.
- **Next:** (1) simulate the Bug 1 fix on the Questa PC; (2) re-flash + re-test
  LeNet; (3) Bug 2 fix (see move-instruction decision); (4) merge
  `v0.6-workspace` → `main`.

## v0.6 — Convolution and Pooling (specs + implementation merged to v0.6-workspace)

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

### LeNet-5 hardware bring-up — tooling and findings (2026-07-17)

- **No input-embed in the assembler:** the network input is a runtime `@main` arg
  (`inputs[0][0]`), mapped to `0x1` but emitted with ZERO bytes. Weights ship in
  `MEM.bin` (0x2+); the image reaches `0x1` ONLY via the Message-Protocol INPUT
  header (`0x49`), which the Programmer already supports (external mode; preserves
  the resident program, runs on STOP). `System/LeNet-5_Test.py` streams a known
  MNIST digit (quantized with conv1 `S_in=__scales__[conv1.weight][0]`) via
  INPUT/MEM(0x1)/STOP using `Send_2_Arduino`.
- **New golden/verify tooling — `System/Assembler/Interpreter/`:** `Emulator.py`
  (golden layer-level int8 LeNet model) and `Interpreter.py` (bit-exact executor
  of the actual `PROGRAM.bin` against a modeled ISA memory: `0x0`=zero,
  `0x1`=isolated 1024-word buffer, `>=0x2` flat main). Both reuse the assembler's
  quant helpers so they can't drift. They localized BOTH bugs with no hardware and
  are the reference for the Questa regression.
- **Exonerated:** network, quantization, calibration, bias-removal, requant math,
  im2col K-order, accumulator/tiling — all correct (golden argmax = true label).
  The failure was purely RTL windowed addressing (Bug 1) + assembler output-tiling
  (Bug 2). See memory `lenet5-hardware-bringup-rtl-windowed-0x1-bug`.

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
