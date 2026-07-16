# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-15)

- **v0.5.1 released and tagged** (`main`, tag on `origin`). Latest shipped work;
  v0.4/v0.5.0 also shipped and tagged. See "Shipped history" below for detail.
- **Active focus: v0.6 (im2col + max-pool for conv)** — design in progress this
  chat; see "Next Focus" for decisions. No code/spec written yet.

## Next Focus — v0.6 (im2col for convolution)

Direction (from memory `lenet5-end-goal-roadmap`): on-device `im2col` for conv —
must run on-device (conv feature maps are runtime intermediates, not compile-time
reshapes). CONFIG/SHAPE taxonomy in memory `v0-5-partitioning-redo`.

**Target model (confirmed as checked in):** modernized LeNet-5 — 1×28×28 input,
2× conv 3×3 (6, 16 ch), 2× max-pool 2×2, 3× FC (400→120→84→10), ReLU throughout,
logits out. NOT classic (no 32×32/5×5/tanh/avg). Trained ~95.3% (under-trained:
5k-subset/5-epoch).

### v0.6 spec additions the user must author (from 2026-07-15 repo audit)

Grouped; specs are user-owned/read-only to agents — this is the enumeration only.

See "v0.6 ISA design decisions" below for the resolved shape of these. Spec text
still to author (all user-owned):

- **im2col / convolution:** generalize A-Format into the unary Format (see
  ACT-windowed merge below); bind the SHAPE Type opcode (0001-0111/1011-1110
  free); im2col Function row; window-config under CONFIG (new Function or multiple
  writes — settle Stage 2); HW-Spec copy-datapath. **Drags in:** a feature-map
  layout convention — ISA:22 currently forbids order>2 tensors, but conv operands
  are N×C×H×W.
- **Pooling — MAX (decided 2026-07-15).** Bind the POOL Type opcode (shares the
  execute Format above); max Function row; window params via the shared CONFIG
  window-config; HW-Spec reduce-datapath. Needs a *reduction* (max) path the ALU
  lacks (ADD-only). Chosen over avg: cheap compare tree, scale-preserving in int8
  (no requant, fits calibration). Avg rejected (reuses matmul but breaks the
  scale-preserving assumption).
- **Final classifier — no on-device argmax/softmax needed.** Model emits raw
  logits; the FPGA writes the first 10 outputs (= per-digit scores, index i =
  digit i) to a VGA screen and a human reads the max. Softmax only needed if
  displayed *calibrated probabilities* are wanted (not planned).

**SETTLED — not spec items:**
- Message Protocol: COMPLETE. No spec work. Remaining is *code* to implement the
  INPUT header (0x49) — feed a new input without reprogramming (comms/assembler
  task). "External mode" = external *input*; output stays on-device (first 10
  values → VGA screen), so no wire-readback path is needed. (Audit misread
  external mode as MISO readback — corrected.)
- Broadcast/per-channel bias add: NOT needed — bias intentionally dropped
  (LeNet-5 <1% drop). See Standing Decisions + memory `bias-removal-validated-lenet5`.
- Activation: ReLU only — already in ISA+RTL; no spec change (tanh would need a
  new ACT funct, but target model is ReLU).
- Memory enlargement: ISA/HW already parametric; sizing is a main.md/RTL task,
  not spec text.

### v0.6 ISA design decisions (in progress, 2026-07-15 — user authors spec)

**ISA Stage 1 (rename/reorg) — APPLIED 2026-07-15.** Hierarchy now **Format (bit
layout) → Type (opcode) → Function (funct field)**. Formats renamed to arbitrary
letter codes **M/A/E/C/S-Format** (was MUL/ACT/ELEM/CONFIG/SYSTEM); letters are
convenience-only, no binding logic (a format's purpose may shift). Function =
funct3, except S-Format = funct7. Bit layouts + opcodes UNCHANGED
(proofread-verified). Types keep MUL/ACT/ELEM/CONFIG/SYSTEM; Functions keep
mult/multip/relu/add/stride/end/syscall. One Format may host multiple Types (see
ACT-windowed merge below). Changelog v0.6.0 entry still to write (user). Adopt
this vocab going fwd.

- **im2col mechanism = dedicated SHAPE Type + CONFIG bank** (resolves the old
  gather-vs-SYSCALL fork). Heavy geometry can't fit one execute instruction: two
  24-bit addresses (48b) + ~10 geometry fields overflow 128b, or force ≤255 dim
  caps + a HW divider for H_out/W_out. Geometry is per-layer-constant →
  CONFIG-once / execute-many (mirrors v0.5 `stride`). **APPLIED to ISA:** new
  **WINCONFIG** Type + new **W-Format** (opcode→Format kept 1:1); old CONFIG Type
  renamed **LDCONFIG** with "Configuration" as the category; `window-config`
  packs H/W/C/H_out/W_out (12b each) + kh/kw/sh/sw/ph/pw (4b each), H_out/W_out
  explicit (no HW divide).
- **Shared windowing core** (im2col + pool both read the WINCONFIG descriptor):
  C,H,W, kh,kw, sh,sw, ph,pw, H_out,W_out. **Feature-map layout (#1, decided):**
  generalized Row-Major, C outermost (CHW-planar) — order>2 rule generalized (not
  an exception); arithmetic ops still ≤2-D. **im2col K-axis order (#2, LOCKED):**
  channel-major `k = c·(kh·kw) + ki·kw + kj` (in-ch → kh → kw); matches PyTorch
  `(out,in,kh,kw)` weight flatten with NO permutation, and aligns with CHW
  traversal. Assembler weight-flatten MUST match — mismatch is a silent wrong
  conv, not a crash.
- **Executes (SHAPE/POOL on A-Format), exact edits proposed 2026-07-15:** im2col =
  Type SHAPE (opcode 1101, funct3 0x0); max = Type POOL (opcode 1100, funct3 0x0).
  `aux`+reserved = 0 (all geometry from WINCONFIG; both int8-preserving, no requant;
  outputs written contiguously — no `ld` needed). No future funct3 reserved
  (transpose/avg added only if/when needed). Fills: im2col out-of-bounds → 0 (from
  0x0); max out-of-bounds → min representable (never wins). W-Format fields renamed
  to `<what><axis>`: inh/inw/chans/outh/outw/winh/winw/strh/strw/padh/padw.
- **Window descriptor registers (gap found 2026-07-15, fix proposed):** the
  `window` instruction loaded a descriptor with no registers defined (unlike
  `stride`→`ld1`/`ld2`); `im2col`/`max` read undefined state. Fix: add a *window
  descriptor* register bank (chans/inh/inw/outh/outw/winh/winw/strh/strw/padh/padw)
  to the Registers section, set by `window`, persists until next `window`/reset, no
  default. Proofreader missed it (absence, not a written error) — sharpen its prompt
  to check config instructions against the stride→ld precedent. Awaiting apply +
  re-proofread, then changelog final.
- **ACT + windowed MERGED → one unary Format (decided 2026-07-15).** relu,
  im2col, transpose, maxpool are all 1-src→1-dst, so instead of a new windowed
  Format, *generalize A-Format* (`opcode|src|dst|aux16|reserved|funct3`) to host
  three Types: **ACT** (aux=len; Function relu), **SHAPE** (Functions im2col now,
  transpose later — pure movement), **POOL** (Functions max now, avg later —
  windowed reduction). 6 Formats → 5; zero new bits (A-Format already carries
  src+dst+funct3 + 73 free). Decoder routes by Type; POOL→reduce datapath.
  **Watch:** assumes 1-src→1-dst — a future 2-source op (e.g. indexed gather)
  would NOT fit; use the ELEM shape or its own Format then. Geometry stays in
  CONFIG banks. MUL/ELEM NOT merged (per-operand dims + requant vs shared dims);
  CONFIG/SYSTEM unique.

### Free wins available now (no decisions; Phase 0)

Assembler: harden legalizer against silent op-drop; add ReLU lowering (dialect +
`encode_relu` + RTL already exist, legalizer never emits `ReluOp`); fix stale test
`test_serializer_and_e2e.py:205` (expects (172,19), correct is (129,19)). NN:
re-export stale LeNet artifacts (2026-06-24, missing `__M__`/`__scales__`).

Carry-over housekeeping: optional cosmetic — in the HW-Spec `SPI_Slave`
§Transmit, *MISO* is italicized while the signal list uses bare "MISO".

## Shipped history (compressed — full detail in memory)

- **v0.5.1** (tagged on `origin`) — SPI_Slave rewritten as a synchronous
  oversampling receiver; fixed large-model flashing (root cause: 5V→3.3V
  resistive divider feeding soft edges into raw async SCK-clock/CS-reset pins).
  Questa PASS + Bigger_NN 3×/3× on hardware. Memory: `comms-flashing-known-issues`.
- **v0.5.0** (tagged) — GEMM-style strided (leading-dimension) addressing:
  `stride`/CONFIG sets `ld1`/`ld2`; `mult`/`multip` walk strided sub-blocks
  (0 = contiguous, v0.4-compatible). Transposes offline in the assembler; no HW
  transpose op. Memory: `v0-5-partitioning-redo`.

## Roadmap — end goal LeNet-5

- v0.5 partitioning reaches the FC/GEMM path. v0.6 = on-device `im2col` for conv
  (feature maps are runtime, not compile-time reshapes). Pooling later. Shared
  future gap: a gather/scatter data-movement primitive (im2col, pooling,
  row-parallel reassembly). Detail in memory (`lenet5-end-goal-roadmap`).

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
