# Claude - Repository Index and Routing Rules

> This is the root instruction file for the whole repository. Read it first,
> then defer to the section-level `CLAUDE.md` (when one exists) for the section
> you are working in. Section-level rules win on conflicts within their own
> section.

## Section Map

| Path | Domain / Toolchain | Section CLAUDE.md | Owning Agent | Agent Edit Policy |
|---|---|---|---|---|
| `Specifications/` | Markdown spec sheets (ISA, Message Protocol, Hardware Spec) | none | `spec-proofreader` | **READ-ONLY — never edit (see below)** |
| `System/Tensor_Processing_Unit/` | Verilog HDL 2001, Quartus Prime Lite, Questa sim | `System/Tensor_Processing_Unit/CLAUDE.md` | `tpu-rtl` | Full edit (dark factory) |
| `System/Assembler/` | Python, pytest, StableHLO MLIR → TPU binary | `System/Assembler/CLAUDE.md` | `assembler` | Full edit (dark factory) |
| `System/Communication/` | Arduino C++ (PlatformIO), Python serial/SPI | none | `comms` | Full edit (dark factory) |
| `System/Neural_Networks/` | PyTorch training / export | none | `nn-trainer` | **Propose-only — suggest diffs, never edit directly** |
| `Tests/` | User's personal scratchpad | none | none | **EXCLUDED — agents never read from or write to this directory** |
| `Resources/` | Saved reference material | none | none | Read-only reference |

## Build Artifacts

Any agent or subagent may **regenerate build artifacts** — re-run export/build
scripts that produce generated outputs (`*.npz`, MLIR, binaries, and the like) —
even for a section it does not own, **provided it modifies no _code_ in a
restricted or non-owned location** (nothing under `Specifications/`, no
`CLAUDE.md`, no `Tests/`, and no source in another agent's section).
Regenerating an artifact by running a script is distinct from editing code:
e.g. the `assembler` agent may run the `System/Neural_Networks` export to
regenerate a model's `weights.npz` for an end-to-end test, without touching any
Neural_Networks source (which stays `nn-trainer` propose-only).

## Routing Rules

Route work to the section (and agent) that owns the subject matter:

- Verilog / `.v` files / testbenches / Questa regression → `tpu-rtl`
- StableHLO-to-binary assembly, serialization, weight processing →
  `assembler`
- Arduino firmware, PC↔Arduino serial, Arduino↔FPGA SPI → `comms`
- PyTorch models, training, `.mlir` export scripts → `nn-trainer`
- Spec sheet review (logic errors, spelling, cross-doc consistency) →
  `spec-proofreader`
- `Tests/` is out of scope for all agents, always.

## Decision Authority and Approval Gates

Claude is an agent of the user's own work. Any decision that is even
somewhat large is the user's to make — architecture, interfaces, build-plan
content and ordering, conventions, dependencies, anything hard to reverse.
**If unsure whether a decision is large enough to escalate, it is: always
ask.** Subagents cannot prompt the user directly, so a subagent that hits an
open decision must stop that line of work and surface the decision in its
final report instead of choosing.

Specific approval gates:

- **`main.md` files:** new build plans and major edits (adding, removing,
  reordering, or rewording steps; scope changes) require user approval
  before being written. Routine bookkeeping is allowed without approval:
  marking an existing step complete once its tests have actually passed.
- **`CLAUDE.md` files (all of them, including this one):**
  propose-then-approve. Agents may draft a change as a diff in chat, but
  nothing is written to any `CLAUDE.md` until the user explicitly approves
  in that session.

## Specifications Are Read-Only

**No agent or subagent may ever edit, create, or delete anything under
`Specifications/`.** The user writes and manages all spec documents and
changelogs personally. This rule has no exceptions and overrides any other
instruction.

The `spec-proofreader` agent exists only to catch mistakes the user makes
(logic errors, spelling, inconsistencies between documents). It reports
findings as a structured list in chat (file, line, issue, suggested
wording) and never fixes anything on disk.

## Questa Machine Gate

Simulation runs only on the Questa PC (`vsim.exe` present at
`C:\intelFPGA_lite\23.1std\questa_fse\win64\`; hostname `CEC-EGB267-05` is a
secondary identifier). On any other machine: still write and update
testbenches, but record `Verification: PENDING` in
`System/Tensor_Processing_Unit/log.md` and never attempt to simulate. Full
details in `System/Tensor_Processing_Unit/CLAUDE.md`.

## Parallel-Agent Git Discipline

- Editing agents launched in parallel run in **isolated git worktrees**, one
  per agent, and commit normally per their section's conventions.
- Read-only agents (`spec-proofreader`, routing/synthesis steps) run in the
  shared tree and make no commits.
- Worktree branches merge back **per section** after the user reviews the
  fan-out report. Sections are path-disjoint, so merges should be clean;
  surface any conflict to the user instead of resolving it silently.
- An agent working directly in the shared tree must stage only its own
  section's paths (`git add <section>/`), never `git add .`.

## Global Style Rules

These apply in every section, in addition to section-level rules.

### Chat Responses

- **Summarize** responses into brief, high-level overviews to optimize token
  usage.
- **Remove** extra conversational prose (introductory text, pleasantries,
  etc.).
- **Prioritize** functional, bulleted responses.

### Comment Rules

- **Rule 1 — Names explain _what_**: choose clear, descriptive names; a good
  name needs no comment.
- **Rule 2 — Code explains _how_**: readable code, not comments restating it.
- **Rule 3 — Comments explain _why_**: comment only where the reason behind a
  decision is not obvious from the code.
