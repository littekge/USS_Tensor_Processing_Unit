# AI Project Journal

> Portable session context for Claude across the user's machines. Maintained
> by Claude. Keep under ~3k tokens: prune stale entries; when a decision
> hardens into a permanent rule, graduate it into the appropriate CLAUDE.md
> (via the user's approval gate) and delete it here. Binding rules live in
> CLAUDE.md files, not here — this file holds decision history (with reasons),
> in-flight state, and machine notes.

## Current State (2026-07-07)

- 3-layer agent workflow built on branch `AI-workflows-test`: root
  `CLAUDE.md` (60aa7f9), five agents in `.claude/agents/` (4762e1e),
  `/fan-out` workflow + skill (a5763b8, portability fix ee76fde).
- `/fan-out` plan step smoke-tested PASS. **Execute step untested** — agent
  types register at session start; first end-to-end run pending a restart.
- Spec-proofreader pilot ran over the v0.3 specs; fixes deferred (below).

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

- **Spec fixes from proofreader pilot (user applies by hand):** 6 logic
  errors (self-negating rst/tpu_rst changelog entry; ISA-changelog
  copy-paste in HW changelog; XLEN never defined; clamp bounds/x unbound in
  requant equation; relu rs1-vs-src1 naming; undefined M in buffer drain
  loop), 3 cross-doc (v0.3 date conflict 07-01 vs 07-06; reserved address
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
