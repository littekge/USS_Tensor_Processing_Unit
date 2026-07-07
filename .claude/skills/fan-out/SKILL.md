---
name: fan-out
description: Route a task across the repo's sections and run the per-section agents in parallel, with a user-approval checkpoint between planning and execution. Usage - /fan-out <task description>
---

# /fan-out — parallel section dispatch

Drive the `fan-out` workflow (`.claude/workflows/fan-out.js`) through its
two-step, approval-gated flow. Never skip the approval checkpoint.

## Step 1 — Plan

Call the Workflow tool with `name: "fan-out"` and
`args: {task: "<the user's task text>"}`. It returns
`{mode: "plan", plan: {routes, openQuestions}}`.

## Step 2 — Approval checkpoint (mandatory)

- If `openQuestions` is non-empty: put those questions to the user first
  (AskUserQuestion). Fold the answers into the affected routes' subtask
  text before proceeding.
- Present the dispatch plan as a table: section, agent, worktree yes/no
  (editors: tpu-rtl, assembler, comms; shared tree: nn-trainer,
  spec-proofreader), and each subtask. Ask the user to approve, amend, or
  cancel. Do not proceed on silence.

## Step 3 — Execute

On approval, call the Workflow tool again with `name: "fan-out"` and
`args: {routes: [...]}` using the approved (possibly amended) routes.

## Step 4 — Report and merge

Present per section: summary, files touched, commits, pending
verifications (Questa-gated tests), and open decisions. Then:

- Surface every `openDecisions` item to the user — these were deliberately
  left unmade.
- For each worktree branch with commits, propose the per-section merge
  commands and run them only with the user's approval. Surface any merge
  conflict to the user instead of resolving it silently.
- Remind the user of `Verification: PENDING` items if this machine lacks
  Questa.
