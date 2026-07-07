export const meta = {
  name: 'fan-out',
  description: 'Route a task across repo sections, then (after user approval) execute per-section agents in parallel',
  whenToUse: 'Multi-section tasks in the USS TPU repo. Two-step: pass {task: "..."} to get a dispatch plan for user approval; pass {routes: [...]} (approved) to execute.',
  phases: [
    { title: 'Route', detail: 'one agent maps the task onto repo sections' },
    { title: 'Execute', detail: 'one section agent per route, editors in worktrees' },
  ],
}

// Section agents and whether they edit tracked files.
// Editors get worktree isolation; non-editors run in the shared tree
// (nn-trainer is propose-only, spec-proofreader is read-only).
const AGENTS = {
  'tpu-rtl': { editing: true },
  'assembler': { editing: true },
  'comms': { editing: true },
  'nn-trainer': { editing: false },
  'spec-proofreader': { editing: false },
}

// args may arrive JSON-stringified depending on the caller; normalize first
let input = args
if (typeof input === 'string') {
  try { input = JSON.parse(input) } catch (e) {
    throw new Error('fan-out args string is not valid JSON: ' + e.message)
  }
}

// ---------- Step 1: plan (args = {task}) ---------- //

if (input && input.task) {
  phase('Route')
  const ROUTE_SCHEMA = {
    type: 'object',
    required: ['routes', 'openQuestions'],
    properties: {
      routes: {
        type: 'array',
        items: {
          type: 'object',
          required: ['section', 'agent', 'subtask'],
          properties: {
            section: { type: 'string', description: 'repo-relative section path' },
            agent: { type: 'string', enum: Object.keys(AGENTS) },
            subtask: { type: 'string', description: 'complete, self-contained instruction for that agent' },
            rationale: { type: 'string' },
          },
        },
      },
      openQuestions: {
        type: 'array',
        items: { type: 'string' },
        description: 'decisions the user must make before this task can be routed; non-empty means DO NOT execute yet',
      },
    },
  }

  const plan = await agent(
    `Read the root CLAUDE.md of C:\\Users\\littekge\\Git\\USS_Tensor_Processing_Unit ` +
    `(section map, routing rules, decision-authority rules). Then decompose the ` +
    `following task into per-section subtasks, one route per affected section.\n\n` +
    `TASK: ${input.task}\n\n` +
    `Rules:\n` +
    `- Agents: tpu-rtl (System/Tensor_Processing_Unit), assembler (System/Assembler), ` +
    `comms (System/Communication), nn-trainer (System/Neural_Networks, propose-only), ` +
    `spec-proofreader (Specifications/, read-only reporting).\n` +
    `- Tests/ is excluded for all agents. Specifications/ is never edited by anyone.\n` +
    `- Each subtask must be self-contained: the receiving agent sees ONLY your subtask ` +
    `text plus its own agent definition, not the original task or the other routes.\n` +
    `- Include cross-section interface details (formats, signal names, spec references) ` +
    `in every subtask that needs them.\n` +
    `- If the task involves a decision that is even somewhat large (architecture, ` +
    `interfaces, plan changes, conventions) and the task text does not already decide it, ` +
    `put it in openQuestions instead of guessing. When unsure whether a decision ` +
    `qualifies: it does.\n` +
    `- A single-section task is fine: return one route.`,
    { label: 'route', phase: 'Route', schema: ROUTE_SCHEMA }
  )

  if (!plan) throw new Error('routing agent returned no plan')
  log(`route: ${plan.routes.length} section(s), ${plan.openQuestions.length} open question(s)`)
  return { mode: 'plan', task: input.task, plan }
}

// ---------- Step 2: execute (args = {routes}) ---------- //

if (input && input.routes) {
  phase('Execute')
  const EXEC_SCHEMA = {
    type: 'object',
    required: ['summary', 'filesTouched', 'commits', 'pendingVerification', 'openDecisions'],
    properties: {
      summary: { type: 'string', description: 'what was accomplished, brief' },
      details: { type: 'string', description: 'full report body (findings, diffs, test results)' },
      filesTouched: { type: 'array', items: { type: 'string' } },
      commits: {
        type: 'array',
        items: { type: 'string' },
        description: 'branch name + SHA + subject for each commit made (empty if none)',
      },
      pendingVerification: {
        type: 'array',
        items: { type: 'string' },
        description: 'tests written but not run (e.g. Questa unavailable on this machine)',
      },
      openDecisions: {
        type: 'array',
        items: { type: 'string' },
        description: 'somewhat-large decisions surfaced for the user instead of made',
      },
    },
  }

  const results = await parallel(input.routes.map((r) => () => {
    const editing = AGENTS[r.agent] ? AGENTS[r.agent].editing : false
    return agent(
      `${r.subtask}\n\n` +
      `Reporting requirements: fill the structured report completely. If you work in ` +
      `a git worktree, list every commit as "branch sha subject". Record any test you ` +
      `wrote but could not run under pendingVerification. Any somewhat-large decision ` +
      `you encountered goes under openDecisions, unmade.`,
      {
        agentType: r.agent,
        label: r.section,
        phase: 'Execute',
        schema: EXEC_SCHEMA,
        ...(editing ? { isolation: 'worktree' } : {}),
      }
    ).then((res) => ({ section: r.section, agent: r.agent, report: res }))
  }))

  const done = results.filter(Boolean)
  if (done.length < input.routes.length) {
    log(`warning: ${input.routes.length - done.length} agent(s) skipped or failed`)
  }
  const decisions = done.flatMap((x) => (x.report && x.report.openDecisions) || [])
  const pending = done.flatMap((x) => (x.report && x.report.pendingVerification) || [])
  log(`execute: ${done.length}/${input.routes.length} sections done, ` +
      `${decisions.length} open decision(s), ${pending.length} pending verification(s)`)
  return {
    mode: 'executed',
    sections: done,
    failedSections: input.routes.length - done.length,
    openDecisions: decisions,
    pendingVerification: pending,
  }
}

throw new Error('fan-out needs args {task: "..."} (plan) or {routes: [...]} (execute)')
