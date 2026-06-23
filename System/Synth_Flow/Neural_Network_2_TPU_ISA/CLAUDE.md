
# Claude Code — Project Instructions

> These rules apply to every task in this project.
> Claude Code reads this file automatically from the project root.

> **NOTE:** All paths are described relative to the directory in which this file is located unless otherwise specified.

## Project Context

- **Build Spec:** `main.md` — the single source of truth for what to build (includes build plan).
- **Change Log:** `log.md` — append an entry every time you make a change.
- **Specifications:** — located at `../../../Specifications/`
	- **Instruction Set Architecture:** `Functional_TPU_ISA_v0.2.md` — TPU instruction set
	- **Messaging Protocol:** `Functional_TPU_Message_Protocol_v0.1.md` — defines byte format for communicating with the TPU
- **Development Platform:** Debian 13 Linux
- **Languages:** Python 3.13, MLIR, C/C++
- **Virtual Environment:** Stored **outside** this directory at `~/Git/USS_Tensor_Processing_Unit/USS_TPU.venv` to unify environment across multiple projects. Virtual environment is not uploaded to Github (venvs contain hardcoded absolute paths that break across machines). Activate with:
  ```
  source "~/Git/USS_Tensor_Processing_Unit/USS_TPU.venv/bin/activate"
  ```

## Iterative Build Workflow

1. **Before starting work**, read `main.md` to understand the spec and `log.md` to see what's already been done.
2. **Work in small increments** — one module or feature at a time.
3. **After every change**, append a dated entry to `log.md` describing what was added, changed, or fixed.
4. **Build and run unit tests** after every build step in the `test/` directory using pytest: `python -m pytest test/ -v`
5. **Run the program** to verify it works as expected up to the current build step: `python ./src/main.py Tiny_NN Recent`
6. **Commit often** with descriptive messages: `git add . ; git commit -m "feat: ..."`
7. **Mark completed build steps** in `main.md`.

## Coding Standards

### No Global Variables

Do **not** use global variables. Pass state through:
- Constructor parameters (`__init__`)
- A shared context/config object
- Method arguments

Every class should receive its dependencies explicitly. If a module needs config values, pass the config dict or object in — don't import a global.

### Comment Rules

Follow three rules for comments:

- **Rule 1 — Names explain *what***: Choose clear, descriptive names for classes, methods, and variables. If the name is good enough, no comment is needed to explain what it does.
- **Rule 2 — Code explains *how***: The code itself should be readable enough to show how things work. Don't write comments that restate the code.
- **Rule 3 — Comments explain *why***: Only add comments when the reason behind a decision isn't obvious from the code. Explain *why* this approach was chosen, *why* a workaround exists, or *why* a non-obvious value is used.

```python
# BAD — restates what the code does
self.score = 0  # set score to zero

# GOOD — explains why
self.score = 0  # Reset between innings; accumulated score is in game_state.total_runs
```

### Other Conventions

- **Type hints** on all function signatures.
- **Docstrings** on public classes and public methods (one-liner is fine if the name is clear).
- **Imports:** Group as stdlib → third-party → project (`SRC.*`), separated by blank lines.
- **No wildcard imports** (`from module import *`).
- **f-strings** for string formatting (not `.format()` or `%`).
- **pathlib.Path** for file paths, not string concatenation.

## Architecture Rules

- `/mlir_passes/` — you have free reign over this subdirectory. It is a workspace for you alone and I will not modify it.
- `/tmp/` — stores **ALL** intermediate files between build steps (MLIR code after each pass, binary archives, etc.).
- `/out/` — stores the final output of the project.

## Log Format

When appending to `log.md`, use this format:

```markdown
## YYYY-MM-DD — Short Title

- What was done (bullet points)
- Files created or modified
- Tests added or updated
- Any issues encountered
```