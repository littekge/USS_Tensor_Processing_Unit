# Neural Network Assembler — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-07-06 — v0.3: Per-layer requantization & bias removal

- Implemented the full v0.3 build plan (assembler side of the system-wide v0.3 quantization upgrade). Work done on branch `v0.3-assember`.
- **Step 1-2 — `Process_Weights.py`:** replaced Q0.7 with per-tensor symmetric int8 (`w_q = clamp(round(w / S_w), -127, 127)`, reading `S_w` from the npz `__scales__<name>` array so weights stay consistent with the exported `M`). Added `REQUANT_MANTISSA_BITS = 8` and `decompose_multiplier(M) → (M0, n)` (dyadic `M0 · 2^-n`) with mantissa-overflow renormalization and range asserts (`128 ≤ M0 ≤ 255`, `0 ≤ n ≤ 255`, `M > 0`). Biases are still quantized/mapped (self-scale) so addresses stay stable; they are now tagged `kind: "bias"` in `weight_map.json` and carry no `M0`/`n`. Weight entries gain `M0`/`n`.
- **Step 3 — `MLIR/dialect.py`:** added optional `M0`/`n` attributes to `MultOp`; serialize/parse round-trip them as a `{M0 = .., n = ..}` attribute dict.
- **Step 4 — `MLIR/legalize.py` + `Process_MLIR.py`:** `legalize_text`/`legalize` now accept `weight_map`; each `dot_general → tpu.mult` is annotated with its weight operand's `M0`/`n` (whichever operand is the `kind: "weight"` entry). `Process_MLIR` loads `weight_map.json` and passes it in.
- **Step 5 — `MLIR/bias_removal.py` (new):** `remove_bias_adds(program, weight_map)` drops `AddOp`s with a `*.bias` operand (`kind: "bias"`), reroutes consumers to the non-bias (matmul) operand, and leaves non-bias adds intact. `Process_MLIR` runs it after legalization and writes `/tmp/optimized.nobias.tpu.mlir`.
- **Step 6 — `Assembler.py`:** `encode_mult` now places `M0` in bits 59-52 and `n` in bits 51-44 (per `Functional_TPU_ISA.md` MUL format) with field-range asserts; `assemble_program` requires each mult to carry `M0`/`n`. `_resolve_address` now maps both `weight` and `bias` kinds to their memory address. `Assemble()` consumes the bias-removed dialect (`optimized.nobias.tpu.mlir`).
- **Step 7 — Verification:** all 26 pytest tests pass (was 19). Manually ran the pipeline on a synthesized 2-layer network: confirmed `weight_map.json` carries `M0`/`n` on weights only, `optimized.nobias.tpu.mlir` drops both bias adds and reroutes consumers, and the emitted program is 2 MULs + END (no ADD) with `M0`/`n` bit-fields decoding correctly.
- Docs: marked all v0.3 steps ✅ in `main.md` and refreshed its Current State; updated `MLIR/README.md` (mult grammar with `M0`/`n`, new `bias_removal.py`) and the dialect header comment to v0.3.
- Files created: `nn_assembler/MLIR/bias_removal.py`. Files modified: `nn_assembler/Process_Weights.py`, `nn_assembler/MLIR/dialect.py`, `nn_assembler/MLIR/legalize.py`, `nn_assembler/Process_MLIR.py`, `nn_assembler/Assembler.py`, `nn_assembler/MLIR/README.md`, `main.md`, `log.md`, and all four `test/test_*.py` covering the above.
- Tests added/updated: per-tensor int8 quant + clamp; `M → (M0, n)` decomposition incl. renormalization and dead/degenerate-layer asserts; `weight_map.json` kinds + `M0`/`n`; MEM byte values; `MultOp` `M0`/`n` round-trip; legalization annotation; bias-removal drop/reroute + non-bias-add preservation; MUL `M0`/`n` bit-fields; no-ADD program; and the full pipeline (bias-removed, 3 instructions, framed transmission).
- Issue / open item: the checked-in `Neural_Networks/Tiny_NN/Tiny_NN_Recent.weights.npz` predates the v0.3 export and lacks `__M__*`/`__scales__*`; the pytest E2E synthesizes its own npz with that metadata, so the suite fully exercises the new paths. The manual real-file run (`Convert.py Tiny_NN Recent`) is pending regeneration of that npz (being handled separately).

- Implemented the v0.1.1 build plan: aligned the code with `Functional_TPU_Message_Protocol_v0.2.md`, which renamed the `START` header code to `FLASH` (ASCII `U` / `0x55` unchanged) and regrouped function codes into header/command/trailer categories. The protocol is functionally identical, so this is a rename + cosmetic-structure change with no behavioral effect.
- Clarification resolved: `main.md`'s v0.1.1 section originally said the *PROGRAM* code was renamed — that was an error. The spec changelog shows it was *START* → *FLASH*; `PROGRAM` (`P` / `0x50`) kept its name and was only reclassified as a command code. Corrected the `main.md` v0.1.1 wording accordingly. Per user decision, the new `INPUT` header code (`I` / `0x49`) is out of scope for this version.
- `Protocol.py`: renamed `START` → `FLASH`; grouped constants into header/command/trailer sections with comments mirroring the spec; fixed the stale `Functional_TPU_Message_Protocol_v0.1.md` docstring reference → `v0.2`.
- `Serializer.py`: updated import and transmission-framing to use `FLASH`; updated docstring.
- `test/test_serializer_and_e2e.py`: updated import and the two framing assertions (`data[0] == FLASH`).
- Files modified: `nn_assembler/Protocol.py`, `nn_assembler/Serializer.py`, `test/test_serializer_and_e2e.py`, `main.md`, `log.md`.
- Tests: all 19 pytest tests pass (including `test_full_pipeline`). Verified `python ./nn_assembler/Convert.py Tiny_NN Recent` still writes `out/TRANSMISSION.bin` (105 bytes), framed with `0x55` ('U'/FLASH) first and `0x53` ('S'/STOP) last.

## 2026-06-24 — Renamed `src/` → `nn_assembler/` and switched to strict editable install (IDE resolution fix)

- Reason: `from nn_assembler import Convert` worked at runtime but showed as *unresolved* in the IDE. Root cause: setuptools 81 defaults to the "lenient" editable install, which registers a dynamic `MetaPathFinder` (`__editable___..._finder.py`) instead of putting a path in the `.pth`. Pylance/static analyzers can't execute that finder, so they can't locate the package.
- Renamed the package directory `src/` → `nn_assembler/` (via `git mv`) so the on-disk name matches the import name, and dropped the now-unnecessary `[tool.setuptools.package-dir]` remap from `pyproject.toml`. (`packages = ["nn_assembler", "nn_assembler.MLIR"]` unchanged.)
- The actual resolution fix: install with `pip install -e . --config-settings editable_mode=compat`. Compat mode writes a **plain static path** (the project root, which contains `nn_assembler/`) into the `.pth`, with no finder hook and **no files added to the project tree** — so Pylance resolves the package while edits stay live. (Strict mode also fixes resolution but materializes a `build/` proxy tree inside the project, which contradicts the `/tmp/` + `/out/` layout in `main.md`; compat avoids that.) A plain `pip install -e .` still re-introduces the finder hook.
- Updated the direct-run bootstrap in `nn_assembler/Convert.py` (`from src.Convert` → `from nn_assembler.Convert`; comment now says `python ./nn_assembler/Convert.py`).
- Simplified `test/conftest.py` source-tree fallback to `import nn_assembler` (dir name now matches import name; dropped the `import src as nn_assembler` alias).
- Added `System/Assembler/.gitignore` (ignores `*.egg-info/`, `.pytest_cache/`, `__pycache__/`, `build/`).
- Updated `main.md` and `CLAUDE.md`: all `/src/...` paths → `/nn_assembler/...`, run command → `python ./nn_assembler/Convert.py`, and install instructions now specify the `editable_mode=compat` flag with rationale.
- Files modified: `pyproject.toml`, `nn_assembler/Convert.py`, `test/conftest.py`, `main.md`, `CLAUDE.md`, `log.md`. Files created: `.gitignore`. Directory renamed: `src/` → `nn_assembler/`.
- Tests: all 19 pytest tests pass (including `test_full_pipeline`); verified `python ./nn_assembler/Convert.py Tiny_NN Recent` still writes `out/TRANSMISSION.bin`, and that the regenerated `.pth` now contains a static path with no finder hook.
- Note (pre-existing): `nn_assembler.egg-info/` is tracked in git from before this change; left as-is to avoid scope creep, though it is now covered by `.gitignore` for future checkouts.

## 2026-06-24 — Decoupled Neural_Networks lookup from directory depth

- Reason: prep for relocating the project from `System/Synth_Flow/Neural_Network_2_TPU_ISA` to `System/Assembler` (one level shallower). The old `Path(__file__).parent.parent.parent.parent / "Neural_Networks"` walk depended on the exact depth and would have silently broken at runtime after the move.
- Replaced the fixed parent walk with `find_networks_dir()`: checks the `NN_ASSEMBLER_NETWORKS_DIR` env var, then searches upward from the package for a `Neural_Networks` directory.
- `NN_import` gained an optional `nn_dir` argument (explicit override; also used by tests).
- Added `test/test_nn_import.py` (env override, missing-dir error, explicit-dir import, descriptive missing-network error).
- Files modified: `src/Convert.py`, `main.md`. Files created: `test/test_nn_import.py`.
- Tests: 19 pass. Pipeline verified still working from the current location.

## 2026-06-24 — Turned the project into an installable package (`nn_assembler`)

- Reason: relative/flat imports only worked when running from inside `src/`, which broke importing `Convert` from another file.
- Added `pyproject.toml` mapping the `src/` directory to the importable package name **`nn_assembler`** (`tool.setuptools.package-dir`), so the documented `/src/...` layout is unchanged — no files were moved. Defines deps (`numpy`) and a `nn-assemble` console script (`nn_assembler.Convert:main`).
- Converted intra-project imports from flat (`from Process_MLIR import ...`) to package-relative (`from .Process_MLIR import ...`, `from .MLIR.legalize import ...`, `from .Protocol import ...`) across `Convert.py`, `Process_MLIR.py`, `Process_Weights.py`, `Assembler.py`, `Serializer.py`. (`MLIR/legalize.py` already used relative imports.)
- Added `src/__init__.py` (re-exports `Convert`, `NN_import`, `main`; sets `__version__`) and `src/__main__.py` (enables `python -m nn_assembler`).
- Refactored `Convert.py`'s `__main__` block into a `main()` function (used by the console script and `-m`), and added a small bootstrap so the original `python ./src/Convert.py <model> <run>` still works despite the relative imports.
- Updated `test/conftest.py` to import the installed package, with a fallback that aliases the source tree as `nn_assembler` when it is not installed; updated all test imports to `nn_assembler.*`.
- Added `.gitignore` for build/packaging artifacts (`*.egg-info/`, `__pycache__/`, `.pytest_cache/`, etc.).
- Verified four usage paths all produce `out/TRANSMISSION.bin`: `from nn_assembler import Convert` (run from `/tmp`), `python -m nn_assembler`, `nn-assemble`, and `python ./src/Convert.py`. All 15 pytest tests pass.
- Files created: `pyproject.toml`, `.gitignore`, `src/__init__.py`, `src/__main__.py`.
- Files modified: `src/Convert.py`, `src/Process_MLIR.py`, `src/Process_Weights.py`, `src/Assembler.py`, `src/Serializer.py`, `test/conftest.py`, all `test/test_*.py`, `main.md`, `CLAUDE.md`.
- Install step: `pip install -e .` from the project root (one-time, into the shared venv).

## 2026-06-24 — Completed v0.1 (build steps 1-8)

- Implemented the full v0.1 lowering pipeline end-to-end; `python ./src/Convert.py Tiny_NN Recent` now produces `/out/TRANSMISSION.bin`.
- **Custom dialect (Step 4):** built the *Functional TPU* dialect in `/src/MLIR/` in **Python** (Step 4 grants free reign over the language). `dialect.py` defines op classes (`MultOp`, `AddOp`, `ReluOp`, `EndOp`, `ReturnOp`) plus textual serialize/parse; grammar and the rationale for choosing Python over TableGen/C++ are documented in `/src/MLIR/README.md`.
- **Legalization (Step 5):** `MLIR/legalize.py` lowers StableHLO → TPU dialect, folding shape-only `reshape` ops so every dialect op maps 1:1 to an ISA instruction (no synthesized-away ops). Wired into `Process_MLIR.py` as the second pass; writes `/tmp/optimized.tpu.mlir`.
- **Weight processing (Steps 1-3):** `Process_Weights.py` reads `weights.npz`, quantizes f32 → Q0.7 (round-to-nearest with clamping), classifies args via their `states[i]`/`inputs[...]` loc labels, maps the input to `0x1` and weights contiguously from `0x2`, and writes `/tmp/MEM.bin` and `/tmp/weight_map.json`.
- **Assembler (Steps 6-7):** `Assembler.py` resolves SSA names to addresses (weights from the map; input + all intermediates → scratch `0x1`), encodes MUL/ELEM/ACT/SYSTEM instruction formats to 128-bit values, and exports `/tmp/PROGRAM.bin`.
- **Serializer (Step 8):** `Serializer.py` wraps `MEM.bin` then `PROGRAM.bin` with START/STOP into `/out/TRANSMISSION.bin`.
- **Protocol:** added `Protocol.py` centralizing Message Protocol function codes and MEM/PROGRAM block builders (little-endian).
- Files created: `src/Protocol.py`, `src/Process_Weights.py`, `src/Assembler.py`, `src/Serializer.py`, `src/MLIR/{__init__,dialect,legalize}.py`, `src/MLIR/README.md`, `test/conftest.py`, `test/test_process_weights.py`, `test/test_dialect_and_legalize.py`, `test/test_assembler.py`, `test/test_serializer_and_e2e.py`.
- Files modified: `src/Convert.py` (orchestration + cleanup), `src/Process_MLIR.py` (added legalization pass, `check=True`), `main.md` (marked steps complete, updated Current State).
- Tests: 15 pytest tests added and passing; verified decoded instruction bit-fields match hand-derived values and the transmission is framed with START/STOP.
- Note: `Process_MLIR` still depends on the `stablehlo-opt` binary; the full-pipeline test skips if it is not on PATH.

## 2026-06-24 — Spec cleanup: fixed errors and contradictions in main.md

- Reviewed `main.md` for logical inconsistencies, spelling/grammar errors, and unclear areas.
- Fixed typos: `weighs.npz` → `weights.npz` (×2), `Assember.py` → `Assembler.py`.
- Corrected Specifications path `../../Specifications/` → `../../../Specifications/` (verified against filesystem; now matches `CLAUDE.md`).
- Corrected fixed-point terminology: "7 decimal bits" → "7 fractional bits".
- Fixed protocol terminology: Lowering Pipeline step 4.2 said "START and END function codes"; the protocol defines START/STOP, so changed to "START and STOP".
- Resolved a logical contradiction in transmission payload order: Lowering Pipeline (PROGRAM then MEM) disagreed with Build Step 8 (MEM then PROGRAM). Per user decision, standardized on **MEM then PROGRAM**.
- Cosmetic grammar cleanups: "of computation graph" → "of the computation graph"; "self contained" → "self-contained".
- Files modified: `main.md`.
- No tests affected (documentation-only change).