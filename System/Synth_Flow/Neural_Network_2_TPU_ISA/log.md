# Neural Network Assembler — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

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