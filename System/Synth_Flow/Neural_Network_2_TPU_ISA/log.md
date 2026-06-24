# Neural Network Assembler — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

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