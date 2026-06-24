# Functional TPU — Custom MLIR Dialect

This directory contains the custom *Functional TPU* MLIR dialect and the
StableHLO → TPU legalization pass. It is the bridge between StableHLO and
`Functional_TPU_ISA_v0.2.md` machine code.

## Implementation choice (v0.1): Python, not TableGen/C++

`main.md` Step 4 grants free reign over the language used in this directory.
v0.1 implements the dialect and pass in **Python** rather than as a compiled
TableGen/C++ MLIR dialect. Rationale:

- **Verifiable now.** A real C++ dialect requires a CMake project linked against
  the MLIR libraries plus a loadable pass plugin — hours of build setup that is
  easy to get wrong and hard to test in CI. A Python implementation is fully
  unit-tested today.
- **The contract is what matters.** The deliverable is a textual IR that maps
  1:1 to machine code and a pass that produces it. The grammar below is defined
  precisely so a future version can re-implement it in TableGen/C++ without
  changing the rest of the pipeline.

## Files

- `dialect.py` — in-memory op classes (`MultOp`, `AddOp`, `ReluOp`, `EndOp`,
  `ReturnOp`) plus `serialize()`/`parse_program()` for the textual form.
- `legalize.py` — the StableHLO → TPU dialect pass (`legalize()`), invoked by
  `src/Process_MLIR.py`.

## Dialect grammar

Every `tpu.*` op corresponds to exactly one ISA instruction. Shape-only ops
(reshape/transpose) are **folded** by the legalizer and never appear here, so
nothing is synthesized away during assembly.

```
module @<name> {
  tpu.func @main {
    %res = tpu.mult %a : RxC, %b : RxC -> RxC   // -> ISA mult  (MUL format)
    %res = tpu.add  %a : RxC, %b : RxC -> RxC   // -> ISA add   (ELEM format)
    %res = tpu.relu %a : N -> N                 // -> ISA relu  (ACT format)
    tpu.return %res : RxC                        // marks the program output
    tpu.end                                      // -> ISA end   (SYSTEM format)
  }
}
```

Operand names are SSA values: `%argN` for a mapped weight or the network input,
`%N` for the result of a prior op. The assembler resolves these names to ISA
memory addresses using `/tmp/weight_map.json` (weights) and address `0x1` (the
network input and all intermediate results).
