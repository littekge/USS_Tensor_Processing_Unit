# Functional TPU — Custom MLIR Dialect

This directory contains the custom *Functional TPU* MLIR dialect and the
StableHLO → TPU legalization pass. It is the bridge between StableHLO and
`Functional_TPU_ISA.md` machine code.

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

- `dialect.py` — in-memory op classes (`MultOp`, `MultipOp`, `StrideOp`,
  `WindowOp`, `Im2colOp`, `MaxPoolOp`, `AddOp`, `ReluOp`, `EndOp`, `ReturnOp`) plus
  `serialize()`/`parse_program()` for the textual form.
- `transpose_analysis.py` — post-import pass (`analyze_transposes_file()`) that
  scans `initial.mlir`, classifies each `stablehlo.transpose` as a weight
  transpose (resolved offline) or a runtime transpose (out of scope, flagged),
  and writes `/tmp/transpose_manifest.json`.
- `legalize.py` — the StableHLO → TPU dialect pass (`legalize()`), invoked by
  `Process_MLIR.py`. Annotates each `tpu.mult` with its weight's dyadic
  requantization pair `M0`/`n` from `weight_map.json`, and folds resolved weight
  transposes (their data was transposed offline by `Process_Weights`). v0.6 adds
  the LeNet-5 subset: `stablehlo.convolution` → `window`+`im2col`+`mult`,
  `stablehlo.reduce_window` (max) → `window`+`max`, and `call @relu*` → `relu`. It
  walks only `@main` (private ReLU helpers are classified, not lowered), emits a
  `window` only when the descriptor changes, and asserts on any unhandled op
  rather than silently dropping it.
- `bias_removal.py` — TPU-dialect pass (`remove_bias_adds()`) that drops bias
  `add`s (v0.3: bias is folded into the matmul's requant multiplier) and reroutes
  their consumers to the matmul result. Non-bias adds are left untouched.
- `partition.py` — dialect→dialect pass (`partition_program()`, v0.5) that
  tiles any matmul exceeding `MAX_MATMUL_SIZE` into array-sized `mult`/`multip`
  runs preceded by a `stride`, addressing sub-blocks in place via leading
  dimensions. Matmuls that already fit pass through unchanged.
- `stage_output.py` — final dialect→dialect pass (`stage_output()`, v0.7) that
  inserts a `tpu.move %out -> @io` before the `tpu.return` terminator, staging the
  network output (written to main memory like any intermediate) into the reserved
  I/O address `0x1` for readback. Retires LeNet-5 "Bug 2" (tiled output previously
  spilled past `0x1` into weight memory).

## Dialect grammar

Every `tpu.*` op corresponds to exactly one ISA instruction. Shape-only ops
(reshape/transpose) are **folded** by the legalizer and never appear here, so
nothing is synthesized away during assembly.

```
module @<name> {
  tpu.func @main {
    %res = tpu.mult %a : RxC, %b : RxC -> RxC {M0 = <int>, n = <int>}
                                                // -> ISA mult  (MUL format); M0/n
                                                //    are the dyadic requant pair
    %res = tpu.add  %a : RxC, %b : RxC -> RxC   // -> ISA add   (ELEM format)
    %res = tpu.relu %a : N -> N                 // -> ISA relu  (ACT format)
    tpu.move %res -> @io : RxC                    // -> ISA move  (SHAPE / A-format,
                                                 //    funct3 MOVE); stages the output
                                                 //    into I/O 0x1 (v0.7)
    tpu.return %res : RxC                        // marks the program output
    tpu.end                                      // -> ISA end   (SYSTEM format)
  }
}
```

### Windowed ops (v0.6: convolution and pooling)

A convolution lowers to a `window` (geometry) + `im2col` (feature map → column
matrix) + `mult` (weight · columns); the matmul is tiled by `partition.py` like
any other. Max pooling lowers to `window` + `max`. The `window` descriptor
persists until the next `window`, so the legalizer emits one only when it changes.

```
    tpu.window chans = C, inh = H, inw = W, outh = OH, outw = OW,
               winh = KH, winw = KW, strh = SH, strw = SW, padh = PH, padw = PW
                                                 // -> ISA window (WINCONFIG / W-format)
    %col = tpu.im2col %src -> KxN                // -> ISA im2col (SHAPE / A-format)
                                                 //    K = C*KH*KW (channel-major),
                                                 //    N = OH*OW
    %res = tpu.max    %src -> CxOHxOW            // -> ISA max    (POOL / A-format)
```

`im2col`'s K-axis is channel-major (`k = c*(KH*KW) + ki*KW + kj`); the conv
weight is flattened `(out,in,kh,kw) → (out_ch × in*kh*kw)` in the same order (no
permutation) and used as the matmul LHS. Conv/pool outputs are CHW-planar, so
each feeds the next layer's window op (and the final flatten to the FC vector)
with no reshape.

### Partitioned matmuls (v0.5)

A matmul larger than `MAX_MATMUL_SIZE` is lowered by `partition.py` into a
`stride` plus a set of array-sized tiles. `multip` shares `mult`'s MUL layout and
differs only in `funct3` (it keeps the accumulator); a run of `multip`s
terminated by a `mult` accumulates one contraction (K) tile-by-tile.

```
    tpu.stride ld1 = <K>, ld2 = <N>              // -> ISA stride (CONFIG format)
    %res = tpu.multip %a : RxC, %b : RxC -> RxC {M0=.., n=.., lo=.., ro=.., do=.., cap=..}
    %res = tpu.mult   %a : RxC, %b : RxC -> RxC {M0=.., n=.., lo=.., ro=.., do=.., cap=..}
```

The tiling attributes are element offsets of each sub-block corner within its
parent matrix — `lo`/`ro` for the two sources, `do` for the destination — plus
`cap`, the parent result's capacity in words (so the assembler allocates the
whole result once). They are omitted for un-tiled ops, which keep the plain
`{M0 = .., n = ..}` form. Every tile of a matmul shares one `result` name (the
parent) and one preceding `stride`.

Operand names are SSA values: `%argN` for a mapped weight or the network input,
`%N` for the result of a prior op. The assembler resolves these names to ISA
memory addresses using `/tmp/weight_map.json` (weights) and address `0x1` (the
network input and the final output); intermediate results live in main data
memory. For a tile, the assembler adds the tile's element offset to the parent's
base address.
