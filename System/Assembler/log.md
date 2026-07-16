# Neural Network Assembler — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-07-16 — v0.6 Step 6: end-to-end LeNet-5

- Drove the full pipeline (`NN_import` -> transpose analysis -> `Process_Weights`
  -> `Process_MLIR` -> `Assemble` -> `Serialize`) on the real re-exported
  `LeNet_5_Recent.{mlir,weights.npz}` artifact (already carries per-layer
  `__M__`/`__scales__` for conv1/conv2/fc1-3). CLI (`python -m nn_assembler LeNet_5
  Recent`) writes an 83015-byte framed `out/TRANSMISSION.bin` (FLASH 'U' ... STOP 'S').
- Verified in `test/test_serializer_and_e2e.py::test_full_pipeline_lenet5_real`:
  framed transmission; exactly 4 deduped `window`s with correct geometry
  (conv1 1/28/28->26/26 3x3 s1; pool1 6/26/26->13/13 2x2 s2; conv2 6/13/13->11/11
  3x3 s1; pool2 16/11/11->5/5 2x2 s2); im2col columns channel-major `[9,676]` and
  `[54,121]` (K = chans*kh*kw); per-channel pool outputs `[6,13,13]`, `[16,5,5]`;
  4 ReLUs; CHW intermediates chaining with no reshape (input -> conv1 im2col; relu
  -> pool; pool -> conv2 im2col; relu -> pool2); one `stride` per matmul
  `[(9,676),(54,121),(400,120),(120,84),(84,10)]` = `(K,N)`; conv1 tiled in K
  (9->8+1) and N (676->84*8+4) with weight `[out_ch,K]` as LHS and im2col columns
  as RHS, correct sub-block bases (`lhs=ki`, `rhs=ki*N+ni`, `dst=ni`) and ragged
  edges; `M0/n` on every conv1/conv2 tile equal to the exported multiplier's dyadic
  decomposition; FC3 output `[1,10]` tiled in N with base pinned to I/O 0x1
  (tiles at 0x1 + {0,8}).
- Files modified: `test/test_serializer_and_e2e.py`, `main.md`, `log.md`.
- Tests: 79 pass (incl. new LeNet-5 E2E), 1 pre-existing fail
  (`test_full_pipeline_bigger_nn_real`, Bigger_NN artifact metadata mismatch,
  M0/n != (172,19)) documented since 2026-07-15 and unrelated to v0.6 — reproduces
  identically on the v0.6 base commit.

## 2026-07-16 — v0.6 Step 5: feature-map & im2col memory allocation

- No new allocation code was needed: the Step 2 generalization of
  `_result_words`/`_operand_names` to any-order tensors plus the existing
  `IntermediateAllocator` (free-list with reuse) already place the im2col `[K, N]`
  column scratch and the 3-D CHW conv/pool outputs in main data memory (0x2+, i.e.
  past the weight region), reuse freed regions, and pin the network output to 0x1.
  Confirmed on the real LeNet-5 run: im2col scratch and CHW intermediates all land
  past the weight region (first free = 60076), the conv1 ReLU output reuses the
  freed im2col scratch, and the final output base is 0x1. Conv/pool outputs are
  CHW-planar by construction (conv → `[out_ch, outh*outw]`; pool → `[C, oh, ow]`),
  so each feeds the next window op and the FC flatten with no reshape.
- Files modified: `test/test_assembler.py`, `main.md`, `log.md`.
- Tests: added an assemble-level test driving a conv→relu→pool→relu dialect chain,
  asserting im2col scratch + CHW pool output in main memory, freed-region reuse,
  and the final output at 0x1. 78 pass, 1 pre-existing fail (Bigger_NN artifact).

## 2026-07-16 — v0.6 Step 4: conv/pool legalization + ReLU + hardening

- Rewrote `nn_assembler/MLIR/legalize.py` to lower the LeNet-5 subset while staying
  backward-compatible with the linear-net path:
  - Extracts and walks only the `@main` body via paren/brace matching
    (`_find_functions`); classifies private ReLU helpers (`maximum(x,0)`) but does
    not lower their bodies.
  - `stablehlo.convolution` → `WindowOp` + `Im2colOp` + `MultOp`. Descriptor
    computed from the conv geometry (NCHW/OIHW, `window = {}` = unit stride/no pad).
    im2col columns are `[K, N]` with K channel-major (`K = chans*kh*kw`,
    `N = outh*outw`); the conv weight is the matmul LHS `[out_ch, K]` (no offline
    transpose). MultOp annotated with the layer M0/n.
  - `stablehlo.reduce_window` (max body) → `WindowOp` + `MaxPoolOp`; multi-line op
    collapsed to a single placeholder first (`_collapse_reduce_windows`), asserting
    max reduction and no padding/dilation.
  - `call @relu*` → `ReluOp` (length = flattened element count). ReLU is now
    emitted (the encoder/dialect existed but the legalizer never produced it).
  - `stablehlo.broadcast_in_dim` folded like reshape (only feeds the bias path).
  - Emits a `window` only when the descriptor differs from the live one.
  - Hardening: any unrecognized `stablehlo.*`/`call @` op in `@main` raises an
    assert instead of being silently dropped.
- `nn_assembler/MLIR/bias_removal.py`: reroute `Im2colOp`/`MaxPoolOp` `src` (not
  just ReluOp) so a windowed op consuming a dropped bias-add result reroutes to the
  surviving matmul result.
- `Process_MLIR.py` unchanged (legalize signature unchanged).
- Manual full-pipeline run on the real re-exported `LeNet_5_Recent` artifact
  produced a framed `out/TRANSMISSION.bin`. Legalized dialect: 4 deduped windows,
  2 im2col, 2 max, 5 mults (conv1/conv2/fc1-3), 4 relu; partitioned: 5 strides
  (one per matmul), conv/fc matmuls tiled, all weights carry M0/n.
- Files modified: `nn_assembler/MLIR/legalize.py`, `nn_assembler/MLIR/bias_removal.py`,
  `nn_assembler/MLIR/README.md`, `test/test_dialect_and_legalize.py`, `main.md`,
  `log.md`.
- Tests: added conv→(window+im2col+mult), reduce_window→(window+max), ReLU-emitted,
  unhandled-op assert, and bias-removal windowed-src reroute. 77 pass, 1 pre-existing
  fail (Bigger_NN artifact; unrelated).

## 2026-07-16 — v0.6 Step 3: conv weight flatten (4-D, channel-major K)

- `nn_assembler/Process_Weights.py`: added `flatten_conv_weight_shape` and applied
  it so a 4-D conv kernel `(out_ch, in_ch, kh, kw)` is recorded as the 2-D matrix
  `(out_ch) x (in_ch*kh*kw)`. The row-major flatten of the PyTorch kernel is
  already channel-major (`k = c*(kh*kw)+ki*kw+kj`) -- the exact order `im2col`
  gathers columns -- so it is a pure reshape with NO permutation and the physical
  MEM bytes are unchanged (only the recorded shape becomes 2-D). Per-tensor int8
  quantization + dyadic M0/n are reused unchanged. Conv weights are the matmul LHS
  `[out_ch, K]` (weights need no offline transpose, unlike the FC path where the
  weight is the RHS `Wᵀ`).
- Files modified: `nn_assembler/Process_Weights.py`, `test/test_process_weights.py`,
  `main.md`, `log.md`.
- Tests: `flatten_conv_weight_shape` unit test; a channel-major numeric check on a
  2x2x2x2 kernel (recorded shape [2,8]; MEM bytes equal the row-major channel-major
  int8 flatten). 72 pass, 1 pre-existing fail (Bigger_NN artifact; unrelated).

## 2026-07-16 — v0.6 Step 2: window / im2col / max encoders + dispatch

- `nn_assembler/Assembler.py`: added opcodes (WINDOW=1110, IM2COL/SHAPE=1101,
  MAX/POOL=1100) and encoders. `encode_window` packs the 11 W-Format fields at
  their ISA bit positions (inh 123-112, inw 111-100, chans 99-88, outh 87-76,
  outw 75-64, winh 63-60, winw 59-56, strh 55-52, strw 51-48, padh 47-44, padw
  43-40; five 12-bit, six 4-bit). `encode_im2col`/`encode_max` share an A-Format
  helper (rs1 123-100 = src, rd 99-76 = dest, aux 75-60 = 0). Added dispatch
  branches in `assemble_program`: WindowOp emits config (no result); Im2colOp /
  MaxPoolOp allocate their result via the existing IntermediateAllocator and free
  dead sources.
- Generalized `_result_words`/`_operand_names` to any-order tensors (windowed
  results are CHW/[K,N]) via a local `_prod`; added Im2colOp/MaxPoolOp to the
  `last_use` liveness scan (MultipOp still excluded so tiled runs are not freed
  early).
- Files modified: `nn_assembler/Assembler.py`, `test/test_assembler.py`, `main.md`,
  `log.md`.
- Tests: added window (conv + pool geometry), im2col, and max bit-field tests. 70
  pass, 1 pre-existing fail (`test_full_pipeline_bigger_nn_real`, Bigger_NN
  artifact metadata; unrelated).

## 2026-07-16 — v0.6 Step 1: dialect window / im2col / max ops

- Added three dialect ops to `nn_assembler/MLIR/dialect.py`, each 1:1 with an ISA
  v0.6 instruction: `WindowOp` (11 window-descriptor fields, WINCONFIG), `Im2colOp`
  (src Operand + `[K, N]` out_shape, SHAPE), `MaxPoolOp` (src Operand + CHW
  out_shape, POOL). Added serialize branches and parse regexes (`_WINDOW_RE`,
  `_IM2COL_RE`, `_MAX_RE`) so each round-trips through the textual `.tpu.mlir`.
  Textual forms: `tpu.window chans = .., inh = .., ...`; `%r = tpu.im2col %src ->
  KxN`; `%r = tpu.max %src -> CxOHxOW`.
- Files modified: `nn_assembler/MLIR/dialect.py`, `test/test_dialect_and_legalize.py`,
  `main.md`, `log.md`.
- Tests: added window/im2col/max serialize-parse round-trip tests. 66 pass, 1 fail
  (`test_full_pipeline_bigger_nn_real`) — the failure is pre-existing (Bigger_NN
  artifact requant-metadata mismatch, M0/n != (172,19)), independent of v0.6.
- Run pytest with `PYTHONPATH` pointed at this worktree (the shared venv editable
  install resolves `nn_assembler` to the main working copy).

## 2026-07-15 — Docs: correct bias-removal rationale

- Fixed the `MLIR/bias_removal.py` module docstring, which wrongly stated biases
  "were validated as accuracy-neutral for the target networks." Recorded the
  actual rationale: bias removal was validated on LeNet-5 (<1% accuracy drop,
  acceptable). Tiny_NN and Bigger_NN do lose significant accuracy without biases,
  but they are functionless test/benchmark networks (Tiny_NN a regression sanity
  check; Bigger_NN exercises matrix partitioning), so their accuracy is
  irrelevant to the decision. Kept the accurate points (bias tensors stay
  resident in weight memory via Process_Weights; no longer referenced in the
  instruction stream; consumers rerouted to the non-bias/matmul operand).
- Docstring text only — no code logic changed.
- Files modified: `nn_assembler/MLIR/bias_removal.py`, `log.md`.
- Tests: 63 pass, 1 fail (`test_full_pipeline_bigger_nn_real`); the failure is
  pre-existing (Bigger_NN artifact requant-metadata mismatch, M0/n != (172,19))
  and independent of this docstring-only change — confirmed by reproducing it on
  the unmodified tree.

## 2026-07-13 — v0.5 review fixes: stale-stride reset + real Bigger_NN e2e

- **FIX 1 — pass-through matmul stride reset (user-approved rule).** `ld1`/`ld2`
  persist across instructions until the next `stride`, so an in-array
  (pass-through) matmul — emitted as a bare `mult` with no `stride` — could inherit
  stale leading dimensions from a preceding partitioned matmul. Per the approved
  rule ("an in-array matmul following a partitioned matmul should always reset
  stride to (0,0)"), the partition pass now emits a `stride(0,0)` (contiguous)
  reset immediately before every pass-through `mult`, guaranteeing it never runs
  with stale strides regardless of what precedes it.
  - `nn_assembler/MLIR/partition.py`: `partition_program` prepends `StrideOp(0,0)`
    to each in-array `MultOp`; module + function docstrings document the guarantee.
  - `main.md` Step 5: added the reset behavior (user-approved edit).
  - Tests: `test_partition.py` — renamed the pass-through case to assert the
    `stride(0,0)` reset immediately precedes the `mult`; added
    `test_pass_through_matmul_after_partitioned_resets_stride` (partitioned matmul
    then in-array matmul → strides `(K,N)` then `(0,0)`, reset directly before the
    small mult). `test_assembler.py` — pass-through case now asserts the reset
    assembles to a `stride` instruction with `ld1=ld2=0`. `test_serializer_and_e2e.py`
    Tiny_NN case updated for the two extra stride resets (5 instrs: 2 stride + 2
    mult + end), opcode-filtered.
- **FIX 2 — regenerated real Bigger_NN artifact + real e2e (new artifact-regen
  rule).** The checked-in `Bigger_NN_Recent.weights.npz` predated the v0.3 export
  (only `__order__`), forcing a synthesized-metadata workaround. Re-ran the
  torchax export (`Start.py Bigger_NN Recent -e`, loading the existing trained
  `.pth`) to regenerate `Bigger_NN_Recent.{weights.npz,mlir,pt2}`; the npz now
  carries real `__M__`/`__scales__` per weight. No `Neural_Networks` code modified
  (artifacts only).
  - `test/test_serializer_and_e2e.py`: replaced the synthetic `_BIGGER` graph +
    synthesized-metadata test with `test_full_pipeline_bigger_nn_real`, which drives
    the real artifact through `NN_import → transpose analysis → Process_Weights →
    Process_MLIR → Assemble → Serialize`. Asserts the npz carries `__M__`/`__scales__`,
    a framed transmission, one `stride` per matmul `[(1,1000),(1000,100),(100,1)]`,
    the middle (both-tiled) matmul's 1625 tiles (13 N × 125 K) with in-order
    `multip`-runs terminated by `mult`, sub-block bases (`lhs=ki`, `rhs=ki*N+ni`,
    `dst=ni`), ragged N edge (remainder 4), `M0`/`n` = (172,19) on every tile, the
    last matmul's ragged K edge (remainder 4), and the final output pinned to 0x1.
  - `main.md` Step 6: updated to describe the real (no-longer-synthesized) run
    (1767 instructions, real strides, ragged edges).
  - Verified the CLI too: `python -m nn_assembler Bigger_NN Recent` writes a
    132254-byte framed `out/TRANSMISSION.bin`.
- Files modified: `nn_assembler/MLIR/partition.py`, `main.md`, `log.md`,
  `test/test_partition.py`, `test/test_assembler.py`,
  `test/test_serializer_and_e2e.py`. Artifact regenerated:
  `System/Neural_Networks/Bigger_NN/Bigger_NN_Recent.{weights.npz,mlir,pt2}`.
- Tests: 64 pytest pass (was 63; +1 net from the new mixed pass-through test). Run
  with `PYTHONPATH` pointed at this worktree so `nn_assembler` resolves here rather
  than the shared editable install.

## 2026-07-13 — v0.5: Matmul partitioning (leading-dimension tiling)

- Implemented the full v0.5 build plan (assembler side). Decomposes any matmul
  exceeding the systolic array into array-sized tiles addressed in place via
  leading dimensions, resolves weight transposes offline, and emits a new
  inspectable dialect artifact. Verified against `Functional_TPU_ISA.md` v0.5
  (read-only): MUL 128-bit layout with `mult`/`multip` sharing it (funct3 MULT=0x0,
  MULTIP=0x1), CONFIG `stride` opcode 1111 with im1/im2 = ld1/ld2.
- **Step 1 — `Assembler.py` + `MLIR/dialect.py`:** added `MAX_MATMUL_SIZE = 8`
  (coupled to TPU Build Parameter N=8) and changed the MUL dim assert from the
  4-bit field max (15) to the tiling bound (`MAX_MATMUL_SIZE`). Dialect gained
  `MultipOp` and `StrideOp`; `Operand` gained an `offset`, and `MultOp`/`MultipOp`
  gained `dst_offset` + `parent_out_shape` for sub-block addressing. Serialize/parse
  round-trip the new ops and tiling attributes (`lo`/`ro`/`do`/`cap`), keeping the
  legacy `{M0 = .., n = ..}` form for un-tiled ops.
- **Step 2 — `Assembler.py`:** `encode_mult` now takes a `funct3`;
  `encode_mult_in_place` (funct3 0x1) and `encode_stride` (opcode 1111, im1/im2 =
  ld1/ld2, 24-bit) added; `assemble_program` dispatches `MultipOp`/`StrideOp` and
  adds each operand's element offset to its resolved base address. A tiled result
  is allocated once (by parent capacity) and shared across its tiles.
- **Step 3 — `MLIR/transpose_analysis.py` (new):** `analyze_transposes_file()`
  scans `initial.mlir`, classifies each `stablehlo.transpose` (operand is a
  `states[...]` parameter → weight transpose, resolved offline; the input or an
  intermediate → runtime transpose, flagged out of scope), and writes
  `/tmp/transpose_manifest.json`. Wired into `Convert` right after import.
- **Step 4 — `Process_Weights.py` + `MLIR/legalize.py`:** `Process_Weights`
  physically transposes manifest-flagged weights before quantization (per-tensor
  symmetric int8 scale is transpose-invariant), stores `Wᵀ` row-major, and records
  the transposed shape. `legalize` folds a resolved weight/constant transpose like
  a reshape and raises on a runtime (live) transpose instead of silently dropping.
- **Step 5 — `MLIR/partition.py` (new):** `partition_program()` tiles each
  oversized `MultOp` into one `stride(ld1=K, ld2=N)` then row-parallel output tiles
  (M-tile × N-tile), each walking K in array steps as `multip × (k-1)` then a
  terminating `mult`. Sub-block corners use `offset = row_off*ld + col_off`; ragged
  edges use real sizes; `M0`/`n` carried on every tile. Ops within the array pass
  through unchanged. Wired into `Process_MLIR` as the final dialect pass, emitting
  `/tmp/optimized.partitioned.tpu.mlir`; `Assemble` now consumes that artifact.
- **Step 6 — Verification:** ran the pipeline on the real `Bigger_NN` graph
  (1→1000→100→1, middle weight a genuine 2-D transpose) with synthesized requant
  metadata — 1767 instructions, one `stride` per matmul (ld1=K/ld2=N), 139 `mult` +
  1624 `multip`, correct sub-block bases (`%1` intermediate at 0x18F3B, final output
  0x1), ragged K/N edges, and `M0`/`n` on every tile. `Convert Tiny_NN Recent` still
  frames a 72-byte `out/TRANSMISSION.bin` (small matmuls pass through, no stride).
- Files created: `nn_assembler/MLIR/partition.py`,
  `nn_assembler/MLIR/transpose_analysis.py`, `test/test_partition.py`,
  `test/test_transpose_analysis.py`. Files modified: `nn_assembler/Assembler.py`,
  `nn_assembler/Convert.py`, `nn_assembler/Process_MLIR.py`,
  `nn_assembler/Process_Weights.py`, `nn_assembler/MLIR/dialect.py`,
  `nn_assembler/MLIR/legalize.py`, `nn_assembler/MLIR/README.md`, `main.md`, `log.md`,
  and `test/test_assembler.py`, `test/test_dialect_and_legalize.py`,
  `test/test_serializer_and_e2e.py`.
- Tests: 63 pytest tests pass (was 41). Added: `multip`/`stride` serialize/parse
  round-trip and un-tiled legacy form; transpose fold + runtime-transpose rejection;
  `multip`/`stride` encoding and MAX-dim rejection; tiled emission (stride count,
  tile count, sub-block bases, ragged edges) + small-matmul pass-through; partition
  K-accumulation order, row-parallel M/N offsets, ragged edges, `MAX_MATMUL_SIZE`
  boundary; transpose classification + manifest + transposed-weight layout numeric
  check; and a full both-tiled Bigger_NN E2E (framing + tiling structure + output
  pinned to 0x1). Run pytest with `PYTHONPATH` pointed at this worktree — the shared
  venv's editable install resolves `nn_assembler` to the main working copy.
- Open items surfaced for the user (not decided):
  1. The checked-in `Bigger_NN_Recent.weights.npz` predates the v0.3 export and
     carries only `__order__` (no `__M__*`/`__scales__*`), so the *real* file cannot
     be assembled (weights fall back to `kind: bias`, no `M0`/`n`). The E2E test and
     the manual Step 6 run synthesize the metadata to exercise tiling; regenerating
     the real npz is a `Neural_Networks`/`nn-trainer` (propose-only) task.
  2. Per the approved plan, an in-array matmul passes through as a single `mult`
     with **no** `stride`. The `ld` registers persist across instructions, so a
     small matmul following a partitioned one would inherit stale leading
     dimensions. Every `Bigger_NN` matmul is partitioned (each emits its own
     stride), so this is latent, not hit today — but a mixed network could need a
     `stride(0,0)` reset before pass-through matmuls. Flagged for a plan decision.

## 2026-07-09 — v0.4: Expanded memory & address widening

- Aligned the assembler with the system-wide v0.4 memory overhaul: 24-bit
  instruction address fields, a 3-byte MEM address, and a unified data memory
  where 0x1 is the reserved network I/O designation and all intermediates live
  in main data memory (0x2+). Matmul partitioning (`MAX_MATMUL_SIZE`) stays
  deferred. Verified against `Specifications/Functional_TPU_ISA.md` v0.4 and
  `Functional_TPU_Message_Protocol.md` v0.3 (both read-only).
- **Step 1 — `Assembler.py`:** rewrote `encode_mult`/`encode_add`/`encode_relu`
  to the v0.4 field layout (MULT: rs1 123-100, sz11 99-96, sz12 95-92, rs2 91-68,
  sz21 67-64, sz22 63-60, rd 59-36, M0 35-28, n 27-20; ELEM add: rs1 123-100,
  rs2 99-76, sz1 75-68, sz2 67-60, rd 59-36; ACT relu: rs1 123-100, rd 99-76,
  len 75-60). M0/n moved from the old 59-52/51-44 positions. Added 24-bit
  address-field asserts; instructions remain 128 bits with reserved bits zeroed.
- **Step 2 — `Protocol.py`:** MEM command now emits a 3-byte address
  (LADD/MADD/UADD, LSB first) then the unchanged 16-bit length (LLEN/ULEN);
  `ADDRESS_SPACE` widened to 2^24. Added `build_mem_blocks(address, data)` which
  chunks tensors longer than 65535 words across consecutive MEM commands with
  incrementing 24-bit addresses (e.g. Bigger_NN's 100000-word layer).
- **Step 3 — `Process_Weights.py` + `Assembler.py`:** added
  `FINAL_OUTPUT_ADDRESS = 0x1`, `first_free_address(weight_map)` (first address
  past the contiguous weight/bias region), and an `IntermediateAllocator` (free
  list with adjacent-region merge; simple scope reuse, no full liveness).
  `Process_Weights._write_mem` now uses `build_mem_blocks`. `assemble_program`
  allocates each intermediate result in main memory, reuses regions once their
  last consuming op has run, and pins the `tpu.return` value to 0x1 for readback.
  Weights/biases still map contiguously from 0x2.
- **Step 4 — verification:** ran `python -m nn_assembler Tiny_NN Recent` (Tiny_NN
  npz now carries v0.3 requant metadata, regenerated 2026-07-07); it produced a
  framed `out/TRANSMISSION.bin`. Decode confirmed: two MULTs carry 24-bit
  addresses with M0/n at 35-28/27-20 and reserved bits zero; `%1` intermediate
  lands at 0xF (main memory, past the 13-word weight region), the final output
  targets 0x1, and the single MEM command uses a 3-byte address (addr 0x2,
  len 13).
- Files modified: `nn_assembler/Assembler.py`, `nn_assembler/Protocol.py`,
  `nn_assembler/Process_Weights.py`, `test/test_assembler.py`,
  `test/test_process_weights.py`, `test/test_serializer_and_e2e.py`, `log.md`.
  Files created: `test/test_protocol.py`.
- Tests: 41 pytest tests pass (was 26). Added: 24-bit MULT/ADD/RELU field
  placement incl. moved M0/n and full-24-bit addresses; assembler intermediate
  placement in main memory, freed-region reuse, and final output at 0x1; 3-byte
  MEM address, over-length/over-address rejection, and multi-command chunking of
  a >65535-word tensor; `IntermediateAllocator` bump/reuse/merge and
  `first_free_address`; 0x1 reserved for I/O with weights never on 0x0/0x1.
- Note: run pytest with `PYTHONPATH` pointed at this tree — the shared venv's
  editable install resolves `nn_assembler` to the main working copy, not a
  worktree.
- Open item: the v0.4 build plan (`### v0.4 — Expanded Memory & Address
  Widening`) is not present in this tree's `main.md`; per section rules the plan
  was not written and no steps were marked complete. Surfaced for user approval.

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