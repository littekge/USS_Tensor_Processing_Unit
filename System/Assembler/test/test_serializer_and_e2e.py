"""Serializer tests and a full end-to-end pipeline test (build step 8)."""

import numpy as np
import pytest

from nn_assembler.Assembler import MAX_MATMUL_SIZE, OPCODE_ADD, Assemble
from nn_assembler.MLIR.dialect import MultipOp, MultOp, StrideOp, parse_program
from nn_assembler.MLIR.transpose_analysis import analyze_transposes_file
from nn_assembler.Process_MLIR import Process_MLIR
from nn_assembler.Process_Weights import Process_Weights
from nn_assembler.Protocol import FLASH, MEM, PROGRAM, STOP
from nn_assembler.Serializer import Serialize


def test_serialize_wraps_mem_then_program(tmp_path):
    (tmp_path / "MEM.bin").write_bytes(bytes([MEM, 0x02, 0x00, 0x01, 0x00, 0xAB]))
    (tmp_path / "PROGRAM.bin").write_bytes(bytes([PROGRAM]) + (0).to_bytes(16, "little"))

    out_dir = tmp_path / "out"
    result = Serialize(tmp_path, out_dir)
    data = result.read_bytes()

    assert data[0] == FLASH
    assert data[-1] == STOP
    # MEM payload comes before the PROGRAM payload.
    assert data.index(MEM) < data.index(PROGRAM)


def _have_stablehlo_opt():
    import shutil

    return shutil.which("stablehlo-opt") is not None


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
def test_full_pipeline(tmp_path):
    # Recreate the Tiny_NN inputs from the optimized example + a small weights set.
    example = (
        "module @jit_func {\n"
        "  func.func public @main("
        '%arg0: tensor<4x1xf32> loc("states[0]"), '
        '%arg1: tensor<4xf32> loc("states[1]"), '
        '%arg2: tensor<1x4xf32> loc("states[2]"), '
        '%arg3: tensor<1xf32> loc("states[3]"), '
        '%arg4: tensor<1x1xf32> loc("inputs[0][0]")) '
        "-> (tensor<1x1xf32>) {\n"
        "    %0 = stablehlo.reshape %arg0 : (tensor<4x1xf32>) -> tensor<1x4xf32>\n"
        "    %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [1] x [0] : "
        "(tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>\n"
        "    %2 = stablehlo.reshape %arg1 : (tensor<4xf32>) -> tensor<1x4xf32>\n"
        "    %3 = stablehlo.add %2, %1 : tensor<1x4xf32>\n"
        "    %4 = stablehlo.reshape %arg2 : (tensor<1x4xf32>) -> tensor<4x1xf32>\n"
        "    %5 = stablehlo.dot_general %3, %4, contracting_dims = [1] x [0] : "
        "(tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<1x1xf32>\n"
        "    %6 = stablehlo.reshape %arg3 : (tensor<1xf32>) -> tensor<1x1xf32>\n"
        "    %7 = stablehlo.add %6, %5 : tensor<1x1xf32>\n"
        "    return %7 : tensor<1x1xf32>\n"
        "  }\n"
        "}\n"
    )
    (tmp_path / "initial.mlir").write_text(example)
    S_w = 1.0 / 127
    np.savez(
        tmp_path / "weights.npz",
        **{
            "hidden.weight": np.zeros((4, 1), dtype=np.float32),
            "hidden.bias": np.zeros((4,), dtype=np.float32),
            "output.weight": np.zeros((1, 4), dtype=np.float32),
            "output.bias": np.zeros((1,), dtype=np.float32),
            "__order__": np.array(["hidden.weight", "hidden.bias", "output.weight", "output.bias"]),
            # v0.3 requant metadata: weights carry M (biases do not).
            "__M__hidden.weight": np.float64(0.75),
            "__scales__hidden.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
            "__M__output.weight": np.float64(0.5),
            "__scales__output.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
        },
    )

    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    # Bias adds are dropped: two mults + end remain.
    assert len(instructions) == 3

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    # No ADD opcodes survive, and each mult carries its layer's M0/n (v0.4: M0 at
    # bits 35-28, n at bits 27-20).
    assert all(_bits(instr, 127, 124) != OPCODE_ADD for instr in instructions)
    assert (_bits(instructions[0], 35, 28), _bits(instructions[0], 27, 20)) == (192, 8)  # M=0.75
    assert (_bits(instructions[1], 35, 28), _bits(instructions[1], 27, 20)) == (128, 8)  # M=0.5

    # The final output (second mult's result) is pinned to the I/O address 0x1.
    assert _bits(instructions[1], 59, 36) == 1

    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP


# A Bigger_NN-shaped network (three Linear layers) with a genuinely 2-D transposed
# middle weight and a matmul (1x12 @ 12x10) that tiles in BOTH K and N. Mirrors the
# torchax addmm export pattern so stablehlo-opt reproduces the real transpose fold.
_BIGGER = (
    "module @jit_func {\n"
    "  func.func public @main("
    '%arg0: tensor<12x1xf32> loc("states[0]"), '
    '%arg1: tensor<12xf32> loc("states[1]"), '
    '%arg2: tensor<10x12xf32> loc("states[2]"), '
    '%arg3: tensor<10xf32> loc("states[3]"), '
    '%arg4: tensor<1x10xf32> loc("states[4]"), '
    '%arg5: tensor<1xf32> loc("states[5]"), '
    '%arg6: tensor<1x1xf32> loc("inputs[0][0]")) -> (tensor<1x1xf32>) {\n'
    "    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>\n"
    "    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<12x1xf32>) -> tensor<1x12xf32>\n"
    "    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<12xf32>\n"
    "    %2 = stablehlo.multiply %arg1, %1 : tensor<12xf32>\n"
    "    %3 = stablehlo.dot_general %arg6, %0, contracting_dims = [1] x [0] : "
    "(tensor<1x1xf32>, tensor<1x12xf32>) -> tensor<1x12xf32>\n"
    "    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x12xf32>\n"
    "    %5 = stablehlo.multiply %4, %3 : tensor<1x12xf32>\n"
    "    %6 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<12xf32>) -> tensor<1x12xf32>\n"
    "    %7 = stablehlo.add %6, %5 : tensor<1x12xf32>\n"
    "    %8 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<10x12xf32>) -> tensor<12x10xf32>\n"
    "    %9 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10xf32>\n"
    "    %10 = stablehlo.multiply %arg3, %9 : tensor<10xf32>\n"
    "    %11 = stablehlo.dot_general %7, %8, contracting_dims = [1] x [0] : "
    "(tensor<1x12xf32>, tensor<12x10xf32>) -> tensor<1x10xf32>\n"
    "    %12 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x10xf32>\n"
    "    %13 = stablehlo.multiply %12, %11 : tensor<1x10xf32>\n"
    "    %14 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<10xf32>) -> tensor<1x10xf32>\n"
    "    %15 = stablehlo.add %14, %13 : tensor<1x10xf32>\n"
    "    %16 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<1x10xf32>) -> tensor<10x1xf32>\n"
    "    %17 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1xf32>\n"
    "    %18 = stablehlo.multiply %arg5, %17 : tensor<1xf32>\n"
    "    %19 = stablehlo.dot_general %15, %16, contracting_dims = [1] x [0] : "
    "(tensor<1x10xf32>, tensor<10x1xf32>) -> tensor<1x1xf32>\n"
    "    %20 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x1xf32>\n"
    "    %21 = stablehlo.multiply %20, %19 : tensor<1x1xf32>\n"
    "    %22 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>\n"
    "    %23 = stablehlo.add %22, %21 : tensor<1x1xf32>\n"
    "    return %23 : tensor<1x1xf32>\n"
    "  }\n"
    "}\n"
)


def _k_run_covers_contraction(tiles, K, S):
    """A per-output-tile K-run must be (multip x (k-1), mult) covering all of K."""
    k_offsets = list(range(0, K, S))
    for start in range(0, len(tiles), len(k_offsets)):
        run = tiles[start : start + len(k_offsets)]
        assert [t.lhs.offset for t in run] == k_offsets  # in-order K walk
        assert all(isinstance(t, MultipOp) for t in run[:-1])
        assert isinstance(run[-1], MultOp)  # terminating mult


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
def test_full_pipeline_bigger_nn_tiled(tmp_path):
    (tmp_path / "initial.mlir").write_text(_BIGGER)
    S_w = 1.0 / 127
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "weights.npz",
        **{
            "hidden1.weight": rng.standard_normal((12, 1)).astype(np.float32),
            "hidden1.bias": np.zeros((12,), dtype=np.float32),
            "hidden2.weight": rng.standard_normal((10, 12)).astype(np.float32),
            "hidden2.bias": np.zeros((10,), dtype=np.float32),
            "output.weight": rng.standard_normal((1, 10)).astype(np.float32),
            "output.bias": np.zeros((1,), dtype=np.float32),
            "__order__": np.array(
                ["hidden1.weight", "hidden1.bias", "hidden2.weight",
                 "hidden2.bias", "output.weight", "output.bias"]
            ),
            "__M__hidden1.weight": np.float64(0.75),
            "__scales__hidden1.weight": np.array([1.0, S_w, 1.0]),
            "__M__hidden2.weight": np.float64(0.5),
            "__scales__hidden2.weight": np.array([1.0, S_w, 1.0]),
            "__M__output.weight": np.float64(0.375),
            "__scales__output.weight": np.array([1.0, S_w, 1.0]),
        },
    )

    manifest = analyze_transposes_file(tmp_path)
    assert len(manifest["weight_transposes"]) == 3  # all three weights transposed
    assert manifest["runtime_transposes"] == []

    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    # Framed transmission.
    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP

    program = parse_program((tmp_path / "optimized.partitioned.tpu.mlir").read_text())
    strides = [op for op in program.ops if isinstance(op, StrideOp)]
    assert len(strides) == 3  # exactly one stride per matmul

    # The middle matmul (1x12 @ 12x10) tiles in both K and N; its stride is (12, 10).
    both_tiled = [s for s in strides if s.ld1 == 12 and s.ld2 == 10]
    assert len(both_tiled) == 1

    # Collect that matmul's tiles: everything between its stride and the next stride.
    idx = program.ops.index(both_tiled[0])
    tiles = []
    for op in program.ops[idx + 1 :]:
        if isinstance(op, StrideOp):
            break
        if isinstance(op, (MultOp, MultipOp)):
            tiles.append(op)

    S = MAX_MATMUL_SIZE
    K, N = 12, 10
    n_tiles = len(range(0, N, S))  # 2
    k_tiles = len(range(0, K, S))  # 2
    assert len(tiles) == n_tiles * k_tiles  # 4 tiles for the both-tiled matmul

    # Every K-run is a multip-run terminated by a mult, covering the full contraction.
    _k_run_covers_contraction(tiles, K, S)
    # N-tiles cover the full output width via their destination offsets.
    assert sorted({t.dst_offset for t in tiles}) == [0, S]
    # Ragged edges: K remainder 4, N remainder 2 appear as real tile sizes.
    assert {t.lhs.shape[1] for t in tiles} == {S, K - S}
    assert {t.rhs.shape[1] for t in tiles} == {S, N - S}

    # Every tile carries the layer's requant pair, and the final output targets 0x1.
    assert all(t.M0 == 128 and t.n == 8 for t in tiles)  # M=0.5 -> (128, 8)

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    last_mult = next(
        instructions[i]
        for i in range(len(instructions) - 1, -1, -1)
        if _bits(instructions[i], 127, 124) == 0b1000
    )
    assert _bits(last_mult, 59, 36) == 1  # returned result pinned to I/O address 0x1
