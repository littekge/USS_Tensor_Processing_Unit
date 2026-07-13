"""Serializer tests and a full end-to-end pipeline test (build step 8)."""

from pathlib import Path

import numpy as np
import pytest

from nn_assembler.Assembler import (
    MAX_MATMUL_SIZE,
    OPCODE_ADD,
    OPCODE_MULT,
    OPCODE_STRIDE,
    Assemble,
)
from nn_assembler.Convert import NN_import
from nn_assembler.MLIR.dialect import MultipOp, MultOp, StrideOp, parse_program
from nn_assembler.MLIR.transpose_analysis import analyze_transposes_file
from nn_assembler.Process_MLIR import Process_MLIR
from nn_assembler.Process_Weights import Process_Weights
from nn_assembler.Protocol import FLASH, MEM, PROGRAM, STOP
from nn_assembler.Serializer import Serialize

# The real, exported Bigger_NN artifacts (three Linear layers, 1->1000->100->1).
# Its middle matmul (1x1000 @ 1000x100) tiles in both K and N, exercising the full
# partitioning path against a genuine torchax export rather than a synthetic graph.
_NETWORKS_DIR = Path(__file__).resolve().parents[2] / "Neural_Networks"
_BIGGER_NN_DIR = _NETWORKS_DIR / "Bigger_NN"
_have_bigger_nn = (_BIGGER_NN_DIR / "Bigger_NN_Recent.mlir").is_file() and (
    _BIGGER_NN_DIR / "Bigger_NN_Recent.weights.npz"
).is_file()


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

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    # Bias adds are dropped. Both matmuls are in-array (pass-through), so each is
    # preceded by a stride(0,0) contiguous reset: 2 strides + 2 mults + end.
    assert len(instructions) == 5
    mults = [i for i in instructions if _bits(i, 127, 124) == OPCODE_MULT]
    strides = [i for i in instructions if _bits(i, 127, 124) == OPCODE_STRIDE]
    assert len(mults) == 2 and len(strides) == 2
    # Every pass-through stride reset is contiguous (ld1 = ld2 = 0).
    assert all(_bits(s, 123, 100) == 0 and _bits(s, 99, 76) == 0 for s in strides)

    # No ADD opcodes survive, and each mult carries its layer's M0/n (v0.4: M0 at
    # bits 35-28, n at bits 27-20).
    assert all(_bits(instr, 127, 124) != OPCODE_ADD for instr in instructions)
    assert (_bits(mults[0], 35, 28), _bits(mults[0], 27, 20)) == (192, 8)  # M=0.75
    assert (_bits(mults[1], 35, 28), _bits(mults[1], 27, 20)) == (128, 8)  # M=0.5

    # The final output (second mult's result) is pinned to the I/O address 0x1.
    assert _bits(mults[1], 59, 36) == 1

    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP


def _k_run_covers_contraction(tiles, K, S):
    """A per-output-tile K-run must be (multip x (k-1), mult) covering all of K."""
    k_offsets = list(range(0, K, S))
    for start in range(0, len(tiles), len(k_offsets)):
        run = tiles[start : start + len(k_offsets)]
        assert [t.lhs.offset for t in run] == k_offsets  # in-order K walk
        assert all(isinstance(t, MultipOp) for t in run[:-1])
        assert isinstance(run[-1], MultOp)  # terminating mult


def _tiles_between_strides(program, stride):
    """All mult/multip tiles emitted for a matmul: from its stride to the next."""
    idx = program.ops.index(stride)
    tiles = []
    for op in program.ops[idx + 1 :]:
        if isinstance(op, StrideOp):
            break
        if isinstance(op, (MultOp, MultipOp)):
            tiles.append(op)
    return tiles


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
@pytest.mark.skipif(not _have_bigger_nn, reason="Bigger_NN_Recent artifacts not present")
def test_full_pipeline_bigger_nn_real(tmp_path):
    """Full pipeline on the REAL exported Bigger_NN (1->1000->100->1) artifact.

    Uses the checked-in, torchax-exported `Bigger_NN_Recent.{mlir,weights.npz}`
    directly -- no synthesized requant metadata. Exercises all three matmuls,
    with the middle one (1x1000 @ 1000x100) tiling in both K and N.
    """
    NN_import("Bigger_NN", "Recent", tmp_dir=tmp_path, nn_dir=_NETWORKS_DIR)

    # The regenerated npz carries the v0.3 requant metadata (__M__/__scales__ per
    # weight); the pre-v0.3 file did not, forcing the earlier synthesized workaround.
    weights = np.load(tmp_path / "weights.npz", allow_pickle=True)
    assert any(k.startswith("__M__") for k in weights.files)
    assert any(k.startswith("__scales__") for k in weights.files)

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
    # One stride per matmul, each (ld1 = K, ld2 = N) for the three layers.
    assert [(s.ld1, s.ld2) for s in strides] == [(1, 1000), (1000, 100), (100, 1)]

    S = MAX_MATMUL_SIZE

    # --- Middle matmul (1x1000 @ 1000x100): tiles in BOTH K and N. ---
    both_tiled = next(s for s in strides if (s.ld1, s.ld2) == (1000, 100))
    tiles = _tiles_between_strides(program, both_tiled)
    K, N = 1000, 100
    n_tiles = len(range(0, N, S))  # 13 (100 -> 12 full + ragged 4)
    k_tiles = len(range(0, K, S))  # 125 (1000 = 125*8 exactly)
    assert len(tiles) == n_tiles * k_tiles  # 1625 tiles

    # Each output tile's K-run accumulates via multip and terminates with a mult.
    _k_run_covers_contraction(tiles, K, S)
    # N-tiles cover the full 100-wide output via their destination offsets.
    assert sorted({t.dst_offset for t in tiles}) == list(range(0, N, S))
    # Sub-block corners: lhs offset = ki (M=1); rhs offset = ki*N + ni.
    assert {t.lhs.offset for t in tiles} == set(range(0, K, S))
    assert {t.rhs.offset for t in tiles} == {ki * N + ni for ki in range(0, K, S) for ni in range(0, N, S)}
    # Ragged edge: N remainder 4 (100 - 12*8) appears; K divides evenly so no K remainder.
    assert {t.rhs.shape[1] for t in tiles} == {S, N - 12 * S}
    assert {t.lhs.shape[1] for t in tiles} == {S}
    # Every tile carries the layer's requant pair (hidden2.weight: M0=172, n=19).
    assert all((t.M0, t.n) == (172, 19) for t in tiles)

    # --- Last matmul (1x100 @ 100x1): ragged K remainder 4 (100 - 12*8). ---
    last_stride = next(s for s in strides if (s.ld1, s.ld2) == (100, 1))
    last_tiles = _tiles_between_strides(program, last_stride)
    _k_run_covers_contraction(last_tiles, 100, S)
    assert {t.lhs.shape[1] for t in last_tiles} == {S, 100 - 12 * S}

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    last_mult = next(
        instructions[i]
        for i in range(len(instructions) - 1, -1, -1)
        if _bits(instructions[i], 127, 124) == 0b1000
    )
    assert _bits(last_mult, 59, 36) == 1  # returned result pinned to I/O address 0x1
