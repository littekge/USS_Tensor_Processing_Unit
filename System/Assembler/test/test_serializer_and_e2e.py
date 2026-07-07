"""Serializer tests and a full end-to-end pipeline test (build step 8)."""

import numpy as np
import pytest

from nn_assembler.Assembler import OPCODE_ADD, Assemble
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

    # No ADD opcodes survive, and each mult carries its layer's M0/n.
    assert all(_bits(instr, 127, 124) != OPCODE_ADD for instr in instructions)
    assert (_bits(instructions[0], 59, 52), _bits(instructions[0], 51, 44)) == (192, 8)  # M=0.75
    assert (_bits(instructions[1], 59, 52), _bits(instructions[1], 51, 44)) == (128, 8)  # M=0.5

    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP
