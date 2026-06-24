"""Serializer tests and a full end-to-end pipeline test (build step 8)."""

import numpy as np
import pytest

from Assembler import Assemble
from Process_MLIR import Process_MLIR
from Process_Weights import Process_Weights
from Protocol import MEM, PROGRAM, START, STOP
from Serializer import Serialize


def test_serialize_wraps_mem_then_program(tmp_path):
    (tmp_path / "MEM.bin").write_bytes(bytes([MEM, 0x02, 0x00, 0x01, 0x00, 0xAB]))
    (tmp_path / "PROGRAM.bin").write_bytes(bytes([PROGRAM]) + (0).to_bytes(16, "little"))

    out_dir = tmp_path / "out"
    result = Serialize(tmp_path, out_dir)
    data = result.read_bytes()

    assert data[0] == START
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
    np.savez(
        tmp_path / "weights.npz",
        **{
            "hidden.weight": np.zeros((4, 1), dtype=np.float32),
            "hidden.bias": np.zeros((4,), dtype=np.float32),
            "output.weight": np.zeros((1, 4), dtype=np.float32),
            "output.bias": np.zeros((1,), dtype=np.float32),
            "__order__": np.array(["hidden.weight", "hidden.bias", "output.weight", "output.bias"]),
        },
    )

    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    # 4 compute ops + end.
    assert len(instructions) == 5
    data = out_path.read_bytes()
    assert data[0] == START and data[-1] == STOP
