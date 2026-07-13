"""Tests for transpose analysis (classification + manifest) and offline transpose."""

import numpy as np

from nn_assembler.MLIR.transpose_analysis import analyze_transposes, analyze_transposes_file
from nn_assembler.Process_Weights import (
    Process_Weights,
    quantize_per_tensor_int8,
)

# A weight transpose (%arg0 is a state/parameter) and a runtime transpose (%arg1
# is the network input). Only the former may be resolved offline.
_MLIR = (
    "module @m {\n"
    '  func.func public @main(%arg0: tensor<2x3xf32> loc("states[0]"), '
    '%arg1: tensor<2x2xf32> loc("inputs[0][0]")) -> (tensor<3x2xf32>) {\n'
    "    %0 = stablehlo.transpose %arg0, dims = [1, 0] : "
    "(tensor<2x3xf32>) -> tensor<3x2xf32> loc(#loc)\n"
    "    %1 = stablehlo.transpose %arg1, dims = [1, 0] : "
    "(tensor<2x2xf32>) -> tensor<2x2xf32> loc(#loc)\n"
    "    return %0 : tensor<3x2xf32>\n"
    "  }\n"
    "}\n"
)


def test_classifies_weight_vs_runtime_transpose():
    manifest = analyze_transposes(_MLIR)
    weight = manifest["weight_transposes"]
    runtime = manifest["runtime_transposes"]

    assert len(weight) == 1 and len(runtime) == 1
    assert weight[0]["operand"] == "%arg0"
    assert weight[0]["perm"] == [1, 0]
    assert weight[0]["in_shape"] == [2, 3]
    assert weight[0]["out_shape"] == [3, 2]
    # The input transpose is flagged (out of scope) rather than resolved offline.
    assert runtime[0]["operand"] == "%arg1"


def test_manifest_written_to_tmp(tmp_path):
    (tmp_path / "initial.mlir").write_text(_MLIR)
    manifest = analyze_transposes_file(tmp_path)
    assert (tmp_path / "transpose_manifest.json").is_file()
    assert manifest["weight_transposes"][0]["operand"] == "%arg0"


def _mem_payload(mem_bytes: bytes) -> bytes:
    """Extract the DATA of the (single) MEM command: M, 3-byte addr, 2-byte len, data."""
    assert mem_bytes[0] == ord("M")
    length = int.from_bytes(mem_bytes[4:6], "little")
    return mem_bytes[6 : 6 + length]


def test_transposed_weight_stored_row_major(tmp_path):
    # A 2x3 weight fed through a transpose must be stored as its 3x2 transpose,
    # row-major, in MEM.bin.
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # 2x3
    S_w = float(np.abs(w).max()) / 127.0
    np.savez(
        tmp_path / "weights.npz",
        **{
            "w0": w,
            "__order__": np.array(["w0"]),
            "__M__w0": np.float64(0.5),
            "__scales__w0": np.array([1.0, S_w, 1.0], dtype=np.float64),
        },
    )
    (tmp_path / "initial.mlir").write_text(_MLIR)
    analyze_transposes_file(tmp_path)
    weight_map = Process_Weights(tmp_path)

    # The weight_map records the transposed (3x2) shape for the arg.
    assert weight_map["%arg0"]["kind"] == "weight"
    assert weight_map["%arg0"]["shape"] == [3, 2]

    stored = _mem_payload((tmp_path / "MEM.bin").read_bytes())
    expected = quantize_per_tensor_int8(w.T.reshape(-1), S_w).astype(np.uint8).tobytes()
    assert stored == expected
