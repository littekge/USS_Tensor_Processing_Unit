"""Tests for per-tensor int8 quantization, dyadic M decomposition, and mapping.

v0.3: weights are quantized per-tensor symmetric int8 using the exported S_w, and
each layer's requantization multiplier M is decomposed into the ISA dyadic pair
(M0, n). Biases are still quantized/mapped (for stable addresses) but carry no
M0/n.
"""

import numpy as np
import pytest

from nn_assembler.Process_Weights import (
    Process_Weights,
    decompose_multiplier,
    quantize_per_tensor_int8,
)
from nn_assembler.Protocol import MEM


def test_quantize_per_tensor_symmetric_int8():
    # scale = absmax/127 = 0.5/127 maps +/-0.5 to the symmetric int8 extremes.
    values = np.array([0.5, -0.5, 0.25, 0.0], dtype=np.float32)
    result = quantize_per_tensor_int8(values, scale=0.5 / 127)
    assert result.tolist() == [127, -127, 64, 0]  # round(0.25/scale) = 63.5 -> 64
    assert result.dtype == np.int8


def test_quantize_clamps_to_symmetric_range():
    # -128 is excluded: the symmetric range is [-127, 127].
    values = np.array([2.0, -2.0], dtype=np.float32)
    result = quantize_per_tensor_int8(values, scale=0.5 / 127)
    assert result.tolist() == [127, -127]


def test_decompose_multiplier_basic():
    assert decompose_multiplier(0.75) == (192, 8)  # 192 * 2^-8 = 0.75
    assert decompose_multiplier(0.5) == (128, 8)
    assert decompose_multiplier(1.5) == (192, 7)


def test_decompose_multiplier_renormalizes_mantissa_overflow():
    # m rounds up to 2^8; renormalization halves M0 and decrements n (lossless).
    assert decompose_multiplier(0.999999) == (128, 7)


def test_decompose_multiplier_rejects_dead_and_degenerate_layers():
    with pytest.raises(AssertionError, match="must be positive"):
        decompose_multiplier(0.0)
    with pytest.raises(AssertionError, match="mis-calibrated"):
        decompose_multiplier(256.0)  # n would be negative


def _write_fixture(tmp_path, M=0.75):
    mlir = (
        'module @m {\n'
        '  func.func public @main(%arg0: tensor<2x1xf32> loc("states[0]"), '
        '%arg1: tensor<1xf32> loc("states[1]"), '
        '%arg2: tensor<1x1xf32> loc("inputs[0][0]")) -> (tensor<1x1xf32>) {\n'
        '    return %arg2 : tensor<1x1xf32>\n'
        '  }\n'
        '}\n'
    )
    (tmp_path / "initial.mlir").write_text(mlir)
    S_w = 0.5 / 127  # makes +/-0.5 quantize to the int8 extremes
    np.savez(
        tmp_path / "weights.npz",
        **{
            "hidden.weight": np.array([[0.5], [-0.5]], dtype=np.float32),
            "hidden.bias": np.array([0.25], dtype=np.float32),
            "__order__": np.array(["hidden.weight", "hidden.bias"]),
            "__M__hidden.weight": np.float64(M),
            "__scales__hidden.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
        },
    )


def test_weight_map_addresses_kinds_and_requant(tmp_path):
    _write_fixture(tmp_path)
    weight_map = Process_Weights(tmp_path)

    # Weight entry carries M0/n from decomposing M=0.75.
    assert weight_map["%arg0"] == {
        "kind": "weight",
        "label": "hidden.weight",
        "address": 2,
        "shape": [2, 1],
        "num_words": 2,
        "M0": 192,
        "n": 8,
    }
    # Bias follows contiguously and carries no M0/n.
    assert weight_map["%arg1"]["kind"] == "bias"
    assert weight_map["%arg1"]["address"] == 4
    assert "M0" not in weight_map["%arg1"]
    # The input maps to scratch 0x1.
    assert weight_map["%arg2"]["kind"] == "input"
    assert weight_map["%arg2"]["address"] == 1


def test_mem_bin_uses_per_tensor_int8_values(tmp_path):
    _write_fixture(tmp_path)
    Process_Weights(tmp_path)
    mem = (tmp_path / "MEM.bin").read_bytes()

    # One contiguous block: MEM, addr=0x0002, len=0x0003, then 3 int8 bytes.
    assert mem[0] == MEM
    assert mem[1:3] == (2).to_bytes(2, "little")
    assert mem[3:5] == (3).to_bytes(2, "little")
    # weight +0.5 -> 127, -0.5 -> -127 (0x81); bias 0.25 -> self-scale -> 127.
    assert list(mem[5:8]) == [127, 0x81, 127]
