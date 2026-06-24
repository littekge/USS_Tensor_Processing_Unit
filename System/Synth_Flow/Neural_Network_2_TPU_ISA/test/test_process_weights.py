"""Tests for Q0.7 quantization and weight mapping (build steps 1-3)."""

import numpy as np

from Process_Weights import Process_Weights, quantize_q07
from Protocol import MEM


def test_quantize_round_to_nearest():
    values = np.array([0.0, 0.49396998, 0.0600809, 0.381], dtype=np.float32)
    result = quantize_q07(values)
    # round(x * 128): 0, 63, 8, 49.
    assert result.tolist() == [0, 63, 8, 49]
    assert result.dtype == np.int8


def test_quantize_clamps_out_of_range():
    values = np.array([-1.1003141, 1.5, -3.0], dtype=np.float32)
    result = quantize_q07(values)
    assert result.tolist() == [-128, 127, -128]


def _write_fixture(tmp_path):
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
    np.savez(
        tmp_path / "weights.npz",
        **{
            "w": np.array([[0.5], [-0.5]], dtype=np.float32),
            "b": np.array([0.25], dtype=np.float32),
            "__order__": np.array(["w", "b"]),
        },
    )


def test_weight_map_addresses_and_kinds(tmp_path):
    _write_fixture(tmp_path)
    weight_map = Process_Weights(tmp_path)

    assert weight_map["%arg0"] == {
        "kind": "weight",
        "label": "w",
        "address": 2,
        "shape": [2, 1],
        "num_words": 2,
    }
    # Second weight follows the first contiguously: 0x2 + 2 words = 0x4.
    assert weight_map["%arg1"]["address"] == 4
    assert weight_map["%arg1"]["num_words"] == 1
    # The input maps to scratch 0x1.
    assert weight_map["%arg2"]["kind"] == "input"
    assert weight_map["%arg2"]["address"] == 1


def test_mem_bin_layout(tmp_path):
    _write_fixture(tmp_path)
    Process_Weights(tmp_path)
    mem = (tmp_path / "MEM.bin").read_bytes()

    # One contiguous block: MEM, addr=0x0002, len=0x0003, then 3 Q0.7 bytes.
    assert mem[0] == MEM
    assert mem[1:3] == (2).to_bytes(2, "little")
    assert mem[3:5] == (3).to_bytes(2, "little")
    # round(0.5*128)=64, round(-0.5*128)=-64 -> 0xC0, round(0.25*128)=32.
    assert list(mem[5:8]) == [64, 0xC0, 32]
