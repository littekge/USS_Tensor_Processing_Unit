"""Tests for per-tensor int8 quantization, dyadic M decomposition, and mapping.

v0.3: weights are quantized per-tensor symmetric int8 using the exported S_w, and
each layer's requantization multiplier M is decomposed into the ISA dyadic pair
(M0, n). Biases are still quantized/mapped (for stable addresses) but carry no
M0/n.
"""

import numpy as np
import pytest

from nn_assembler.Process_Weights import (
    FINAL_OUTPUT_ADDRESS,
    INPUT_ADDRESS,
    IntermediateAllocator,
    Process_Weights,
    decompose_multiplier,
    first_free_address,
    flatten_conv_weight_shape,
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

    # One contiguous command: MEM, 3-byte addr=0x000002, 2-byte len=0x0003, data.
    assert mem[0] == MEM
    assert mem[1:4] == (2).to_bytes(3, "little")  # LADD, MADD, UADD
    assert mem[4:6] == (3).to_bytes(2, "little")  # LLEN, ULEN
    # weight +0.5 -> 127, -0.5 -> -127 (0x81); bias 0.25 -> self-scale -> 127.
    assert list(mem[6:9]) == [127, 0x81, 127]


def test_flatten_conv_weight_shape():
    # (out_ch, in_ch, kh, kw) -> (out_ch, in_ch*kh*kw).
    assert flatten_conv_weight_shape([6, 1, 3, 3]) == [6, 9]
    assert flatten_conv_weight_shape([16, 6, 3, 3]) == [16, 54]


def _write_conv_fixture(tmp_path, M=0.75):
    mlir = (
        'module @m {\n'
        '  func.func public @main(%arg0: tensor<2x2x2x2xf32> loc("states[0]"), '
        '%arg1: tensor<1x1x3x3xf32> loc("inputs[0][0]")) -> (tensor<1x1xf32>) {\n'
        '    return %arg1 : tensor<1x1xf32>\n'
        '  }\n'
        '}\n'
    )
    (tmp_path / "initial.mlir").write_text(mlir)
    # A 2-out-channel, 2-in-channel, 2x2 kernel with distinct per-position values so
    # the channel-major flatten order is observable in the quantized bytes.
    kernel = np.arange(1, 17, dtype=np.float32).reshape(2, 2, 2, 2)  # values 1..16
    S_w = 16.0 / 127  # absmax/127 so value 16 -> 127
    np.savez(
        tmp_path / "weights.npz",
        **{
            "conv.weight": kernel,
            "__order__": np.array(["conv.weight"]),
            "__M__conv.weight": np.float64(M),
            "__scales__conv.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
        },
    )
    return kernel, S_w


def test_conv_weight_flatten_is_channel_major(tmp_path):
    kernel, S_w = _write_conv_fixture(tmp_path)
    weight_map = Process_Weights(tmp_path)

    # Recorded shape is the 2-D matrix (out_ch) x (in_ch*kh*kw).
    entry = weight_map["%arg0"]
    assert entry["kind"] == "weight"
    assert entry["shape"] == [2, 8]  # 2 out, 2*2*2 = 8

    # MEM bytes must be the channel-major row-major flatten of (out,in,kh,kw):
    # k = c*(kh*kw) + ki*kw + kj, matching im2col column order.
    mem = (tmp_path / "MEM.bin").read_bytes()
    data = mem[6:]  # after M + 3-byte addr + 2-byte len
    expected = np.clip(np.round(kernel.reshape(-1) / S_w), -127, 127).astype(np.int8)
    assert list(data) == list(expected.astype(np.uint8))


def test_input_reserved_at_scratch_and_output_shares_it():
    # 0x1 is the single reserved I/O designation for both input and final output.
    assert INPUT_ADDRESS == 0x1
    assert FINAL_OUTPUT_ADDRESS == 0x1


def test_weights_are_not_placed_at_reserved_io_address(tmp_path):
    _write_fixture(tmp_path)
    weight_map = Process_Weights(tmp_path)
    mapped = [e["address"] for e in weight_map.values() if e["kind"] in ("weight", "bias")]
    assert all(address >= 2 for address in mapped)  # never on 0x0/0x1


def test_first_free_address_is_past_weight_region():
    weight_map = {
        "%arg0": {"kind": "weight", "address": 2, "num_words": 4},
        "%arg1": {"kind": "bias", "address": 6, "num_words": 4},
        "%argIn": {"kind": "input", "address": 1, "num_words": 1},
    }
    assert first_free_address(weight_map) == 10  # max(2+4, 6+4); input ignored


def test_first_free_address_defaults_when_no_weights():
    assert first_free_address({"%in": {"kind": "input", "address": 1}}) == 2


def test_intermediate_allocator_bumps_high_water():
    allocator = IntermediateAllocator(base_address=10)
    assert allocator.allocate(4) == 10
    assert allocator.allocate(2) == 14
    assert allocator.allocate(1) == 16


def test_intermediate_allocator_reuses_freed_regions():
    allocator = IntermediateAllocator(base_address=10)
    first = allocator.allocate(4)  # 10
    second = allocator.allocate(4)  # 14
    allocator.free(first, 4)  # region 10..13 returns to the free list
    third = allocator.allocate(4)  # reuses the freed region
    assert (first, second, third) == (10, 14, 10)


def test_intermediate_allocator_merges_adjacent_free_regions():
    allocator = IntermediateAllocator(base_address=10)
    a = allocator.allocate(4)  # 10
    b = allocator.allocate(4)  # 14
    allocator.free(a, 4)
    allocator.free(b, 4)  # adjacent -> merges into a single 8-word region
    assert allocator.allocate(8) == 10
