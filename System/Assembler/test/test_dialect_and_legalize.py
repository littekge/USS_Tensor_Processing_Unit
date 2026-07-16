"""Tests for the TPU dialect (serialize/parse), legalization, and bias removal."""

from pathlib import Path

from nn_assembler.MLIR.bias_removal import remove_bias_adds
from nn_assembler.MLIR.dialect import (
    AddOp,
    EndOp,
    Im2colOp,
    MaxPoolOp,
    MultipOp,
    MultOp,
    Operand,
    Program,
    ReluOp,
    ReturnOp,
    StrideOp,
    WindowOp,
    parse_program,
    serialize,
)
from nn_assembler.MLIR.legalize import legalize_text

EXAMPLE = Path(__file__).parent.parent / "examples" / "v0.1_optimized_example.mlir"

# Weight map matching the example's @main args: arg0/arg2 are weights (with
# distinct M0/n), arg1/arg3 are biases, arg4 is the network input.
_EXAMPLE_WEIGHT_MAP = {
    "%arg0": {"kind": "weight", "address": 2, "M0": 192, "n": 8},
    "%arg1": {"kind": "bias", "address": 6},
    "%arg2": {"kind": "weight", "address": 7, "M0": 200, "n": 9},
    "%arg3": {"kind": "bias", "address": 11},
    "%arg4": {"kind": "input", "address": 1},
}


def test_legalize_folds_reshape_and_lowers_ops():
    program = legalize_text(EXAMPLE.read_text())
    kinds = [type(op).__name__ for op in program.ops]
    # reshapes are folded away; only mult/add remain, plus return + end.
    assert kinds == ["MultOp", "AddOp", "MultOp", "AddOp", "ReturnOp", "EndOp"]


def test_legalize_resolves_reshaped_operands():
    program = legalize_text(EXAMPLE.read_text())
    first_mult = program.ops[0]
    assert isinstance(first_mult, MultOp)
    assert first_mult.lhs.name == "%arg4"
    assert first_mult.lhs.shape == [1, 1]
    assert first_mult.rhs.name == "%arg0"
    assert first_mult.rhs.shape == [1, 4]
    assert first_mult.out_shape == [1, 4]

    add = program.ops[1]
    assert isinstance(add, AddOp)
    assert add.lhs.name == "%arg1"
    assert add.rhs.name == "%1"


def test_legalize_annotates_mult_with_requant():
    program = legalize_text(EXAMPLE.read_text(), _EXAMPLE_WEIGHT_MAP)
    mults = [op for op in program.ops if isinstance(op, MultOp)]
    # First mult's weight operand is %arg0; second's is %arg2.
    assert (mults[0].M0, mults[0].n) == (192, 8)
    assert (mults[1].M0, mults[1].n) == (200, 9)


def test_multop_roundtrip_with_requant():
    program = Program(name="m")
    program.ops.append(
        MultOp(
            result="%1",
            lhs=Operand("%arg4", [1, 1]),
            rhs=Operand("%arg0", [1, 4]),
            out_shape=[1, 4],
            M0=192,
            n=8,
        )
    )
    program.ops.append(EndOp())

    text = serialize(program)
    assert "{M0 = 192, n = 8}" in text
    reparsed = parse_program(text)
    mult = reparsed.ops[0]
    assert (mult.M0, mult.n) == (192, 8)
    assert serialize(reparsed) == text


def test_dialect_roundtrip():
    program = legalize_text(EXAMPLE.read_text(), _EXAMPLE_WEIGHT_MAP)
    text = serialize(program)
    reparsed = parse_program(text)
    assert serialize(reparsed) == text


def test_bias_removal_drops_and_reroutes_bias_adds():
    program = legalize_text(EXAMPLE.read_text(), _EXAMPLE_WEIGHT_MAP)
    remove_bias_adds(program, _EXAMPLE_WEIGHT_MAP)

    kinds = [type(op).__name__ for op in program.ops]
    # Both bias adds are gone; two mults + return + end remain.
    assert kinds == ["MultOp", "MultOp", "ReturnOp", "EndOp"]

    # Consumers were rerouted to the matmul results (not the erased add results).
    second_mult = program.ops[1]
    assert second_mult.lhs.name == "%1"  # was %3 (the dropped first bias add)
    ret = program.ops[2]
    assert isinstance(ret, ReturnOp)
    assert ret.value == "%5"  # was %7 (the dropped second bias add)


_TRANSPOSE_MLIR = (
    "module @m {\n"
    '  func.func public @main(%arg0: tensor<1x1xf32>, %arg2: tensor<100x1000xf32>) '
    "-> (tensor<1x100xf32>) {\n"
    "    %4 = stablehlo.transpose %arg2, dims = [1, 0] : "
    "(tensor<100x1000xf32>) -> tensor<1000x100xf32>\n"
    "    %5 = stablehlo.dot_general %arg0, %4, contracting_dims = [1] x [0] : "
    "(tensor<1x1000xf32>, tensor<1000x100xf32>) -> tensor<1x100xf32>\n"
    "    return %5 : tensor<1x100xf32>\n"
    "  }\n"
    "}\n"
)


def test_legalize_folds_resolved_weight_transpose():
    # %arg2 is a weight (transposed offline), so the surviving transpose folds like
    # a reshape: the dot's rhs becomes %arg2 read with the transposed shape.
    weight_map = {
        "%arg0": {"kind": "input", "address": 1},
        "%arg2": {"kind": "weight", "address": 2, "M0": 128, "n": 8},
    }
    program = legalize_text(_TRANSPOSE_MLIR, weight_map)
    mult = next(op for op in program.ops if isinstance(op, MultOp))
    assert mult.rhs.name == "%arg2"
    assert mult.rhs.shape == [1000, 100]
    assert (mult.M0, mult.n) == (128, 8)


def test_legalize_rejects_runtime_transpose():
    # A transpose of a non-weight (live) value is out of scope, not silently dropped.
    weight_map = {
        "%arg0": {"kind": "input", "address": 1},
        "%arg2": {"kind": "input", "address": 3},
    }
    import pytest

    with pytest.raises(AssertionError, match="out of scope"):
        legalize_text(_TRANSPOSE_MLIR, weight_map)


_CONV_POOL_MLIR = (
    "module @m {\n"
    '  func.func public @main(%arg0: tensor<2x1x3x3xf32> loc("states[0]"), '
    '%arg1: tensor<2xf32> loc("states[1]"), '
    '%arg2: tensor<1x1x5x5xf32> loc("inputs[0][0]")) '
    "-> (tensor<1x2x1x1xf32> {jax.result_info = \"result[0]\"}) {\n"
    "    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>\n"
    "    %0 = stablehlo.convolution(%arg2, %arg0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], "
    "window = {} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : "
    "(tensor<1x1x5x5xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x3x3xf32>\n"
    "    %1 = stablehlo.reshape %arg1 : (tensor<2xf32>) -> tensor<1x2x1x1xf32>\n"
    "    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x2x1x1xf32>) -> tensor<1x2x3x3xf32>\n"
    "    %3 = stablehlo.add %0, %2 : tensor<1x2x3x3xf32>\n"
    "    %4 = call @relu(%3) : (tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32>\n"
    "    %5 = \"stablehlo.reduce_window\"(%4, %cst) "
    "<{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({\n"
    "    ^bb0(%arg11: tensor<f32>, %arg12: tensor<f32>):\n"
    "      %6 = stablehlo.maximum %arg11, %arg12 : tensor<f32>\n"
    "      stablehlo.return %6 : tensor<f32>\n"
    "    }) : (tensor<1x2x3x3xf32>, tensor<f32>) -> tensor<1x2x1x1xf32>\n"
    "    return %5 : tensor<1x2x1x1xf32>\n"
    "  }\n"
    "  func.func private @relu(%arg0: tensor<1x2x3x3xf32>) -> tensor<1x2x3x3xf32> {\n"
    "    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x2x3x3xf32>\n"
    "    %0 = stablehlo.maximum %arg0, %cst : tensor<1x2x3x3xf32>\n"
    "    return %0 : tensor<1x2x3x3xf32>\n"
    "  }\n"
    "}\n"
)

_CONV_POOL_WEIGHT_MAP = {
    "%arg0": {"kind": "weight", "address": 2, "M0": 192, "n": 8},
    "%arg1": {"kind": "bias", "address": 20},
    "%arg2": {"kind": "input", "address": 1},
}


def test_legalize_lowers_convolution():
    program = legalize_text(_CONV_POOL_MLIR, _CONV_POOL_WEIGHT_MAP)
    win = program.ops[0]
    assert isinstance(win, WindowOp)
    # conv1: 1 in-channel, 5x5 -> 3x3, 3x3 window, unit stride, no pad.
    assert (win.chans, win.inh, win.inw, win.outh, win.outw) == (1, 5, 5, 3, 3)
    assert (win.winh, win.winw, win.strh, win.strw, win.padh, win.padw) == (3, 3, 1, 1, 0, 0)

    im2col = program.ops[1]
    assert isinstance(im2col, Im2colOp)
    assert im2col.src.name == "%arg2"  # network input
    assert im2col.out_shape == [9, 9]  # K = 1*3*3 = 9 (channel-major); N = 3*3 = 9

    mult = program.ops[2]
    assert isinstance(mult, MultOp)
    assert mult.lhs.name == "%arg0" and mult.lhs.shape == [2, 9]  # weight (out_ch x K) is LHS
    assert mult.rhs.name == "%0_col" and mult.rhs.shape == [9, 9]  # im2col columns are RHS
    assert mult.out_shape == [2, 9]  # (out_ch x N), CHW-planar
    assert (mult.M0, mult.n) == (192, 8)


def test_legalize_lowers_reduce_window_max():
    program = legalize_text(_CONV_POOL_MLIR, _CONV_POOL_WEIGHT_MAP)
    windows = [op for op in program.ops if isinstance(op, WindowOp)]
    pools = [op for op in program.ops if isinstance(op, MaxPoolOp)]
    assert len(pools) == 1
    pool = pools[0]
    # pool: 2 channels, 3x3 -> 1x1, 2x2 window, stride 2.
    pool_window = windows[1]
    assert (pool_window.chans, pool_window.inh, pool_window.inw) == (2, 3, 3)
    assert (pool_window.outh, pool_window.outw) == (1, 1)
    assert (pool_window.winh, pool_window.winw, pool_window.strh, pool_window.strw) == (2, 2, 2, 2)
    assert pool.src.name == "%4"  # the relu output
    assert pool.out_shape == [2, 1, 1]


def test_legalize_emits_relu():
    program = legalize_text(_CONV_POOL_MLIR, _CONV_POOL_WEIGHT_MAP)
    relus = [op for op in program.ops if isinstance(op, ReluOp)]
    assert len(relus) == 1
    assert relus[0].length == 18  # 1*2*3*3 elements
    assert relus[0].src.name == "%3"  # bias-add result (rerouted later by bias removal)


def test_bias_removal_reroutes_maxpool_src():
    # A pool directly after conv+bias: the pool's src is the bias-add result, which
    # is dropped, so the src must reroute to the conv (mult) result.
    weight_map = {
        "%arg0": {"kind": "weight", "address": 2, "M0": 128, "n": 8},
        "%b": {"kind": "bias", "address": 20},
        "%in": {"kind": "input", "address": 1},
    }
    program = Program(name="m")
    program.ops.append(Im2colOp("%c", Operand("%in", []), [9, 9]))
    program.ops.append(MultOp("%0", Operand("%arg0", [2, 9]), Operand("%c", [9, 9]), [2, 9], M0=128, n=8))
    program.ops.append(AddOp("%1", Operand("%0", [2, 9]), Operand("%b", [2, 9]), [2, 9]))
    program.ops.append(MaxPoolOp("%2", Operand("%1", []), [2, 1, 1]))
    program.ops.append(ReturnOp("%2", [2, 1, 1]))
    program.ops.append(EndOp())

    remove_bias_adds(program, weight_map)
    pool = next(op for op in program.ops if isinstance(op, MaxPoolOp))
    assert pool.src.name == "%0"  # rerouted from the dropped bias add %1


def test_legalize_asserts_on_unhandled_op():
    import pytest

    bad = (
        "module @m {\n"
        '  func.func public @main(%arg0: tensor<1x1xf32> loc("inputs[0][0]")) '
        "-> (tensor<1x1xf32>) {\n"
        "    %0 = stablehlo.exponential %arg0 : tensor<1x1xf32>\n"
        "    return %0 : tensor<1x1xf32>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(AssertionError, match="Unhandled op"):
        legalize_text(bad, {"%arg0": {"kind": "input", "address": 1}})


def test_stride_op_roundtrip():
    program = Program(name="m")
    program.ops.append(StrideOp(ld1=1000, ld2=100))
    program.ops.append(EndOp())

    text = serialize(program)
    assert "tpu.stride ld1 = 1000, ld2 = 100" in text
    reparsed = parse_program(text)
    stride = reparsed.ops[0]
    assert isinstance(stride, StrideOp)
    assert (stride.ld1, stride.ld2) == (1000, 100)
    assert serialize(reparsed) == text


def test_multip_op_roundtrip_with_tiling_attrs():
    # A partitioned K-tile: multip addressing a sub-block with offsets + parent cap.
    program = Program(name="m")
    program.ops.append(
        MultipOp(
            result="%5",
            lhs=Operand("%1", [1, 8], offset=8),
            rhs=Operand("%arg2", [8, 4], offset=800),
            out_shape=[1, 4],
            M0=128,
            n=8,
            dst_offset=96,
            parent_out_shape=[1, 100],
        )
    )
    program.ops.append(EndOp())

    text = serialize(program)
    assert "tpu.multip" in text
    assert "lo = 8" in text and "ro = 800" in text and "do = 96" in text and "cap = 100" in text

    reparsed = parse_program(text)
    op = reparsed.ops[0]
    assert isinstance(op, MultipOp)
    assert (op.lhs.offset, op.rhs.offset, op.dst_offset) == (8, 800, 96)
    assert (op.M0, op.n) == (128, 8)
    assert op.out_shape == [1, 4]
    # parent capacity round-trips as a product (cap); assembler only uses prod().
    from nn_assembler.MLIR.dialect import _prod

    assert _prod(op.parent_out_shape) == 100
    assert serialize(reparsed) == text


def test_untiled_mult_keeps_legacy_form():
    # A plain, array-sized mult must serialize with no tiling attrs (backward-compat).
    program = Program(name="m")
    program.ops.append(
        MultOp("%1", Operand("%a", [1, 4]), Operand("%b", [4, 1]), [1, 1], M0=192, n=8)
    )
    program.ops.append(EndOp())
    text = serialize(program)
    assert "{M0 = 192, n = 8}" in text
    assert "lo =" not in text and "cap =" not in text


def test_window_op_roundtrip():
    program = Program(name="m")
    program.ops.append(
        WindowOp(chans=6, inh=26, inw=26, outh=13, outw=13, winh=2, winw=2, strh=2, strw=2, padh=0, padw=0)
    )
    program.ops.append(EndOp())

    text = serialize(program)
    assert "tpu.window chans = 6, inh = 26, inw = 26" in text
    reparsed = parse_program(text)
    win = reparsed.ops[0]
    assert isinstance(win, WindowOp)
    assert (win.chans, win.inh, win.inw, win.outh, win.outw) == (6, 26, 26, 13, 13)
    assert (win.winh, win.winw, win.strh, win.strw, win.padh, win.padw) == (2, 2, 2, 2, 0, 0)
    assert serialize(reparsed) == text


def test_im2col_op_roundtrip():
    program = Program(name="m")
    program.ops.append(Im2colOp("%col", Operand("%0", []), [9, 676]))
    program.ops.append(EndOp())

    text = serialize(program)
    assert "%col = tpu.im2col %0 -> 9x676" in text
    reparsed = parse_program(text)
    op = reparsed.ops[0]
    assert isinstance(op, Im2colOp)
    assert op.result == "%col" and op.src.name == "%0" and op.out_shape == [9, 676]
    assert serialize(reparsed) == text


def test_max_op_roundtrip():
    program = Program(name="m")
    program.ops.append(MaxPoolOp("%p", Operand("%4", []), [6, 13, 13]))
    program.ops.append(EndOp())

    text = serialize(program)
    assert "%p = tpu.max %4 -> 6x13x13" in text
    reparsed = parse_program(text)
    op = reparsed.ops[0]
    assert isinstance(op, MaxPoolOp)
    assert op.result == "%p" and op.src.name == "%4" and op.out_shape == [6, 13, 13]
    assert serialize(reparsed) == text


def test_bias_removal_preserves_non_bias_adds():
    # An add of two non-bias operands (e.g. a future partial-sum) must survive.
    weight_map = {"%arg9": {"kind": "input", "address": 1}}
    program = Program(name="m")
    program.ops.append(
        MultOp("%1", Operand("%arg9", [1, 1]), Operand("%arg9", [1, 1]), [1, 1], M0=128, n=8)
    )
    program.ops.append(AddOp("%2", Operand("%1", [1, 1]), Operand("%1", [1, 1]), [1, 1]))
    program.ops.append(ReturnOp("%2", [1, 1]))
    program.ops.append(EndOp())

    remove_bias_adds(program, weight_map)
    kinds = [type(op).__name__ for op in program.ops]
    assert kinds == ["MultOp", "AddOp", "ReturnOp", "EndOp"]
