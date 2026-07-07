"""Tests for the TPU dialect (serialize/parse), legalization, and bias removal."""

from pathlib import Path

from nn_assembler.MLIR.bias_removal import remove_bias_adds
from nn_assembler.MLIR.dialect import (
    AddOp,
    EndOp,
    MultOp,
    Operand,
    Program,
    ReturnOp,
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
