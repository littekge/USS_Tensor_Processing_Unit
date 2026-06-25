"""Tests for the TPU dialect (serialize/parse) and the legalization pass."""

from pathlib import Path

from nn_assembler.MLIR.dialect import AddOp, EndOp, MultOp, ReturnOp, parse_program, serialize
from nn_assembler.MLIR.legalize import legalize_text

EXAMPLE = Path(__file__).parent.parent / "examples" / "v0.1_optimized_example.mlir"


def test_legalize_folds_reshape_and_lowers_ops():
    program = legalize_text(EXAMPLE.read_text())
    kinds = [type(op).__name__ for op in program.ops]
    # reshapes are folded away; only mult/add remain, plus return + end.
    assert kinds == ["MultOp", "AddOp", "MultOp", "AddOp", "ReturnOp", "EndOp"]


def test_legalize_resolves_reshaped_operands():
    program = legalize_text(EXAMPLE.read_text())
    first_mult = program.ops[0]
    assert isinstance(first_mult, MultOp)
    # %1 = dot_general %arg4, reshape(%arg0); the reshape result folds to %arg0.
    assert first_mult.lhs.name == "%arg4"
    assert first_mult.lhs.shape == [1, 1]
    assert first_mult.rhs.name == "%arg0"
    assert first_mult.rhs.shape == [1, 4]
    assert first_mult.out_shape == [1, 4]

    # %3 = add reshape(%arg1), %1  ->  arg1 folded, second operand is the mult result.
    add = program.ops[1]
    assert isinstance(add, AddOp)
    assert add.lhs.name == "%arg1"
    assert add.rhs.name == "%1"


def test_dialect_roundtrip():
    program = legalize_text(EXAMPLE.read_text())
    text = serialize(program)
    reparsed = parse_program(text)
    assert serialize(reparsed) == text


def test_return_marks_final_value():
    program = legalize_text(EXAMPLE.read_text())
    ret = [op for op in program.ops if isinstance(op, ReturnOp)][0]
    assert ret.value == "%7"
    assert isinstance(program.ops[-1], EndOp)
