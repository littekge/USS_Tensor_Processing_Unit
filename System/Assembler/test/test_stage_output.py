"""Tests for the v0.7 output-staging pass (MLIR/stage_output.py)."""

from nn_assembler.Assembler import assemble_program
from nn_assembler.MLIR.dialect import (
    EndOp,
    MoveOp,
    MultOp,
    Operand,
    Program,
    ReturnOp,
    serialize,
)
from nn_assembler.MLIR.stage_output import stage_output


def _bits(value, hi, lo):
    return (value >> lo) & ((1 << (hi - lo + 1)) - 1)


def _program_with_return():
    program = Program(name="m")
    program.ops.append(
        MultOp("%out", Operand("%in", [1, 4]), Operand("%w", [4, 8]), [1, 8], M0=128, n=8)
    )
    program.ops.append(ReturnOp("%out", [1, 8]))
    program.ops.append(EndOp())
    return program


def test_stage_output_inserts_one_move_before_terminator():
    program = _program_with_return()
    stage_output(program)

    # A single MoveOp is inserted immediately before the ReturnOp.
    move_indices = [i for i, op in enumerate(program.ops) if isinstance(op, MoveOp)]
    return_index = next(i for i, op in enumerate(program.ops) if isinstance(op, ReturnOp))
    assert len(move_indices) == 1
    assert move_indices[0] == return_index - 1

    move = program.ops[move_indices[0]]
    assert move.src.name == "%out"  # stages the returned value
    assert move.src.shape == [1, 8]  # length inferred from the return shape
    assert "tpu.move %out -> @io : 1x8" in serialize(program)


def test_stage_output_is_idempotent():
    program = _program_with_return()
    stage_output(program)
    stage_output(program)  # second pass must not add a second move
    assert sum(isinstance(op, MoveOp) for op in program.ops) == 1


def test_stage_output_noop_without_return():
    program = Program(name="m")
    program.ops.append(
        MultOp("%out", Operand("%in", [1, 4]), Operand("%w", [4, 10]), [1, 10], M0=128, n=8)
    )
    program.ops.append(EndOp())
    stage_output(program)
    assert not any(isinstance(op, MoveOp) for op in program.ops)


def test_staged_output_repointed_to_main_memory_on_assemble():
    # After staging, the returned value is allocated in main memory (not pinned to
    # 0x1); the move copies it into 0x1 for readback.
    program = _program_with_return()
    stage_output(program)
    weight_map = {
        "%w": {"kind": "weight", "address": 2, "num_words": 32, "M0": 128, "n": 8},
        "%in": {"kind": "input", "address": 1, "num_words": 4},
    }
    instructions = assemble_program(serialize(program), weight_map)

    mult, move = instructions[0], instructions[1]
    assert _bits(mult, 127, 124) == 0b1000  # MUL
    output_addr = _bits(mult, 59, 36)
    assert output_addr >= 2 and output_addr != 1  # output re-pointed to main memory

    assert _bits(move, 127, 124) == 0b1101 and _bits(move, 2, 0) == 0x1  # SHAPE/MOVE
    assert _bits(move, 123, 100) == output_addr  # move source = output base
    assert _bits(move, 99, 76) == 1  # dest = I/O address 0x1
    assert _bits(move, 75, 60) == 8  # len = 8
    assert instructions[2] == 0  # end
