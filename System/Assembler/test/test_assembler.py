"""Tests for instruction encoding and assembly (build steps 6-7)."""

from nn_assembler.Assembler import assemble_program, encode_add, encode_end, encode_mult, encode_relu


def _bits(value, hi, lo):
    return (value >> lo) & ((1 << (hi - lo + 1)) - 1)


def test_encode_mult_fields():
    instr = encode_mult(rs1=1, lhs=(1, 1), rs2=2, rhs=(1, 4), rd=1)
    assert _bits(instr, 127, 124) == 0b1000  # opcode
    assert _bits(instr, 123, 108) == 1  # rs1
    assert _bits(instr, 107, 104) == 1  # sz11
    assert _bits(instr, 103, 100) == 1  # sz12
    assert _bits(instr, 99, 84) == 2  # rs2
    assert _bits(instr, 83, 80) == 1  # sz21
    assert _bits(instr, 79, 76) == 4  # sz22
    assert _bits(instr, 75, 60) == 1  # rd
    assert _bits(instr, 59, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3


def test_encode_add_fields():
    instr = encode_add(rs1=6, rs2=1, dims=(1, 4), rd=1)
    assert _bits(instr, 127, 124) == 0b1010
    assert _bits(instr, 123, 108) == 6
    assert _bits(instr, 107, 92) == 1
    assert _bits(instr, 91, 84) == 1
    assert _bits(instr, 83, 76) == 4
    assert _bits(instr, 75, 60) == 1
    assert _bits(instr, 2, 0) == 0


def test_encode_relu_fields():
    instr = encode_relu(rs1=3, rd=1, length=16)
    assert _bits(instr, 127, 124) == 0b1001
    assert _bits(instr, 123, 108) == 3
    assert _bits(instr, 107, 92) == 1
    assert _bits(instr, 91, 76) == 16
    assert _bits(instr, 2, 0) == 0


def test_encode_end_is_all_zero():
    assert encode_end() == 0


def test_assemble_uses_scratch_for_input_and_intermediates():
    tpu_text = (
        "module @m {\n"
        "  tpu.func @main {\n"
        "    %1 = tpu.mult %arg4 : 1x1, %arg0 : 1x4 -> 1x4\n"
        "    %3 = tpu.add %arg1 : 1x4, %1 : 1x4 -> 1x4\n"
        "    tpu.return %3 : 1x4\n"
        "    tpu.end\n"
        "  }\n"
        "}\n"
    )
    weight_map = {
        "%arg0": {"kind": "weight", "address": 2},
        "%arg1": {"kind": "weight", "address": 6},
        "%arg4": {"kind": "input", "address": 1},
    }
    instructions = assemble_program(tpu_text, weight_map)
    assert len(instructions) == 3  # mult, add, end (return emits nothing)

    mult = instructions[0]
    assert _bits(mult, 123, 108) == 1  # %arg4 input -> scratch 0x1
    assert _bits(mult, 99, 84) == 2  # %arg0 weight -> 0x2
    assert _bits(mult, 75, 60) == 1  # %1 intermediate -> scratch 0x1

    add = instructions[1]
    assert _bits(add, 123, 108) == 6  # %arg1 weight -> 0x6
    assert _bits(add, 107, 92) == 1  # %1 intermediate -> scratch 0x1
    assert instructions[2] == 0  # end
