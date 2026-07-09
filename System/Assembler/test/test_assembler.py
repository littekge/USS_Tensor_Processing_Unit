"""Tests for instruction encoding and assembly (v0.4: 24-bit address fields).

Field layouts follow `Functional_TPU_ISA.md` v0.4 exactly. Addresses (rs1/rs2/rd)
are 24 bits; the MUL requantization pair moved to M0 35-28 / n 27-20. Intermediate
results are allocated in main data memory and the final output is pinned to 0x1.
"""

from nn_assembler.Assembler import (
    OPCODE_ADD,
    assemble_program,
    encode_add,
    encode_end,
    encode_mult,
    encode_relu,
)


def _bits(value, hi, lo):
    return (value >> lo) & ((1 << (hi - lo + 1)) - 1)


def _is_128_bit(value):
    return 0 <= value < (1 << 128)


def test_encode_mult_fields_including_requant():
    instr = encode_mult(rs1=1, lhs=(1, 1), rs2=2, rhs=(1, 4), rd=1, M0=192, n=8)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1000  # opcode
    assert _bits(instr, 123, 100) == 1  # rs1 (24-bit)
    assert _bits(instr, 99, 96) == 1  # sz11
    assert _bits(instr, 95, 92) == 1  # sz12
    assert _bits(instr, 91, 68) == 2  # rs2 (24-bit)
    assert _bits(instr, 67, 64) == 1  # sz21
    assert _bits(instr, 63, 60) == 4  # sz22
    assert _bits(instr, 59, 36) == 1  # rd (24-bit)
    assert _bits(instr, 35, 28) == 192  # M0 (moved from 59-52)
    assert _bits(instr, 27, 20) == 8  # n (moved from 51-44)
    assert _bits(instr, 19, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3


def test_encode_mult_uses_full_24_bit_addresses():
    big = (1 << 24) - 1
    instr = encode_mult(rs1=big, lhs=(2, 3), rs2=big, rhs=(3, 2), rd=big, M0=255, n=255)
    assert _is_128_bit(instr)
    assert _bits(instr, 123, 100) == big
    assert _bits(instr, 91, 68) == big
    assert _bits(instr, 59, 36) == big
    assert _bits(instr, 35, 28) == 255
    assert _bits(instr, 27, 20) == 255


def test_encode_add_fields():
    instr = encode_add(rs1=6, rs2=1, dims=(1, 4), rd=1)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1010  # opcode
    assert _bits(instr, 123, 100) == 6  # rs1 (24-bit)
    assert _bits(instr, 99, 76) == 1  # rs2 (24-bit)
    assert _bits(instr, 75, 68) == 1  # sz1
    assert _bits(instr, 67, 60) == 4  # sz2
    assert _bits(instr, 59, 36) == 1  # rd (24-bit)
    assert _bits(instr, 35, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3


def test_encode_relu_fields():
    instr = encode_relu(rs1=3, rd=1, length=16)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1001  # opcode
    assert _bits(instr, 123, 100) == 3  # rs1 (24-bit)
    assert _bits(instr, 99, 76) == 1  # rd (24-bit)
    assert _bits(instr, 75, 60) == 16  # len (16-bit)
    assert _bits(instr, 59, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3


def test_encode_end_is_all_zero():
    assert encode_end() == 0


def test_assemble_places_intermediate_in_main_memory_and_output_at_scratch():
    # Bias-removed dialect: two mults chained; the returned value goes to 0x1.
    tpu_text = (
        "module @m {\n"
        "  tpu.func @main {\n"
        "    %1 = tpu.mult %arg4 : 1x1, %arg0 : 1x4 -> 1x4 {M0 = 192, n = 8}\n"
        "    %5 = tpu.mult %1 : 1x4, %arg2 : 4x1 -> 1x1 {M0 = 200, n = 9}\n"
        "    tpu.return %5 : 1x1\n"
        "    tpu.end\n"
        "  }\n"
        "}\n"
    )
    weight_map = {
        "%arg0": {"kind": "weight", "address": 2, "num_words": 4, "M0": 192, "n": 8},
        "%arg2": {"kind": "weight", "address": 7, "num_words": 4, "M0": 200, "n": 9},
        "%arg4": {"kind": "input", "address": 1, "num_words": 1},
    }
    instructions = assemble_program(tpu_text, weight_map)
    assert len(instructions) == 3  # mult, mult, end (return emits nothing)

    # first_free_address = max(2+4, 7+4) = 11, so %1 lands at 0xB.
    mult = instructions[0]
    assert _bits(mult, 123, 100) == 1  # %arg4 input -> 0x1
    assert _bits(mult, 91, 68) == 2  # %arg0 weight -> 0x2
    assert _bits(mult, 59, 36) == 11  # %1 intermediate -> main memory 0xB
    assert (_bits(mult, 35, 28), _bits(mult, 27, 20)) == (192, 8)  # M0/n

    second_mult = instructions[1]
    assert _bits(second_mult, 123, 100) == 11  # %1 intermediate read from 0xB
    assert _bits(second_mult, 91, 68) == 7  # %arg2 weight -> 0x7
    assert _bits(second_mult, 59, 36) == 1  # %5 final output -> 0x1
    assert instructions[2] == 0  # end


def test_assemble_reuses_freed_intermediate_regions():
    # A four-deep chain: %3 must reuse %1's region once %1 is dead.
    tpu_text = (
        "module @m {\n"
        "  tpu.func @main {\n"
        "    %1 = tpu.mult %arg_in : 1x1, %w0 : 1x4 -> 1x4 {M0 = 192, n = 8}\n"
        "    %2 = tpu.mult %1 : 1x4, %w1 : 4x4 -> 1x4 {M0 = 192, n = 8}\n"
        "    %3 = tpu.mult %2 : 1x4, %w2 : 4x4 -> 1x4 {M0 = 192, n = 8}\n"
        "    %4 = tpu.mult %3 : 1x4, %w3 : 4x1 -> 1x1 {M0 = 128, n = 7}\n"
        "    tpu.return %4 : 1x1\n"
        "    tpu.end\n"
        "  }\n"
        "}\n"
    )
    weight_map = {
        "%w0": {"kind": "weight", "address": 2, "num_words": 4, "M0": 192, "n": 8},
        "%w1": {"kind": "weight", "address": 6, "num_words": 16, "M0": 192, "n": 8},
        "%w2": {"kind": "weight", "address": 22, "num_words": 16, "M0": 192, "n": 8},
        "%w3": {"kind": "weight", "address": 38, "num_words": 4, "M0": 128, "n": 7},
        "%arg_in": {"kind": "input", "address": 1, "num_words": 1},
    }
    instructions = assemble_program(tpu_text, weight_map)

    base = 42  # first_free_address = max weight end (38 + 4)
    assert _bits(instructions[0], 59, 36) == base  # %1
    assert _bits(instructions[1], 59, 36) == base + 4  # %2
    assert _bits(instructions[2], 59, 36) == base  # %3 reuses %1's freed region
    assert _bits(instructions[3], 59, 36) == 1  # %4 final output -> 0x1


def test_assembled_program_has_no_add_opcodes():
    # After bias removal a linear Tiny_NN-style program contains only mults + end.
    tpu_text = (
        "module @m {\n"
        "  tpu.func @main {\n"
        "    %1 = tpu.mult %arg4 : 1x1, %arg0 : 1x4 -> 1x4 {M0 = 192, n = 8}\n"
        "    %5 = tpu.mult %1 : 1x4, %arg2 : 4x1 -> 1x1 {M0 = 200, n = 9}\n"
        "    tpu.return %5 : 1x1\n"
        "    tpu.end\n"
        "  }\n"
        "}\n"
    )
    weight_map = {
        "%arg0": {"kind": "weight", "address": 2, "num_words": 4, "M0": 192, "n": 8},
        "%arg2": {"kind": "weight", "address": 7, "num_words": 4, "M0": 200, "n": 9},
        "%arg4": {"kind": "input", "address": 1, "num_words": 1},
    }
    instructions = assemble_program(tpu_text, weight_map)
    assert all(_bits(instr, 127, 124) != OPCODE_ADD for instr in instructions)
