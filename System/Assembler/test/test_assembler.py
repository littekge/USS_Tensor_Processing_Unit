"""Tests for instruction encoding and assembly (v0.4: 24-bit address fields).

Field layouts follow `Functional_TPU_ISA.md` v0.4 exactly. Addresses (rs1/rs2/rd)
are 24 bits; the MUL requantization pair moved to M0 35-28 / n 27-20. Intermediate
results are allocated in main data memory and the final output is pinned to 0x1.
"""

from nn_assembler.Assembler import (
    MAX_MATMUL_SIZE,
    OPCODE_ADD,
    OPCODE_IM2COL,
    OPCODE_MAX,
    OPCODE_MULT,
    OPCODE_STRIDE,
    OPCODE_WINDOW,
    assemble_program,
    encode_add,
    encode_end,
    encode_im2col,
    encode_max,
    encode_mult,
    encode_mult_in_place,
    encode_relu,
    encode_stride,
    encode_window,
)
from nn_assembler.MLIR.dialect import MultOp, Operand, Program, ReturnOp, EndOp
from nn_assembler.MLIR.partition import partition_program


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


def test_encode_window_fields():
    # conv1 window: chans=1, inh=inw=28, outh=outw=26, winh=winw=3, unit stride, no pad.
    instr = encode_window(
        chans=1, inh=28, inw=28, outh=26, outw=26,
        winh=3, winw=3, strh=1, strw=1, padh=0, padw=0,
    )
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1110  # opcode WINCONFIG
    assert _bits(instr, 123, 112) == 28  # inh (12-bit)
    assert _bits(instr, 111, 100) == 28  # inw (12-bit)
    assert _bits(instr, 99, 88) == 1  # chans (12-bit)
    assert _bits(instr, 87, 76) == 26  # outh (12-bit)
    assert _bits(instr, 75, 64) == 26  # outw (12-bit)
    assert _bits(instr, 63, 60) == 3  # winh (4-bit)
    assert _bits(instr, 59, 56) == 3  # winw (4-bit)
    assert _bits(instr, 55, 52) == 1  # strh (4-bit)
    assert _bits(instr, 51, 48) == 1  # strw (4-bit)
    assert _bits(instr, 47, 44) == 0  # padh (4-bit)
    assert _bits(instr, 43, 40) == 0  # padw (4-bit)
    assert _bits(instr, 39, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3 WINDOW


def test_encode_window_pool_fields():
    # pool1 window: chans=6, in 26x26, out 13x13, 2x2 window stride 2.
    instr = encode_window(
        chans=6, inh=26, inw=26, outh=13, outw=13,
        winh=2, winw=2, strh=2, strw=2, padh=0, padw=0,
    )
    assert _bits(instr, 99, 88) == 6  # chans
    assert _bits(instr, 87, 76) == 13  # outh
    assert _bits(instr, 63, 60) == 2  # winh
    assert _bits(instr, 55, 52) == 2  # strh


def test_encode_im2col_fields():
    instr = encode_im2col(rs1=1, rd=20)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1101  # opcode SHAPE
    assert _bits(instr, 123, 100) == 1  # rs1 (src)
    assert _bits(instr, 99, 76) == 20  # rd (dest)
    assert _bits(instr, 75, 60) == 0  # aux
    assert _bits(instr, 59, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3 IM2COL


def test_encode_max_fields():
    instr = encode_max(rs1=30, rd=40)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1100  # opcode POOL
    assert _bits(instr, 123, 100) == 30  # rs1 (src)
    assert _bits(instr, 99, 76) == 40  # rd (dest)
    assert _bits(instr, 75, 60) == 0  # aux
    assert _bits(instr, 59, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0  # funct3 MAX


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


def test_encode_multip_shares_mult_layout_differs_in_funct3():
    m = encode_mult(rs1=5, lhs=(8, 8), rs2=9, rhs=(8, 8), rd=7, M0=128, n=8)
    mp = encode_mult_in_place(rs1=5, lhs=(8, 8), rs2=9, rhs=(8, 8), rd=7, M0=128, n=8)
    # Identical everywhere except funct3 (bits 2-0): MULT=0x0, MULTIP=0x1.
    assert _bits(m, 2, 0) == 0x0
    assert _bits(mp, 2, 0) == 0x1
    assert (m >> 3) == (mp >> 3)  # all higher fields identical
    assert _bits(mp, 127, 124) == 0b1000  # still MUL opcode


def test_encode_stride_fields():
    instr = encode_stride(ld1=1000, ld2=100)
    assert _is_128_bit(instr)
    assert _bits(instr, 127, 124) == 0b1111  # CONFIG opcode
    assert _bits(instr, 123, 100) == 1000  # im1 = ld1
    assert _bits(instr, 99, 76) == 100  # im2 = ld2
    assert _bits(instr, 75, 3) == 0  # reserved
    assert _bits(instr, 2, 0) == 0x0  # funct3 STRIDE


def test_encode_mult_tile_dim_over_array_rejected():
    import pytest

    with pytest.raises(AssertionError, match="MAX_MATMUL_SIZE"):
        encode_mult(rs1=1, lhs=(MAX_MATMUL_SIZE + 1, 1), rs2=2, rhs=(1, 1), rd=3, M0=128, n=8)


def _tiled_matmul_program(M, K, N):
    program = Program(name="m")
    program.ops.append(
        MultOp(
            "%r",
            Operand("%in", [M, K]),
            Operand("%w", [K, N]),
            [M, N],
            M0=128,
            n=8,
        )
    )
    program.ops.append(ReturnOp("%dummy", [1, 1]))  # keep %r an intermediate
    program.ops.append(EndOp())
    partition_program(program, MAX_MATMUL_SIZE)
    return program


def test_tiled_emission_stride_count_and_bases():
    # 1x16 @ 16x16 -> 1x16: both K and N exceed the 8-wide array.
    weight_map = {
        "%in": {"kind": "weight", "address": 2, "num_words": 16, "M0": 128, "n": 8},
        "%w": {"kind": "weight", "address": 100, "num_words": 256, "M0": 128, "n": 8},
    }
    program = _tiled_matmul_program(1, 16, 16)
    from nn_assembler.MLIR.dialect import serialize

    instructions = assemble_program(serialize(program), weight_map)

    strides = [i for i in instructions if _bits(i, 127, 124) == OPCODE_STRIDE]
    assert len(strides) == 1  # exactly one stride per matmul
    assert _bits(strides[0], 123, 100) == 16  # ld1 = K
    assert _bits(strides[0], 99, 76) == 16  # ld2 = N

    muls = [i for i in instructions if _bits(i, 127, 124) == OPCODE_MULT]
    # 2 N-tiles x 2 K-tiles = 4 tiles; K-run is (multip, mult) per output tile.
    assert len(muls) == 4
    multips = [i for i in muls if _bits(i, 2, 0) == 0x1]
    mults = [i for i in muls if _bits(i, 2, 0) == 0x0]
    assert len(multips) == 2 and len(mults) == 2  # one multip + one mult per N-tile

    # First tile: N-tile 0, K-tile 0. rs1 = base(%in)+0, rs2 = base(%w)+0.
    first = muls[0]
    assert _bits(first, 123, 100) == 2  # %in corner offset 0
    assert _bits(first, 91, 68) == 100  # %w corner offset 0
    # Second-N-tile addressing: rhs corner col offset = 8, dst offset = 8.
    # The mult terminating N-tile 0's K-run reads %w row 8 (offset 8*16=128).
    second_k = muls[1]
    assert _bits(second_k, 91, 68) == 100 + 8 * 16  # %w sub-block corner (ki=8, ni=0)


def test_tiled_emission_ragged_edges():
    # K=10, N=10 with array 8: last tile in each dim has real size 2.
    weight_map = {
        "%in": {"kind": "weight", "address": 2, "num_words": 10, "M0": 128, "n": 8},
        "%w": {"kind": "weight", "address": 50, "num_words": 100, "M0": 128, "n": 8},
    }
    program = _tiled_matmul_program(1, 10, 10)
    tiles = [op for op in program.ops if isinstance(op, MultOp) or type(op).__name__ == "MultipOp"]
    # Ragged tile dims of 2 appear (10 - 8) for both K and N.
    depths = {op.lhs.shape[1] for op in tiles}
    cols = {op.rhs.shape[1] for op in tiles}
    assert depths == {8, 2}
    assert cols == {8, 2}


def test_small_matmul_passes_through_with_contiguous_stride_reset():
    program = _tiled_matmul_program(1, 4, 4)  # all dims <= 8
    from nn_assembler.MLIR.dialect import StrideOp, MultipOp

    # Pass-through matmul: no multip run, but a stride(0,0) reset precedes the mult
    # so it can never inherit a stale leading dimension from a prior partitioned
    # matmul.
    assert not any(isinstance(op, MultipOp) for op in program.ops)
    strides = [op for op in program.ops if isinstance(op, StrideOp)]
    assert len(strides) == 1
    assert (strides[0].ld1, strides[0].ld2) == (0, 0)
    mults = [op for op in program.ops if isinstance(op, MultOp)]
    assert len(mults) == 1
    assert mults[0].lhs.offset == 0 and mults[0].dst_offset == 0

    # The reset assembles to a stride instruction with contiguous (ld=0) fields.
    weight_map = {
        "%in": {"kind": "weight", "address": 2, "num_words": 4, "M0": 128, "n": 8},
        "%w": {"kind": "weight", "address": 50, "num_words": 16, "M0": 128, "n": 8},
    }
    from nn_assembler.MLIR.dialect import serialize

    instructions = assemble_program(serialize(program), weight_map)
    stride_instrs = [i for i in instructions if _bits(i, 127, 124) == OPCODE_STRIDE]
    assert len(stride_instrs) == 1
    assert _bits(stride_instrs[0], 123, 100) == 0  # ld1 = 0 (contiguous)
    assert _bits(stride_instrs[0], 99, 76) == 0  # ld2 = 0 (contiguous)


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
