"""Tests for the Message Protocol v0.3 byte format (3-byte MEM address).

The MEM command carries a 24-bit address (LADD, MADD, UADD; LSB first) and a
16-bit length (LLEN, ULEN; LSB first). The length field is NOT widened, so a
tensor longer than 65535 words is split across consecutive MEM commands with
incrementing 24-bit addresses.
"""

import pytest

from nn_assembler.Protocol import (
    MEM,
    MEM_MAX_LENGTH,
    PROGRAM,
    build_mem_block,
    build_mem_blocks,
    build_program_instruction,
)


def test_mem_block_has_three_byte_address():
    address = 0x123456  # spans all three address bytes
    block = build_mem_block(address, bytes([0xAA, 0xBB]))
    assert block[0] == MEM
    assert block[1:4] == address.to_bytes(3, "little")  # LADD, MADD, UADD
    assert list(block[1:4]) == [0x56, 0x34, 0x12]  # LSB first
    assert block[4:6] == (2).to_bytes(2, "little")  # LLEN, ULEN
    assert block[6:] == bytes([0xAA, 0xBB])


def test_mem_block_rejects_over_length_payload():
    with pytest.raises(AssertionError, match="build_mem_blocks"):
        build_mem_block(0x2, bytes(MEM_MAX_LENGTH + 1))


def test_mem_block_rejects_address_beyond_24_bits():
    with pytest.raises(AssertionError, match="24-bit"):
        build_mem_block(1 << 24, bytes([0x00]))


def test_build_mem_blocks_single_command_when_short():
    blocks = build_mem_blocks(0x2, bytes(range(4)))
    assert blocks == build_mem_block(0x2, bytes(range(4)))  # no extra framing


def test_build_mem_blocks_chunks_long_tensor():
    # Bigger_NN-style 100000-word layer: exceeds the 16-bit length field.
    total = 100_000
    data = bytes((i % 251) for i in range(total))
    base = 0x1000
    blob = build_mem_blocks(base, data)

    # Two commands: 65535 words then the 34465-word remainder.
    first_len = MEM_MAX_LENGTH
    second_len = total - MEM_MAX_LENGTH

    # First command.
    assert blob[0] == MEM
    assert blob[1:4] == base.to_bytes(3, "little")
    assert blob[4:6] == first_len.to_bytes(2, "little")
    assert blob[6 : 6 + first_len] == data[:first_len]

    # Second command starts immediately after the first command's data, with a
    # 24-bit address advanced by exactly the first chunk's word count.
    second = 6 + first_len
    assert blob[second] == MEM
    assert blob[second + 1 : second + 4] == (base + first_len).to_bytes(3, "little")
    assert blob[second + 4 : second + 6] == second_len.to_bytes(2, "little")
    assert blob[second + 6 :] == data[first_len:]

    # Two commands total: each adds a 6-byte header to its data payload.
    assert len(blob) == total + 2 * 6


def test_program_instruction_is_lsb_first_16_bytes():
    instruction = 0x0102030405060708090A0B0C0D0E0F10
    block = build_program_instruction(instruction)
    assert block[0] == PROGRAM
    assert block[1:] == instruction.to_bytes(16, "little")
    assert len(block) == 17
