"""Encoders for the `Functional_TPU_Message_Protocol_v0.1.md` byte format.

Centralizes the function codes and block layouts so every stage of the pipeline
agrees on the wire format. All multi-byte fields are little-endian (least
significant byte first), per the protocol's *Byte Order* section.
"""

from __future__ import annotations

# Single-byte ASCII function codes.
START = ord("U")  # 0x55
STOP = ord("S")  # 0x53
MEM = ord("M")  # 0x4D
PROGRAM = ord("P")  # 0x50

ADDRESS_SPACE = 1 << 16  # 65536 words; addresses must fit in 16 bits.
INSTRUCTION_BYTES = 16  # 128-bit instructions.


def build_mem_block(address: int, data: bytes) -> bytes:
    """Build one MEM block: code, 16-bit address, 16-bit length, then data.

    `data` is written byte-for-byte to consecutive addresses starting at
    `address` (one byte per word, matching the Q0.7 word size).
    """
    assert 0 <= address < ADDRESS_SPACE, f"MEM address {address} out of 16-bit range."
    length = len(data)
    assert length < ADDRESS_SPACE, f"MEM block length {length} exceeds 16-bit range."
    assert address + length <= ADDRESS_SPACE, "MEM block overruns the address space."

    return bytes([MEM]) + address.to_bytes(2, "little") + length.to_bytes(2, "little") + data


def build_program_instruction(instruction: int) -> bytes:
    """Build one PROGRAM block: code followed by a 16-byte little-endian instruction."""
    assert 0 <= instruction < (1 << 128), "Instruction does not fit in 128 bits."
    return bytes([PROGRAM]) + instruction.to_bytes(INSTRUCTION_BYTES, "little")
