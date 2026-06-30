"""Encoders for the `Functional_TPU_Message_Protocol_v0.2.md` byte format.

Centralizes the function codes and block layouts so every stage of the pipeline
agrees on the wire format. All multi-byte fields are little-endian (least
significant byte first), per the protocol's *Byte Order* section.
"""

from __future__ import annotations

# Single-byte ASCII function codes, grouped per the protocol's categories.
#
# Header codes — mark the beginning of a transmission. (INPUT is defined by the
# spec but unused here; the current flow programs the whole peripheral.)
FLASH = ord("U")  # 0x55 — programs the entire peripheral (renamed from START in v0.2)

# Command codes — sub-functions contained within a transmission.
MEM = ord("M")  # 0x4D — write data to device memory
PROGRAM = ord("P")  # 0x50 — write an instruction to program memory

# Trailer codes — mark the end of a transmission.
STOP = ord("S")  # 0x53

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
