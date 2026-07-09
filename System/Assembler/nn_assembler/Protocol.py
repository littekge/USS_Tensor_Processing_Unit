"""Encoders for the `Functional_TPU_Message_Protocol.md` (v0.3) byte format.

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

ADDRESS_SPACE = 1 << 24  # 2^24 words; MEM addresses are 24-bit (v0.4).
MEM_ADDRESS_BYTES = 3  # LADD, MADD, UADD (LSB first).
MEM_LENGTH_BYTES = 2  # LLEN, ULEN (LSB first); the length field is NOT widened.
MEM_MAX_LENGTH = (1 << 16) - 1  # 65535 words per MEM command (16-bit length).
INSTRUCTION_BYTES = 16  # 128-bit instructions.


def build_mem_block(address: int, data: bytes) -> bytes:
    """Build one MEM command: code, 3-byte address, 16-bit length, then data.

    Byte order per Message Protocol v0.3: M, LADD, MADD, UADD, LLEN, ULEN, DATA.
    `data` is written byte-for-byte to consecutive addresses starting at
    `address` (one byte per word). A single command holds at most
    `MEM_MAX_LENGTH` words; use `build_mem_blocks` to chunk larger tensors.
    """
    assert 0 <= address < ADDRESS_SPACE, f"MEM address {address} out of 24-bit range."
    length = len(data)
    assert length <= MEM_MAX_LENGTH, (
        f"MEM command length {length} exceeds the 16-bit length field "
        f"(max {MEM_MAX_LENGTH}); split with build_mem_blocks."
    )
    assert address + length <= ADDRESS_SPACE, "MEM command overruns the address space."

    return (
        bytes([MEM])
        + address.to_bytes(MEM_ADDRESS_BYTES, "little")
        + length.to_bytes(MEM_LENGTH_BYTES, "little")
        + data
    )


def build_mem_blocks(address: int, data: bytes) -> bytes:
    """Build one or more MEM commands, chunking `data` to the 16-bit length limit.

    A tensor longer than `MEM_MAX_LENGTH` words is split across consecutive MEM
    commands with incrementing 24-bit addresses (one word per address).
    """
    assert 0 <= address < ADDRESS_SPACE, f"MEM address {address} out of 24-bit range."
    blob = bytearray()
    offset = 0
    while offset < len(data):
        chunk = data[offset : offset + MEM_MAX_LENGTH]
        blob += build_mem_block(address + offset, chunk)
        offset += len(chunk)
    return bytes(blob)


def build_program_instruction(instruction: int) -> bytes:
    """Build one PROGRAM block: code followed by a 16-byte little-endian instruction."""
    assert 0 <= instruction < (1 << 128), "Instruction does not fit in 128 bits."
    return bytes([PROGRAM]) + instruction.to_bytes(INSTRUCTION_BYTES, "little")
