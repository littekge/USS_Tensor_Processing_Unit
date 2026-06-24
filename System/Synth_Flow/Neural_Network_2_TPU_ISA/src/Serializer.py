"""Lowering step 4 (Step 8): final conversion / serialization.

Concatenates `/tmp/MEM.bin` and `/tmp/PROGRAM.bin` (MEM first, then PROGRAM) and
wraps them with the START and STOP function codes to form a complete Message
Protocol transmission, saved to `/out/TRANSMISSION.bin`.
"""

from __future__ import annotations

from pathlib import Path

from Protocol import START, STOP


def Serialize(tmp_dir: Path | None = None, out_dir: Path | None = None) -> Path:
    """Build the final transmission and write it to out/TRANSMISSION.bin."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "out"

    mem_path = tmp_dir / "MEM.bin"
    program_path = tmp_dir / "PROGRAM.bin"
    assert mem_path.is_file(), f"Missing {mem_path}"
    assert program_path.is_file(), f"Missing {program_path}"

    out_dir.mkdir(parents=True, exist_ok=True)
    transmission_path = out_dir / "TRANSMISSION.bin"

    transmission = bytearray([START])
    transmission += mem_path.read_bytes()
    transmission += program_path.read_bytes()
    transmission.append(STOP)

    transmission_path.write_bytes(bytes(transmission))
    return transmission_path
