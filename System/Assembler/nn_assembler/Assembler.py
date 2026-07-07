"""Second half of lowering step 3 (Steps 6-7): the assembler.

Translates the final *Functional TPU* dialect (`/tmp/optimized.nobias.tpu.mlir`,
the bias-removed IR) into `Functional_TPU_ISA.md` machine code, then exports it
in the Message Protocol PROGRAM format (`/tmp/PROGRAM.bin`). Each `tpu.mult`
carries the dyadic requantization pair `M0`/`n` set during legalization.

Address resolution uses `/tmp/weight_map.json`: mapped weights resolve to their
assigned addresses, while the network input and every intermediate result share
the scratch address 0x1 (the ISA's architecturally-isolated temporary store).
"""

from __future__ import annotations

import json
from pathlib import Path

from .MLIR.dialect import AddOp, EndOp, MultOp, Operand, ReluOp, ReturnOp, parse_program
from .Protocol import build_program_instruction

# Opcodes (bits 127-124) and funct codes from the ISA's Instruction Set Listing.
OPCODE_MULT = 0b1000
OPCODE_RELU = 0b1001
OPCODE_ADD = 0b1010
OPCODE_SYSTEM = 0b0000
FUNCT3_DEFAULT = 0x0
FUNCT7_END = 0x0

SCRATCH_ADDRESS = 0x1  # network input + all intermediate results

MUL_MAX_DIM = 15  # MUL format dimension fields are 4 bits.
ELEM_MAX_DIM = 255  # ELEM format dimension fields are 8 bits.
REQUANT_FIELD_MAX = 255  # MUL M0/n fields are each 8 bits (bits 59-52 / 51-44).


def _as_2d(shape: list[int]) -> tuple[int, int]:
    """Normalize a shape to (rows, cols); a 1-D tensor is a single row."""
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 1:
        return 1, shape[0]
    if len(shape) == 0:
        return 1, 1
    raise AssertionError(f"Tensors of order > 2 are not supported by the ISA: shape {shape}")


def _resolve_address(name: str, weight_map: dict) -> int:
    """Map an SSA name to an ISA address: mapped tensors from the map, else scratch.

    Weights and biases resolve to their assigned memory addresses; the network
    input (kind == "input") and every intermediate op result share scratch 0x1.
    """
    entry = weight_map.get(name)
    if entry is not None and entry["kind"] in ("weight", "bias"):
        return entry["address"]
    return SCRATCH_ADDRESS


def encode_mult(
    rs1: int, lhs: tuple[int, int], rs2: int, rhs: tuple[int, int], rd: int, M0: int, n: int
) -> int:
    """Encode a MUL-format `mult` instruction into its 128-bit value.

    `M0` (bits 59-52) and `n` (bits 51-44) are the dyadic requantization pair.
    """
    sz11, sz12 = lhs
    sz21, sz22 = rhs
    for dim in (sz11, sz12, sz21, sz22):
        assert 0 <= dim <= MUL_MAX_DIM, f"mult dimension {dim} exceeds 4-bit field (max {MUL_MAX_DIM})."
    assert 0 <= M0 <= REQUANT_FIELD_MAX, f"mult M0 {M0} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    assert 0 <= n <= REQUANT_FIELD_MAX, f"mult n {n} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    return (
        (OPCODE_MULT << 124)
        | (rs1 << 108)
        | (sz11 << 104)
        | (sz12 << 100)
        | (rs2 << 84)
        | (sz21 << 80)
        | (sz22 << 76)
        | (rd << 60)
        | (M0 << 52)
        | (n << 44)
        | FUNCT3_DEFAULT
    )


def encode_add(rs1: int, rs2: int, dims: tuple[int, int], rd: int) -> int:
    """Encode an ELEM-format `add` instruction into its 128-bit value."""
    sz1, sz2 = dims
    for dim in (sz1, sz2):
        assert 0 <= dim <= ELEM_MAX_DIM, f"add dimension {dim} exceeds 8-bit field (max {ELEM_MAX_DIM})."
    return (
        (OPCODE_ADD << 124)
        | (rs1 << 108)
        | (rs2 << 92)
        | (sz1 << 84)
        | (sz2 << 76)
        | (rd << 60)
        | FUNCT3_DEFAULT
    )


def encode_relu(rs1: int, rd: int, length: int) -> int:
    """Encode an ACT-format `relu` instruction into its 128-bit value."""
    assert 0 <= length < (1 << 16), f"relu length {length} exceeds 16-bit field."
    return (OPCODE_RELU << 124) | (rs1 << 108) | (rd << 92) | (length << 76) | FUNCT3_DEFAULT


def encode_end() -> int:
    """Encode a SYSTEM-format `end` instruction into its 128-bit value."""
    return (OPCODE_SYSTEM << 124) | FUNCT7_END


def assemble_program(tpu_text: str, weight_map: dict) -> list[int]:
    """Assemble TPU dialect text into a list of 128-bit machine instructions."""
    program = parse_program(tpu_text)
    instructions: list[int] = []

    for op in program.ops:
        if isinstance(op, MultOp):
            assert op.M0 is not None and op.n is not None, (
                f"mult {op.result} is missing its M0/n requantization pair; "
                "legalization should have annotated it from weight_map.json."
            )
            instructions.append(
                encode_mult(
                    _resolve_address(op.lhs.name, weight_map),
                    _as_2d(op.lhs.shape),
                    _resolve_address(op.rhs.name, weight_map),
                    _as_2d(op.rhs.shape),
                    _resolve_address(op.result, weight_map),
                    op.M0,
                    op.n,
                )
            )
        elif isinstance(op, AddOp):
            instructions.append(
                encode_add(
                    _resolve_address(op.lhs.name, weight_map),
                    _resolve_address(op.rhs.name, weight_map),
                    _as_2d(op.out_shape),
                    _resolve_address(op.result, weight_map),
                )
            )
        elif isinstance(op, ReluOp):
            instructions.append(
                encode_relu(
                    _resolve_address(op.src.name, weight_map),
                    _resolve_address(op.result, weight_map),
                    op.length,
                )
            )
        elif isinstance(op, EndOp):
            instructions.append(encode_end())
        elif isinstance(op, ReturnOp):
            # The result already resides at its address (0x1); return marks the
            # program output but emits no instruction.
            continue
        else:
            raise AssertionError(f"Assembler cannot handle op: {op!r}")

    return instructions


def Assemble(tmp_dir: Path | None = None) -> list[int]:
    """Assemble the TPU dialect and export PROGRAM.bin. Returns the instructions."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"

    # Assemble the bias-removed dialect: it is the final IR (bias adds dropped).
    tpu_path = tmp_dir / "optimized.nobias.tpu.mlir"
    weight_map_path = tmp_dir / "weight_map.json"
    assert tpu_path.is_file(), f"Missing {tpu_path}"
    assert weight_map_path.is_file(), f"Missing {weight_map_path}"

    weight_map = json.loads(weight_map_path.read_text())
    instructions = assemble_program(tpu_path.read_text(), weight_map)

    blob = bytearray()
    for instruction in instructions:
        blob += build_program_instruction(instruction)
    (tmp_dir / "PROGRAM.bin").write_bytes(bytes(blob))

    return instructions
