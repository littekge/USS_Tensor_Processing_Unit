"""Second half of lowering step 3 (Steps 6-7): the assembler.

Translates the final *Functional TPU* dialect (`/tmp/optimized.nobias.tpu.mlir`,
the bias-removed IR) into `Functional_TPU_ISA.md` machine code, then exports it
in the Message Protocol PROGRAM format (`/tmp/PROGRAM.bin`). Each `tpu.mult`
carries the dyadic requantization pair `M0`/`n` set during legalization.

Address resolution uses `/tmp/weight_map.json`: mapped weights resolve to their
assigned addresses. The network input occupies the reserved I/O address 0x1, and
the network's final output is written back to 0x1 so the programmer can read it.
Every other intermediate result is allocated in main data memory (0x2+) by an
`IntermediateAllocator`, with freed regions reused across the linear op sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

from .MLIR.dialect import AddOp, EndOp, MultOp, Operand, ReluOp, ReturnOp, parse_program
from .Process_Weights import (
    FINAL_OUTPUT_ADDRESS,
    INPUT_ADDRESS,
    IntermediateAllocator,
    first_free_address,
)
from .Protocol import build_program_instruction

# Opcodes (bits 127-124) and funct codes from the ISA's Instruction Set Listing.
OPCODE_MULT = 0b1000
OPCODE_RELU = 0b1001
OPCODE_ADD = 0b1010
OPCODE_SYSTEM = 0b0000
FUNCT3_DEFAULT = 0x0
FUNCT7_END = 0x0

MUL_MAX_DIM = 15  # MUL format dimension fields are 4 bits.
ELEM_MAX_DIM = 255  # ELEM format dimension fields are 8 bits.
REQUANT_FIELD_MAX = 255  # MUL M0/n fields are each 8 bits (bits 35-28 / 27-20).
ADDRESS_FIELD_MAX = (1 << 24) - 1  # rs1/rs2/rd are 24-bit address fields (v0.4).
RELU_LEN_MAX = (1 << 16) - 1  # ACT `len` field is 16 bits (bits 75-60).


def _as_2d(shape: list[int]) -> tuple[int, int]:
    """Normalize a shape to (rows, cols); a 1-D tensor is a single row."""
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 1:
        return 1, shape[0]
    if len(shape) == 0:
        return 1, 1
    raise AssertionError(f"Tensors of order > 2 are not supported by the ISA: shape {shape}")


def _assert_address(field: str, value: int) -> None:
    """Guard a 24-bit ISA address field (rs1/rs2/rd) against overflow."""
    assert 0 <= value <= ADDRESS_FIELD_MAX, (
        f"{field} address {value} exceeds the 24-bit ISA field (max {ADDRESS_FIELD_MAX})."
    )


def _resolve_address(name: str, weight_map: dict) -> int:
    """Map an SSA name to an ISA memory address using the weight map.

    Weights and biases resolve to their assigned addresses; the network input
    (kind == "input") resolves to its reserved I/O address (0x1). Intermediate
    op results are not in the map -- the assembler resolves those from the
    per-program allocator before calling this helper.
    """
    entry = weight_map.get(name)
    if entry is not None and entry["kind"] in ("weight", "bias"):
        return entry["address"]
    return INPUT_ADDRESS


def encode_mult(
    rs1: int, lhs: tuple[int, int], rs2: int, rhs: tuple[int, int], rd: int, M0: int, n: int
) -> int:
    """Encode a MUL-format `mult` instruction into its 128-bit value (ISA v0.4).

    Field layout (MSB 127 left): opcode 127-124; rs1 123-100; sz11 99-96;
    sz12 95-92; rs2 91-68; sz21 67-64; sz22 63-60; rd 59-36; M0 35-28; n 27-20;
    reserved 19-3; funct3 2-0. `M0`/`n` are the dyadic requantization pair.
    """
    sz11, sz12 = lhs
    sz21, sz22 = rhs
    for field, addr in (("rs1", rs1), ("rs2", rs2), ("rd", rd)):
        _assert_address(field, addr)
    for dim in (sz11, sz12, sz21, sz22):
        assert 0 <= dim <= MUL_MAX_DIM, f"mult dimension {dim} exceeds 4-bit field (max {MUL_MAX_DIM})."
    assert 0 <= M0 <= REQUANT_FIELD_MAX, f"mult M0 {M0} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    assert 0 <= n <= REQUANT_FIELD_MAX, f"mult n {n} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    return (
        (OPCODE_MULT << 124)
        | (rs1 << 100)
        | (sz11 << 96)
        | (sz12 << 92)
        | (rs2 << 68)
        | (sz21 << 64)
        | (sz22 << 60)
        | (rd << 36)
        | (M0 << 28)
        | (n << 20)
        | FUNCT3_DEFAULT
    )


def encode_add(rs1: int, rs2: int, dims: tuple[int, int], rd: int) -> int:
    """Encode an ELEM-format `add` instruction into its 128-bit value (ISA v0.4).

    Field layout: opcode 127-124; rs1 123-100; rs2 99-76; sz1 75-68; sz2 67-60;
    rd 59-36; reserved 35-3; funct3 2-0.
    """
    sz1, sz2 = dims
    for field, addr in (("rs1", rs1), ("rs2", rs2), ("rd", rd)):
        _assert_address(field, addr)
    for dim in (sz1, sz2):
        assert 0 <= dim <= ELEM_MAX_DIM, f"add dimension {dim} exceeds 8-bit field (max {ELEM_MAX_DIM})."
    return (
        (OPCODE_ADD << 124)
        | (rs1 << 100)
        | (rs2 << 76)
        | (sz1 << 68)
        | (sz2 << 60)
        | (rd << 36)
        | FUNCT3_DEFAULT
    )


def encode_relu(rs1: int, rd: int, length: int) -> int:
    """Encode an ACT-format `relu` instruction into its 128-bit value (ISA v0.4).

    Field layout: opcode 127-124; rs1 123-100; rd 99-76; len 75-60;
    reserved 59-3; funct3 2-0.
    """
    for field, addr in (("rs1", rs1), ("rd", rd)):
        _assert_address(field, addr)
    assert 0 <= length <= RELU_LEN_MAX, f"relu length {length} exceeds 16-bit field."
    return (OPCODE_RELU << 124) | (rs1 << 100) | (rd << 76) | (length << 60) | FUNCT3_DEFAULT


def encode_end() -> int:
    """Encode a SYSTEM-format `end` instruction into its 128-bit value."""
    return (OPCODE_SYSTEM << 124) | FUNCT7_END


def _operand_names(op: MultOp | AddOp | ReluOp) -> list[str]:
    """SSA names an op reads (used to reclaim dead intermediate regions)."""
    if isinstance(op, MultOp):
        return [op.lhs.name, op.rhs.name]
    if isinstance(op, AddOp):
        return [op.lhs.name, op.rhs.name]
    return [op.src.name]


def _result_words(op: MultOp | AddOp | ReluOp) -> int:
    """Number of memory words an op's result tensor occupies."""
    if isinstance(op, ReluOp):
        return op.length
    rows, cols = _as_2d(op.out_shape)
    return rows * cols


def assemble_program(tpu_text: str, weight_map: dict) -> list[int]:
    """Assemble TPU dialect text into a list of 128-bit machine instructions.

    Intermediate results are placed in main data memory past the weight region
    and their regions are reused once dead; the value returned by `tpu.return`
    is written to the reserved I/O address 0x1 for readback.
    """
    program = parse_program(tpu_text)
    returned = next((op.value for op in program.ops if isinstance(op, ReturnOp)), None)

    allocator = IntermediateAllocator(first_free_address(weight_map))
    last_use: dict[str, int] = {}
    for index, op in enumerate(program.ops):
        if isinstance(op, (MultOp, AddOp, ReluOp)):
            for name in _operand_names(op):
                last_use[name] = index

    intermediate_addr: dict[str, int] = {}
    intermediate_size: dict[str, int] = {}

    def resolve(name: str) -> int:
        if name in intermediate_addr:
            return intermediate_addr[name]
        return _resolve_address(name, weight_map)

    def assign_result(op: MultOp | AddOp | ReluOp) -> int:
        # The returned value is the network output: pin it to the I/O address so
        # the programmer can read it back. Allocate before freeing this op's dead
        # sources so a reused region can never alias a source still being read.
        if op.result == returned:
            return FINAL_OUTPUT_ADDRESS
        words = _result_words(op)
        address = allocator.allocate(words)
        intermediate_addr[op.result] = address
        intermediate_size[op.result] = words
        return address

    def release_dead_sources(op: MultOp | AddOp | ReluOp, index: int) -> None:
        for name in _operand_names(op):
            if name in intermediate_addr and last_use.get(name) == index:
                allocator.free(intermediate_addr.pop(name), intermediate_size.pop(name))

    instructions: list[int] = []
    for index, op in enumerate(program.ops):
        if isinstance(op, MultOp):
            assert op.M0 is not None and op.n is not None, (
                f"mult {op.result} is missing its M0/n requantization pair; "
                "legalization should have annotated it from weight_map.json."
            )
            rd = assign_result(op)
            instructions.append(
                encode_mult(
                    resolve(op.lhs.name),
                    _as_2d(op.lhs.shape),
                    resolve(op.rhs.name),
                    _as_2d(op.rhs.shape),
                    rd,
                    op.M0,
                    op.n,
                )
            )
            release_dead_sources(op, index)
        elif isinstance(op, AddOp):
            rd = assign_result(op)
            instructions.append(
                encode_add(resolve(op.lhs.name), resolve(op.rhs.name), _as_2d(op.out_shape), rd)
            )
            release_dead_sources(op, index)
        elif isinstance(op, ReluOp):
            rd = assign_result(op)
            instructions.append(encode_relu(resolve(op.src.name), rd, op.length))
            release_dead_sources(op, index)
        elif isinstance(op, EndOp):
            instructions.append(encode_end())
        elif isinstance(op, ReturnOp):
            # The result already resides at 0x1; return marks the program output
            # but emits no instruction.
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
