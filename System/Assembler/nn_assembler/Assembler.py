"""Second half of lowering step 3 (Steps 6-7): the assembler.

Translates the final *Functional TPU* dialect
(`/tmp/optimized.staged.tpu.mlir`, the bias-removed, matmul-partitioned, and
output-staged IR) into `Functional_TPU_ISA.md` machine code, then exports it in
the Message Protocol PROGRAM format (`/tmp/PROGRAM.bin`). Each
`tpu.mult`/`tpu.multip` carries the dyadic requantization pair `M0`/`n` set
during legalization; a partitioned matmul is a `stride` followed by array-sized
`mult`/`multip` tiles.

Address resolution uses `/tmp/weight_map.json`: mapped weights resolve to their
assigned addresses. The network input occupies the reserved I/O address 0x1.
Every op result -- including the network's final output -- is allocated in main
data memory (0x2+) by an `IntermediateAllocator`, with freed regions reused
across the linear op sequence. A final `tpu.move` stages the output into 0x1 so
the programmer can read it back (v0.7 output staging); this keeps tiled output
writes contiguous in main memory instead of spilling past 0x1 into weights.
"""

from __future__ import annotations

import json
from pathlib import Path

from .MLIR.dialect import (
    AddOp,
    EndOp,
    Im2colOp,
    MaxPoolOp,
    MoveOp,
    MultipOp,
    MultOp,
    Operand,
    ReluOp,
    ReturnOp,
    StrideOp,
    WindowOp,
    parse_program,
)
from .Process_Weights import (
    INPUT_ADDRESS,
    IntermediateAllocator,
    first_free_address,
)
from .Protocol import build_program_instruction

# Opcodes (bits 127-124) and funct codes from the ISA's Instruction Set Listing.
OPCODE_MULT = 0b1000
OPCODE_RELU = 0b1001
OPCODE_ADD = 0b1010
OPCODE_MAX = 0b1100  # POOL
OPCODE_IM2COL = 0b1101  # SHAPE
OPCODE_WINDOW = 0b1110  # WINCONFIG
OPCODE_STRIDE = 0b1111
OPCODE_SYSTEM = 0b0000
FUNCT3_DEFAULT = 0x0
FUNCT3_MULT = 0x0
FUNCT3_MULTIP = 0x1  # MULTIP shares the MUL layout; funct3 selects accumulate-keep.
FUNCT3_STRIDE = 0x0
FUNCT3_WINDOW = 0x0
FUNCT3_IM2COL = 0x0
FUNCT3_MOVE = 0x1  # `move` shares the SHAPE opcode (1101) with `im2col` (funct3 0x0).
FUNCT3_MAX = 0x0
FUNCT7_END = 0x0

# Systolic-array edge length. Every MUL/MULTIP tile must fit within the array, so
# the partitioning pass decomposes larger matmuls into tiles of at most this many
# elements per dimension. This MUST match the TPU `Build Parameters` N (N = 8);
# the two are a coupled hardware/software contract.
MAX_MATMUL_SIZE = 8

MUL_MAX_DIM = MAX_MATMUL_SIZE  # MUL tile dims are bounded by the array, not the 4-bit field.
ELEM_MAX_DIM = 255  # ELEM format dimension fields are 8 bits.
LD_FIELD_MAX = (1 << 24) - 1  # CONFIG im1/im2 (ld1/ld2) are 24-bit fields.
REQUANT_FIELD_MAX = 255  # MUL M0/n fields are each 8 bits (bits 35-28 / 27-20).
ADDRESS_FIELD_MAX = (1 << 24) - 1  # rs1/rs2/rd are 24-bit address fields (v0.4).
RELU_LEN_MAX = (1 << 16) - 1  # ACT `len` field is 16 bits (bits 75-60).
MOVE_LEN_MAX = (1 << 16) - 1  # SHAPE `move` aux (len) field is 16 bits (bits 75-60).
WINDOW_DIM_MAX = (1 << 12) - 1  # W-Format inh/inw/chans/outh/outw are 12-bit fields.
WINDOW_WIN_MAX = (1 << 4) - 1  # W-Format winh/winw/strh/strw/padh/padw are 4-bit fields.


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
    rs1: int,
    lhs: tuple[int, int],
    rs2: int,
    rhs: tuple[int, int],
    rd: int,
    M0: int,
    n: int,
    funct3: int = FUNCT3_MULT,
) -> int:
    """Encode a MUL-format `mult`/`multip` instruction into its 128-bit value.

    Field layout (MSB 127 left, ISA v0.5): opcode 127-124; rs1 123-100; sz11 99-96;
    sz12 95-92; rs2 91-68; sz21 67-64; sz22 63-60; rd 59-36; M0 35-28; n 27-20;
    reserved 19-3; funct3 2-0. `mult` and `multip` share this layout and differ
    only in `funct3` (MULT = 0x0, MULTIP = 0x1). `M0`/`n` are the dyadic
    requantization pair. Tile dims are bounded by `MAX_MATMUL_SIZE` (the array).
    """
    sz11, sz12 = lhs
    sz21, sz22 = rhs
    for field, addr in (("rs1", rs1), ("rs2", rs2), ("rd", rd)):
        _assert_address(field, addr)
    for dim in (sz11, sz12, sz21, sz22):
        assert 0 <= dim <= MUL_MAX_DIM, (
            f"mult tile dimension {dim} exceeds MAX_MATMUL_SIZE ({MUL_MAX_DIM}); "
            "the partitioning pass should have tiled it to fit the array."
        )
    assert 0 <= M0 <= REQUANT_FIELD_MAX, f"mult M0 {M0} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    assert 0 <= n <= REQUANT_FIELD_MAX, f"mult n {n} exceeds 8-bit field (max {REQUANT_FIELD_MAX})."
    assert funct3 in (FUNCT3_MULT, FUNCT3_MULTIP), f"mult funct3 {funct3} is not MULT/MULTIP."
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
        | funct3
    )


def encode_mult_in_place(
    rs1: int, lhs: tuple[int, int], rs2: int, rhs: tuple[int, int], rd: int, M0: int, n: int
) -> int:
    """Encode a MUL-format `multip` (accumulate-keep) instruction (funct3 = 0x1).

    Reuses the exact `mult` bit layout; only `funct3` differs, so a run of
    `multip`s terminated by a `mult` accumulates a contraction in the array.
    """
    return encode_mult(rs1, lhs, rs2, rhs, rd, M0, n, funct3=FUNCT3_MULTIP)


def encode_stride(ld1: int, ld2: int) -> int:
    """Encode a CONFIG-format `stride` instruction into its 128-bit value.

    Field layout (ISA v0.5): opcode 127-124 = 1111; im1 123-100 = ld1;
    im2 99-76 = ld2; reserved 75-3; funct3 2-0 = STRIDE (0x0).
    """
    for field, value in (("ld1", ld1), ("ld2", ld2)):
        assert 0 <= value <= LD_FIELD_MAX, (
            f"stride {field} {value} exceeds the 24-bit CONFIG immediate (max {LD_FIELD_MAX})."
        )
    return (OPCODE_STRIDE << 124) | (ld1 << 100) | (ld2 << 76) | FUNCT3_STRIDE


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


def encode_window(
    chans: int,
    inh: int,
    inw: int,
    outh: int,
    outw: int,
    winh: int,
    winw: int,
    strh: int,
    strw: int,
    padh: int,
    padw: int,
) -> int:
    """Encode a WINCONFIG-format `window` instruction into its 128-bit value.

    Field layout (ISA v0.6, W-Format): opcode 127-124 = 1110; inh 123-112;
    inw 111-100; chans 99-88; outh 87-76; outw 75-64; winh 63-60; winw 59-56;
    strh 55-52; strw 51-48; padh 47-44; padw 43-40; reserved 39-3; funct3 2-0
    = WINDOW (0x0). The five feature-map/output dims are 12-bit; the six
    window/stride/pad fields are 4-bit.
    """
    for field, value in (("inh", inh), ("inw", inw), ("chans", chans), ("outh", outh), ("outw", outw)):
        assert 0 <= value <= WINDOW_DIM_MAX, f"window {field} {value} exceeds 12-bit field (max {WINDOW_DIM_MAX})."
    for field, value in (("winh", winh), ("winw", winw), ("strh", strh), ("strw", strw), ("padh", padh), ("padw", padw)):
        assert 0 <= value <= WINDOW_WIN_MAX, f"window {field} {value} exceeds 4-bit field (max {WINDOW_WIN_MAX})."
    return (
        (OPCODE_WINDOW << 124)
        | (inh << 112)
        | (inw << 100)
        | (chans << 88)
        | (outh << 76)
        | (outw << 64)
        | (winh << 60)
        | (winw << 56)
        | (strh << 52)
        | (strw << 48)
        | (padh << 44)
        | (padw << 40)
        | FUNCT3_WINDOW
    )


def _encode_shape_pool(opcode: int, funct3: int, rs1: int, rd: int) -> int:
    """Encode an A-Format SHAPE/POOL instruction (`im2col`/`max`).

    Field layout (ISA v0.6, A-Format): opcode 127-124; rs1 123-100; rd 99-76;
    aux 75-60 = 0; reserved 59-3; funct3 2-0. `im2col`/`max` read the live window
    descriptor, so the only variable fields are the src (`rs1`) and dest (`rd`)
    addresses; `aux` and reserved are zero.
    """
    for field, addr in (("rs1", rs1), ("rd", rd)):
        _assert_address(field, addr)
    return (opcode << 124) | (rs1 << 100) | (rd << 76) | funct3


def encode_im2col(rs1: int, rd: int) -> int:
    """Encode a SHAPE-format `im2col` (opcode 1101): src `rs1`, dest `rd`, aux 0."""
    return _encode_shape_pool(OPCODE_IM2COL, FUNCT3_IM2COL, rs1, rd)


def encode_max(rs1: int, rd: int) -> int:
    """Encode a POOL-format `max` (opcode 1100): src `rs1`, dest `rd`, aux 0."""
    return _encode_shape_pool(OPCODE_MAX, FUNCT3_MAX, rs1, rd)


def encode_move(rs1: int, rd: int, length: int) -> int:
    """Encode a SHAPE-format `move` instruction into its 128-bit value (ISA v0.7).

    Field layout (A-Format): opcode 127-124 = 1101 (SHAPE, shared with `im2col`);
    rs1 123-100 = src; rd 99-76 = dest; aux 75-60 = len; reserved 59-3; funct3 2-0
    = MOVE (0x1). Mirrors `encode_im2col` but carries a non-zero `aux` (the element
    count) and selects the MOVE function. A single `move` relocates up to 65535
    contiguous words (the 16-bit `aux` field).
    """
    for field, addr in (("rs1", rs1), ("rd", rd)):
        _assert_address(field, addr)
    assert 0 <= length <= MOVE_LEN_MAX, (
        f"move length {length} exceeds the 16-bit aux field (max {MOVE_LEN_MAX}); "
        "multi-move chunking is out of scope for v0.7."
    )
    return (
        (OPCODE_IM2COL << 124) | (rs1 << 100) | (rd << 76) | (length << 60) | FUNCT3_MOVE
    )


def encode_end() -> int:
    """Encode a SYSTEM-format `end` instruction into its 128-bit value."""
    return (OPCODE_SYSTEM << 124) | FUNCT7_END


def _prod(shape: list[int]) -> int:
    """Flattened element count (memory words) of a tensor of any order."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def _operand_names(op) -> list[str]:
    """SSA names an op reads (used to reclaim dead intermediate regions)."""
    if isinstance(op, (MultOp, MultipOp, AddOp)):
        return [op.lhs.name, op.rhs.name]
    if isinstance(op, (ReluOp, Im2colOp, MaxPoolOp, MoveOp)):
        return [op.src.name]
    return []


def _result_words(op) -> int:
    """Number of memory words an op's result tensor occupies.

    A partitioned matmul tile writes a sub-block of a larger parent result; the
    whole parent must be allocated, so its `parent_out_shape` capacity is used when
    set. Windowed ops (im2col column matrix, pooled CHW tensor) can be higher-order,
    so their whole flattened shape is used.
    """
    if isinstance(op, ReluOp):
        return op.length
    if isinstance(op, (MultOp, MultipOp)) and op.parent_out_shape is not None:
        return _prod(op.parent_out_shape)
    return _prod(op.out_shape)


def assemble_program(tpu_text: str, weight_map: dict) -> list[int]:
    """Assemble TPU dialect text into a list of 128-bit machine instructions.

    Intermediate results (including the network output) are placed in main data
    memory past the weight region and their regions are reused once dead. The
    network output is staged into the reserved I/O address 0x1 by an explicit
    `tpu.move` (v0.7 output staging); the assembler no longer pins the returned
    value to 0x1, so tiled output writes stay contiguous and never alias weights.
    """
    program = parse_program(tpu_text)

    allocator = IntermediateAllocator(first_free_address(weight_map))
    last_use: dict[str, int] = {}
    for index, op in enumerate(program.ops):
        # MultipOp is excluded: partitioned tiles share a result whose last read is
        # the terminating MultOp, so counting the intervening multips would free a
        # source region early. MoveOp is included so the staged output region stays
        # live until the move reads it.
        if isinstance(op, (MultOp, AddOp, ReluOp, Im2colOp, MaxPoolOp, MoveOp)):
            for name in _operand_names(op):
                last_use[name] = index

    intermediate_addr: dict[str, int] = {}
    intermediate_size: dict[str, int] = {}

    def resolve(name: str) -> int:
        if name in intermediate_addr:
            return intermediate_addr[name]
        return _resolve_address(name, weight_map)

    def assign_result(op) -> int:
        # Every op result -- including the network output -- lands in main data
        # memory; the output is later staged into 0x1 by a MoveOp. Allocate before
        # freeing this op's dead sources so a reused region can never alias a source
        # still being read. A partitioned matmul emits several tiles that share one
        # result name; the parent region is allocated once (on the first tile) and
        # reused by later tiles addressing it at their own sub-block offsets.
        if op.result in intermediate_addr:
            return intermediate_addr[op.result]
        words = _result_words(op)
        address = allocator.allocate(words)
        intermediate_addr[op.result] = address
        intermediate_size[op.result] = words
        return address

    def release_dead_sources(op, index: int) -> None:
        for name in _operand_names(op):
            if name in intermediate_addr and last_use.get(name) == index:
                allocator.free(intermediate_addr.pop(name), intermediate_size.pop(name))

    instructions: list[int] = []
    for index, op in enumerate(program.ops):
        if isinstance(op, (MultOp, MultipOp)):
            mnemonic = "multip" if isinstance(op, MultipOp) else "mult"
            assert op.M0 is not None and op.n is not None, (
                f"{mnemonic} {op.result} is missing its M0/n requantization pair; "
                "legalization should have annotated it from weight_map.json."
            )
            funct3 = FUNCT3_MULTIP if isinstance(op, MultipOp) else FUNCT3_MULT
            rd = assign_result(op)
            instructions.append(
                encode_mult(
                    resolve(op.lhs.name) + op.lhs.offset,
                    _as_2d(op.lhs.shape),
                    resolve(op.rhs.name) + op.rhs.offset,
                    _as_2d(op.rhs.shape),
                    rd + op.dst_offset,
                    op.M0,
                    op.n,
                    funct3=funct3,
                )
            )
            release_dead_sources(op, index)
        elif isinstance(op, StrideOp):
            instructions.append(encode_stride(op.ld1, op.ld2))
        elif isinstance(op, WindowOp):
            instructions.append(
                encode_window(
                    op.chans, op.inh, op.inw, op.outh, op.outw,
                    op.winh, op.winw, op.strh, op.strw, op.padh, op.padw,
                )
            )
        elif isinstance(op, Im2colOp):
            rd = assign_result(op)
            instructions.append(encode_im2col(resolve(op.src.name), rd))
            release_dead_sources(op, index)
        elif isinstance(op, MaxPoolOp):
            rd = assign_result(op)
            instructions.append(encode_max(resolve(op.src.name), rd))
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
        elif isinstance(op, MoveOp):
            # Stage the network output into the reserved I/O address for readback.
            # The source resolves to its main-memory base; dest is fixed at 0x1.
            instructions.append(
                encode_move(resolve(op.src.name), INPUT_ADDRESS, _prod(op.src.shape))
            )
            release_dead_sources(op, index)
        elif isinstance(op, EndOp):
            instructions.append(encode_end())
        elif isinstance(op, ReturnOp):
            # The output has already been staged into 0x1 by the preceding MoveOp;
            # return marks the program output but emits no instruction.
            continue
        else:
            raise AssertionError(f"Assembler cannot handle op: {op!r}")

    return instructions


def Assemble(tmp_dir: Path | None = None) -> list[int]:
    """Assemble the TPU dialect and export PROGRAM.bin. Returns the instructions."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"

    # Assemble the staged dialect: it is the final IR (bias adds dropped, oversized
    # matmuls tiled into array-sized mult/multip runs with strides, and the network
    # output followed by a `move` that stages it into the I/O address 0x1).
    tpu_path = tmp_dir / "optimized.staged.tpu.mlir"
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
