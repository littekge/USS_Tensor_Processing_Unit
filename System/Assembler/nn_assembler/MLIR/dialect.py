"""In-memory representation and textual syntax of the *Functional TPU* dialect.

The dialect is intentionally a thin bridge between StableHLO and
`Functional_TPU_ISA_v0.2.md` machine code: every operation here corresponds 1:1
to a single ISA instruction. There are no operations that are "synthesized away"
during assembly -- shape-only StableHLO ops (reshape/transpose) are folded by the
legalization pass before they ever reach this dialect, so the assembler can walk
the op list and emit one instruction per op.

The textual form is deliberately MLIR-like so it reads as a real dialect, but it
is parsed/printed by this module rather than by the MLIR C++ infrastructure (see
`src/MLIR/README.md` for why v0.1 implements the dialect in Python).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def shape_to_str(shape: list[int]) -> str:
    """Render a shape as the dialect's `RxC` (or `N`) token."""
    return "x".join(str(d) for d in shape) if shape else "1"


def parse_shape(token: str) -> list[int]:
    """Parse a dialect shape token (`1x4`, `4`, ...) into a list of ints."""
    return [int(d) for d in token.split("x")]


@dataclass
class Operand:
    """A value reference plus the dimensions it is used with at this op.

    `name` is the SSA name (e.g. `%arg0` for a weight/input, or `%3` for a prior
    result). `shape` is the shape as seen by the consuming instruction, which may
    differ from the value's original shape because reshapes were folded in. When
    the op addresses a sub-block of a larger (parent) matrix -- as produced by the
    matmul-partitioning pass -- `offset` is the row-major element offset of the
    sub-block corner within the parent tensor; the assembler adds it to the
    parent's base address. `shape` then holds the tile's logical dimensions.
    """

    name: str
    shape: list[int]
    offset: int = 0


@dataclass
class MultOp:
    """Matrix multiply -> ISA `mult` (MUL format), funct3 = MULT (0x0).

    `M0`/`n` are the dyadic requantization pair (`M ~= M0 * 2^-n`) that the
    legalization pass attaches from the weight operand's `weight_map` entry; they
    become the MUL instruction's `M0` (bits 35-28) and `n` (bits 27-20) fields.
    They are optional only so an unannotated op can round-trip; the assembler
    requires them.

    Tiling (v0.5): a `mult` may write a sub-block of a larger result. `dst_offset`
    is the row-major element offset of that sub-block within the parent result,
    and `parent_out_shape` is the parent result's full dimensions (used by the
    assembler to size its allocation once). For an un-tiled op both default to the
    whole tensor (`dst_offset = 0`, `parent_out_shape == out_shape`).
    """

    result: str
    lhs: Operand
    rhs: Operand
    out_shape: list[int]
    M0: int | None = None
    n: int | None = None
    dst_offset: int = 0
    parent_out_shape: list[int] | None = None


@dataclass
class MultipOp:
    """Accumulating matrix multiply -> ISA `multip` (MUL format), funct3 = MULTIP.

    Identical fields to `MultOp`; the only difference is that `multip` does not
    clear the internal accumulator after writeback, so a run of `multip`s
    terminated by a `mult` computes a contraction that exceeds one instruction.
    The matmul-partitioning pass emits the leading K-tiles as `multip` and the
    final K-tile as `mult`.
    """

    result: str
    lhs: Operand
    rhs: Operand
    out_shape: list[int]
    M0: int | None = None
    n: int | None = None
    dst_offset: int = 0
    parent_out_shape: list[int] | None = None


@dataclass
class StrideOp:
    """Set the leading-dimension registers -> ISA `stride` (CONFIG format).

    `ld1`/`ld2` are the physical row strides of `rs1`/`rs2` (and of the
    writeback `rd`, whose stride equals `ld2`). They persist for every subsequent
    MUL-type instruction until the next `stride`. The partitioning pass emits one
    `stride(ld1=K, ld2=N)` before the tiles of each partitioned matmul.
    """

    ld1: int
    ld2: int


@dataclass
class WindowOp:
    """Set the window descriptor -> ISA `window` (WINCONFIG / W-Format).

    Holds the 11 windowed-instruction parameters (feature-map channels/height/
    width, output height/width, window height/width, strides, and padding). It
    persists for every subsequent windowed instruction (`im2col`, `max`) until the
    next `window`. The legalizer emits one only when the descriptor differs from
    the currently live one (mirrors the v0.5 stale-`stride` handling).
    """

    chans: int
    inh: int
    inw: int
    outh: int
    outw: int
    winh: int
    winw: int
    strh: int
    strw: int
    padh: int
    padw: int


@dataclass
class Im2colOp:
    """Feature map -> column matrix -> ISA `im2col` (SHAPE / A-Format).

    Expands the `chans x inh x inw` feature map at `src` into the dense
    `(chans*winh*winw) x (outh*outw)` column matrix at `result`, using the live
    window descriptor. `out_shape` is that `[K, N]` column matrix (K channel-major
    per the ISA), used by the assembler to size the scratch allocation.
    """

    result: str
    src: Operand
    out_shape: list[int]


@dataclass
class MaxPoolOp:
    """Windowed max pooling -> ISA `max` (POOL / A-Format).

    Reduces the `chans x inh x inw` feature map at `src` to the pooled
    `chans x outh x outw` tensor at `result` using the live window descriptor
    (per-channel max over each window). `out_shape` is the pooled CHW tensor.
    """

    result: str
    src: Operand
    out_shape: list[int]


@dataclass
class MoveOp:
    """Contiguous data move -> ISA `move` (SHAPE / A-Format, funct3 = MOVE 0x1).

    Stages a computed tensor into the reserved I/O address so the programmer can
    read the network output back (v0.7 output-staging pass). The destination is a
    fixed symbolic I/O location (`@io`, resolved to 0x1 by the assembler), never a
    physical address in the IR; `len` is inferred from the source tensor's shape
    (`src.shape`), so no explicit length attribute is carried. `move` is
    dimension-agnostic -- it copies raw contiguous words.
    """

    src: Operand


@dataclass
class AddOp:
    """Elementwise add -> ISA `add` (ELEM format)."""

    result: str
    lhs: Operand
    rhs: Operand
    out_shape: list[int]


@dataclass
class ReluOp:
    """Rectified linear unit -> ISA `relu` (ACT format)."""

    result: str
    src: Operand
    length: int


@dataclass
class ReturnOp:
    """Marks the program's output value. Emits no instruction itself."""

    value: str
    shape: list[int]


@dataclass
class EndOp:
    """Program terminator -> ISA `end` (SYSTEM format)."""


@dataclass
class Program:
    """A whole TPU dialect program (one `tpu.func`)."""

    name: str
    ops: list = field(default_factory=list)


def serialize(program: Program) -> str:
    """Render a `Program` to its textual `.tpu.mlir` form."""
    lines: list[str] = [
        "// Functional TPU dialect (v0.3) -- generated by the legalization pass.",
        "// Each tpu.* op maps 1:1 to a Functional_TPU_ISA instruction.",
        f"module @{program.name} {{",
        "  tpu.func @main {",
    ]
    for op in program.ops:
        lines.append("    " + _serialize_op(op))
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _prod(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _mult_attrs(op) -> str:
    """Render the trailing `{...}` attribute dict for a mult/multip op.

    Emits `M0`/`n` when present, and the tiling attributes (`lo`/`ro`/`do` operand
    offsets, `cap` parent capacity in words) only when the op addresses a
    sub-block -- i.e. any offset is non-zero or the parent result is larger than
    this tile. Un-tiled ops keep the legacy `{M0 = .., n = ..}` form.
    """
    pairs: list[str] = []
    if op.M0 is not None and op.n is not None:
        pairs.append(f"M0 = {op.M0}")
        pairs.append(f"n = {op.n}")
    parent = op.parent_out_shape if op.parent_out_shape is not None else op.out_shape
    tiled = (
        op.lhs.offset != 0
        or op.rhs.offset != 0
        or op.dst_offset != 0
        or _prod(parent) != _prod(op.out_shape)
    )
    if tiled:
        pairs.append(f"lo = {op.lhs.offset}")
        pairs.append(f"ro = {op.rhs.offset}")
        pairs.append(f"do = {op.dst_offset}")
        pairs.append(f"cap = {_prod(parent)}")
    return f" {{{', '.join(pairs)}}}" if pairs else ""


def _serialize_op(op) -> str:
    if isinstance(op, (MultOp, MultipOp)):
        mnemonic = "multip" if isinstance(op, MultipOp) else "mult"
        return (
            f"{op.result} = tpu.{mnemonic} {op.lhs.name} : {shape_to_str(op.lhs.shape)}, "
            f"{op.rhs.name} : {shape_to_str(op.rhs.shape)} -> {shape_to_str(op.out_shape)}"
            f"{_mult_attrs(op)}"
        )
    if isinstance(op, StrideOp):
        return f"tpu.stride ld1 = {op.ld1}, ld2 = {op.ld2}"
    if isinstance(op, WindowOp):
        return (
            f"tpu.window chans = {op.chans}, inh = {op.inh}, inw = {op.inw}, "
            f"outh = {op.outh}, outw = {op.outw}, winh = {op.winh}, winw = {op.winw}, "
            f"strh = {op.strh}, strw = {op.strw}, padh = {op.padh}, padw = {op.padw}"
        )
    if isinstance(op, Im2colOp):
        return (
            f"{op.result} = tpu.im2col {op.src.name} -> {shape_to_str(op.out_shape)}"
        )
    if isinstance(op, MaxPoolOp):
        return f"{op.result} = tpu.max {op.src.name} -> {shape_to_str(op.out_shape)}"
    if isinstance(op, MoveOp):
        return f"tpu.move {op.src.name} -> @io : {shape_to_str(op.src.shape)}"
    if isinstance(op, AddOp):
        return (
            f"{op.result} = tpu.add {op.lhs.name} : {shape_to_str(op.lhs.shape)}, "
            f"{op.rhs.name} : {shape_to_str(op.rhs.shape)} -> {shape_to_str(op.out_shape)}"
        )
    if isinstance(op, ReluOp):
        return f"{op.result} = tpu.relu {op.src.name} : {op.length} -> {op.length}"
    if isinstance(op, ReturnOp):
        return f"tpu.return {op.value} : {shape_to_str(op.shape)}"
    if isinstance(op, EndOp):
        return "tpu.end"
    raise AssertionError(f"Cannot serialize unknown TPU op: {op!r}")


_SSA = r"%[\w\.]+"
_SHAPE = r"[\dx]+"

_MULT_RE = re.compile(
    rf"({_SSA})\s*=\s*tpu\.(mult|multip)\s+({_SSA})\s*:\s*({_SHAPE})\s*,\s*"
    rf"({_SSA})\s*:\s*({_SHAPE})\s*->\s*({_SHAPE})"
    rf"(?:\s*\{{([^}}]*)\}})?"
)
_STRIDE_RE = re.compile(r"tpu\.stride\s+ld1\s*=\s*(\d+)\s*,\s*ld2\s*=\s*(\d+)")
_WINDOW_RE = re.compile(
    r"tpu\.window\s+chans\s*=\s*(\d+)\s*,\s*inh\s*=\s*(\d+)\s*,\s*inw\s*=\s*(\d+)\s*,\s*"
    r"outh\s*=\s*(\d+)\s*,\s*outw\s*=\s*(\d+)\s*,\s*winh\s*=\s*(\d+)\s*,\s*winw\s*=\s*(\d+)\s*,\s*"
    r"strh\s*=\s*(\d+)\s*,\s*strw\s*=\s*(\d+)\s*,\s*padh\s*=\s*(\d+)\s*,\s*padw\s*=\s*(\d+)"
)
_IM2COL_RE = re.compile(rf"({_SSA})\s*=\s*tpu\.im2col\s+({_SSA})\s*->\s*({_SHAPE})")
_MAX_RE = re.compile(rf"({_SSA})\s*=\s*tpu\.max\s+({_SSA})\s*->\s*({_SHAPE})")
_MOVE_RE = re.compile(rf"tpu\.move\s+({_SSA})\s*->\s*@io\s*:\s*({_SHAPE})")
_ATTR_RE = re.compile(r"(\w+)\s*=\s*(-?\d+)")
_ADD_RE = re.compile(
    rf"({_SSA})\s*=\s*tpu\.add\s+({_SSA})\s*:\s*({_SHAPE})\s*,\s*"
    rf"({_SSA})\s*:\s*({_SHAPE})\s*->\s*({_SHAPE})"
)
_RELU_RE = re.compile(
    rf"({_SSA})\s*=\s*tpu\.relu\s+({_SSA})\s*:\s*(\d+)\s*->\s*(\d+)"
)
_RETURN_RE = re.compile(rf"tpu\.return\s+({_SSA})\s*:\s*({_SHAPE})")
_MODULE_RE = re.compile(r"module\s+@([\w\.]+)")


def parse_program(text: str) -> Program:
    """Parse textual `.tpu.mlir` into a `Program`.

    Lines that are comments, braces, or the `tpu.func` header are skipped; every
    other non-blank line must match a known op or it is a hard error -- silently
    dropping an unrecognized op would produce a program that does not match the
    source graph.
    """
    module_match = _MODULE_RE.search(text)
    name = module_match.group(1) if module_match else "module"
    program = Program(name=name)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("module") or line.startswith("tpu.func") or line in {"}", "{"}:
            continue

        op = _parse_op(line)
        assert op is not None, f"Unrecognized TPU dialect line: {raw_line!r}"
        program.ops.append(op)

    return program


def _parse_op(line: str):
    m = _MULT_RE.match(line)
    if m:
        res, mnemonic, a, a_sh, b, b_sh, out, attr_blob = m.groups()
        attrs = dict(_ATTR_RE.findall(attr_blob)) if attr_blob else {}
        out_shape = parse_shape(out)
        cap = int(attrs["cap"]) if "cap" in attrs else None
        parent_out_shape = [cap] if cap is not None else None
        cls = MultipOp if mnemonic == "multip" else MultOp
        return cls(
            res,
            Operand(a, parse_shape(a_sh), offset=int(attrs.get("lo", 0))),
            Operand(b, parse_shape(b_sh), offset=int(attrs.get("ro", 0))),
            out_shape,
            M0=int(attrs["M0"]) if "M0" in attrs else None,
            n=int(attrs["n"]) if "n" in attrs else None,
            dst_offset=int(attrs.get("do", 0)),
            parent_out_shape=parent_out_shape,
        )

    m = _STRIDE_RE.match(line)
    if m:
        ld1, ld2 = m.groups()
        return StrideOp(int(ld1), int(ld2))

    m = _WINDOW_RE.match(line)
    if m:
        chans, inh, inw, outh, outw, winh, winw, strh, strw, padh, padw = (int(g) for g in m.groups())
        return WindowOp(chans, inh, inw, outh, outw, winh, winw, strh, strw, padh, padw)

    m = _IM2COL_RE.match(line)
    if m:
        res, src, out = m.groups()
        return Im2colOp(res, Operand(src, []), parse_shape(out))

    m = _MAX_RE.match(line)
    if m:
        res, src, out = m.groups()
        return MaxPoolOp(res, Operand(src, []), parse_shape(out))

    m = _MOVE_RE.match(line)
    if m:
        src, shape = m.groups()
        return MoveOp(Operand(src, parse_shape(shape)))

    m = _ADD_RE.match(line)
    if m:
        res, a, a_sh, b, b_sh, out = m.groups()
        return AddOp(res, Operand(a, parse_shape(a_sh)), Operand(b, parse_shape(b_sh)), parse_shape(out))

    m = _RELU_RE.match(line)
    if m:
        res, src, in_len, out_len = m.groups()
        assert in_len == out_len, f"relu length mismatch on line: {line!r}"
        return ReluOp(res, Operand(src, [int(in_len)]), int(in_len))

    m = _RETURN_RE.match(line)
    if m:
        val, sh = m.groups()
        return ReturnOp(val, parse_shape(sh))

    if line == "tpu.end":
        return EndOp()

    return None
