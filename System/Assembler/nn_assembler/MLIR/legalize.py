"""Legalization pass: StableHLO MLIR -> *Functional TPU* dialect.

This is the second MLIR pass in the lowering pipeline (it runs on the output of
the StableHLO optimization pass). It lowers the `@main` computation graph op by
op; the ops it handles cover the linear nets (v0.1-v0.5) and, as of v0.6, the
convolutional LeNet-5 subset:

  * `stablehlo.reshape`        -- folded away (shape-only; no data movement)
  * `stablehlo.broadcast_in_dim` -- folded away (only appears on the bias path,
                                    which bias removal drops downstream)
  * `stablehlo.transpose`      -- folded (weight transposed offline) or rejected
  * `stablehlo.dot_general`    -- lowered to `tpu.mult`
  * `stablehlo.add`            -- lowered to `tpu.add`
  * `stablehlo.convolution`    -- lowered to `tpu.window` + `tpu.im2col` + `tpu.mult`
  * `stablehlo.reduce_window`  -- (max body) lowered to `tpu.window` + `tpu.max`
  * `call @relu*`              -- lowered to `tpu.relu`
  * `stablehlo.constant`       -- ignored (only the reduce_window -inf init)
  * `return`                   -- marks the program output

Reshapes/broadcasts/resolved transposes are folded rather than emitted: they move
no data the device cares about (tensors live as flat row-major arrays per the
ISA), so the only thing they change is the shape a *consumer* sees. Every other
op in `@main` that is not recognized is a hard error (hardening) -- silently
dropping an op would produce a program that does not match the source graph.

Convolution lowers to on-device im2col + matmul: the `chans x inh x inw` feature
map is expanded into a `(chans*kh*kw) x (outh*outw)` column matrix (K
channel-major, matching the conv weight flatten), then multiplied by the weight
`(out_ch) x (chans*kh*kw)`. The matmul is tiled later by the partitioning pass.
A `window` descriptor is emitted before each windowed op only when it differs
from the currently live one (the descriptor persists until reset, mirroring the
v0.5 stale-`stride` handling).
"""

from __future__ import annotations

import re
from pathlib import Path

from .dialect import (
    AddOp,
    EndOp,
    Im2colOp,
    MaxPoolOp,
    MultOp,
    Operand,
    Program,
    ReluOp,
    ReturnOp,
    WindowOp,
    serialize,
)

# The only convolution layout supported (LeNet-5 subset): NCHW feature maps,
# OIHW kernels, NCHW output. Anything else asserts rather than lowering wrong.
_CONV_DIM_NUMBERS = "[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]"


def _tensor_shape(type_body: str) -> list[int]:
    """Turn the inside of a `tensor<...>` type into a list of dimensions.

    `4x1xf32` -> [4, 1]; `4xf32` -> [4]; `f32` (scalar) -> [].
    """
    tokens = type_body.split("x")
    dims = tokens[:-1]  # last token is the element type (e.g. f32)
    return [int(d) for d in dims]


def _prod(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


_RESHAPE_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.reshape\s+(%[\w]+)\s*:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_BROADCAST_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.broadcast_in_dim\s+(%[\w]+).*?:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_TRANSPOSE_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.transpose\s+(%[\w]+)\s*,\s*dims\s*=\s*\[[\d,\s]*\]\s*:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_DOT_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.dot_general\s+(%[\w]+)\s*,\s*(%[\w]+).*?:\s*"
    r"\(tensor<([^>]*)>\s*,\s*tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_ADD_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.add\s+(%[\w]+)\s*,\s*(%[\w]+)\s*:\s*tensor<([^>]*)>"
)
_CONV_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.convolution\((%[\w]+),\s*(%[\w]+)\)\s*"
    r"dim_numbers\s*=\s*(\[[^\]]*\]x\[[^\]]*\]->\[[^\]]*\])\s*,\s*"
    r"window\s*=\s*\{([^}]*)\}\s*\{([^}]*)\}\s*:\s*"
    r"\(tensor<([^>]*)>\s*,\s*tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_CALL_RE = re.compile(
    r"(%[\w]+)\s*=\s*call\s+@(\w+)\((%[\w]+)\)\s*:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_CONSTANT_RE = re.compile(r"(%[\w]+)\s*=\s*stablehlo\.constant\b")
_RETURN_RE = re.compile(r"return\s+(%[\w]+)\s*:")
_MODULE_RE = re.compile(r"module\s+@([\w\.]+)")

# Multi-line reduce_window op; the region body carries the reduction (maximum).
_REDUCE_WINDOW_RE = re.compile(
    r"(%[\w]+)\s*=\s*\"stablehlo\.reduce_window\"\(\s*(%[\w]+)\s*,\s*(%[\w]+)\s*\)\s*"
    r"<\{([^}]*)\}>\s*\(\{(.*?)\}\)\s*:\s*"
    r"\(tensor<([^>]*)>\s*,\s*tensor<[^>]*>\)\s*->\s*tensor<([^>]*)>",
    re.DOTALL,
)
# Placeholder emitted by _collapse_reduce_windows so the op fits the line parser.
_MAXPOOL_RE = re.compile(
    r"(%[\w]+)\s*=\s*tpu_maxpool\s+(%[\w]+)\s+wd=([\d,]+)\s+ws=([\d,]+)\s+in=(\S+)\s+out=(\S+)"
)
# A `%r = stablehlo.<op>` or `%r = call @f` line the loop must have handled.
_UNHANDLED_RE = re.compile(r"%[\w]+\s*=\s*(?:stablehlo\.|call\s+@)")


def _find_functions(text: str) -> list[tuple[str, str, str]]:
    """Return `(visibility, name, body)` for each `func.func` in the module.

    Uses paren/brace matching (not a single regex) because arg lists contain
    nested parens (`loc(...)`) and the return spec contains a `{...}` attribute
    dict, so the body-opening brace cannot be found by naive matching.
    """
    functions: list[tuple[str, str, str]] = []
    for match in re.finditer(r"func\.func\s+(public|private)\s+@(\w+)\s*\(", text):
        visibility, name = match.group(1), match.group(2)

        # Skip past the balanced argument-list parentheses.
        index = match.end() - 1  # position of the opening '('
        depth = 0
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            index += 1

        # The body '{' is the first brace at paren-depth 0 (past the -> return spec).
        paren_depth = 0
        while index < len(text):
            char = text[index]
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "{" and paren_depth == 0:
                break
            index += 1
        body_start = index

        # Brace-match the body.
        brace_depth = 0
        end = body_start
        while end < len(text):
            char = text[end]
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    break
            end += 1

        functions.append((visibility, name, text[body_start + 1 : end]))
    return functions


def _relu_function_names(functions: list[tuple[str, str, str]]) -> set[str]:
    """Names of private functions that implement ReLU (`maximum(x, 0)`).

    The StableHLO optimizer leaves ReLU as a private helper that maximizes its
    argument against a zero constant; a `call` to such a function is a ReLU.
    """
    names: set[str] = set()
    for visibility, name, body in functions:
        if visibility == "private" and "stablehlo.maximum" in body and "0.000000e+00" in body:
            names.add(name)
    return names


def _int_list(attr_body: str, key: str) -> list[int]:
    """Extract `key = array<i64: a, b, ...>` from a reduce_window attribute body."""
    match = re.search(rf"{key}\s*=\s*array<i64:\s*([\d,\s]*)>", attr_body)
    assert match is not None, f"reduce_window attribute missing {key}: {attr_body!r}"
    return [int(v) for v in match.group(1).split(",") if v.strip()]


def _collapse_reduce_windows(body: str) -> str:
    """Rewrite each multi-line reduce_window op into a single `tpu_maxpool` line.

    This lets the op be processed by the same line-oriented loop as every other
    op while asserting the LeNet-5 pooling subset (max reduction, no padding, no
    dilation).
    """

    def repl(match: re.Match) -> str:
        result, inp, _init, attr, region, in_ty, out_ty = match.groups()
        assert "stablehlo.maximum" in region, (
            f"Only max reduce_window is supported; body was {region!r}."
        )
        for unsupported in ("padding", "dilation"):
            assert unsupported not in attr, (
                f"reduce_window with {unsupported} is out of scope (LeNet-5 subset)."
            )
        wd = _int_list(attr, "window_dimensions")
        ws = _int_list(attr, "window_strides")
        return (
            f"{result} = tpu_maxpool {inp} "
            f"wd={','.join(str(d) for d in wd)} ws={','.join(str(s) for s in ws)} "
            f"in={in_ty} out={out_ty}"
        )

    return _REDUCE_WINDOW_RE.sub(repl, body)


def _weight_requant(name: str, weight_map: dict | None) -> tuple[int | None, int | None]:
    """Return the `(M0, n)` requant pair for a weight operand, else `(None, None)`."""
    if weight_map is None:
        return None, None
    entry = weight_map.get(name)
    if entry is None or entry.get("kind") != "weight":
        return None, None
    return entry.get("M0"), entry.get("n")


def legalize_text(stablehlo_text: str, weight_map: dict | None = None) -> Program:
    """Parse optimized StableHLO text and build a TPU dialect `Program`.

    Only the `@main` function body is lowered; private helper functions (ReLU)
    are classified but not walked. When `weight_map` is supplied, each
    `dot_general`/`convolution -> tpu.mult` is annotated with its weight operand's
    dyadic requantization pair `M0`/`n`.
    """
    module_match = _MODULE_RE.search(stablehlo_text)
    name = module_match.group(1) if module_match else "module"
    program = Program(name=name)

    functions = _find_functions(stablehlo_text)
    relu_names = _relu_function_names(functions)
    main_body = next(
        (body for vis, fname, body in functions if vis == "public" and fname == "main"), None
    )
    assert main_body is not None, "Could not find the public @main function body."
    main_body = _collapse_reduce_windows(main_body)

    # Folded shape ops: map a result to (underlying value, shape a consumer sees).
    reshape_map: dict[str, tuple[str, list[int]]] = {}

    def resolve(value: str) -> str:
        """Follow folded reshapes/broadcasts/transposes back to the real data."""
        while value in reshape_map:
            value = reshape_map[value][0]
        return value

    # The window descriptor persists on-device; emit a new `window` only on change.
    current_window: tuple | None = None

    def emit_window(descriptor: tuple) -> None:
        nonlocal current_window
        if descriptor != current_window:
            program.ops.append(WindowOp(*descriptor))
            current_window = descriptor

    for raw_line in main_body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        m = _RESHAPE_RE.match(line)
        if m:
            result, src, _src_ty, dst_ty = m.groups()
            reshape_map[result] = (resolve(src), _tensor_shape(dst_ty))
            continue

        m = _BROADCAST_RE.match(line)
        if m:
            # broadcast_in_dim only feeds the bias-add path here; fold it to its
            # source so the add's bias operand traces to the *.bias arg (bias
            # removal then drops the whole add).
            result, src, _src_ty, dst_ty = m.groups()
            reshape_map[result] = (resolve(src), _tensor_shape(dst_ty))
            continue

        m = _TRANSPOSE_RE.match(line)
        if m:
            result, src, _src_ty, dst_ty = m.groups()
            src_name = resolve(src)
            entry = weight_map.get(src_name) if weight_map else None
            assert entry is not None and entry.get("kind") in ("weight", "bias"), (
                f"stablehlo.transpose of {src_name} is not a weight/constant "
                "(it is a runtime/live value); on-device transpose is out of scope. "
                "It should have been flagged by the transpose-analysis stage."
            )
            reshape_map[result] = (src_name, _tensor_shape(dst_ty))
            continue

        m = _CONV_RE.match(line)
        if m:
            _lower_convolution(m, program, weight_map, resolve, emit_window)
            continue

        m = _MAXPOOL_RE.match(line)
        if m:
            _lower_maxpool(m, program, resolve, emit_window)
            continue

        m = _DOT_RE.match(line)
        if m:
            result, lhs, rhs, lhs_ty, rhs_ty, out_ty = m.groups()
            lhs_name, rhs_name = resolve(lhs), resolve(rhs)
            # The requant multiplier belongs to whichever operand is a weight.
            M0, n = _weight_requant(lhs_name, weight_map)
            if M0 is None:
                M0, n = _weight_requant(rhs_name, weight_map)
            program.ops.append(
                MultOp(
                    result=result,
                    lhs=Operand(lhs_name, _tensor_shape(lhs_ty)),
                    rhs=Operand(rhs_name, _tensor_shape(rhs_ty)),
                    out_shape=_tensor_shape(out_ty),
                    M0=M0,
                    n=n,
                )
            )
            continue

        m = _CALL_RE.match(line)
        if m:
            result, fname, arg, in_ty, _out_ty = m.groups()
            assert fname in relu_names, (
                f"call @{fname} is not a recognized ReLU helper; unhandled op."
            )
            length = _prod(_tensor_shape(in_ty))
            program.ops.append(
                ReluOp(result=result, src=Operand(resolve(arg), [length]), length=length)
            )
            continue

        m = _ADD_RE.match(line)
        if m:
            result, lhs, rhs, ty = m.groups()
            shape = _tensor_shape(ty)
            program.ops.append(
                AddOp(
                    result=result,
                    lhs=Operand(resolve(lhs), shape),
                    rhs=Operand(resolve(rhs), shape),
                    out_shape=shape,
                )
            )
            continue

        if _CONSTANT_RE.match(line):
            # The only constant in @main is the reduce_window -inf init, which the
            # ISA `max` handles implicitly; it feeds no lowered instruction.
            continue

        m = _RETURN_RE.match(line)
        if m:
            value = resolve(m.group(1))
            program.ops.append(ReturnOp(value=value, shape=_last_shape(program, value)))
            continue

        assert not _UNHANDLED_RE.match(line), (
            f"Unhandled op in @main during legalization: {line!r}. "
            "v0.6 supports the LeNet-5 subset only; unsupported ops must not be "
            "silently dropped."
        )

    program.ops.append(EndOp())
    return program


def _lower_convolution(match, program, weight_map, resolve, emit_window) -> None:
    """Lower a `stablehlo.convolution` to `window` + `im2col` + `mult`."""
    (
        result,
        inp,
        kernel,
        dim_numbers,
        window_attr,
        extra_attrs,
        in_ty,
        kernel_ty,
        out_ty,
    ) = match.groups()

    normalized = re.sub(r"\s+", " ", dim_numbers).strip()
    assert normalized == _CONV_DIM_NUMBERS, (
        f"convolution dim_numbers {normalized!r} unsupported; only NCHW/OIHW "
        f"({_CONV_DIM_NUMBERS}) is handled (LeNet-5 subset)."
    )
    assert window_attr.strip() == "", (
        f"convolution window attribute {window_attr!r} unsupported; only unit "
        "stride with no padding/dilation is handled (LeNet-5 subset)."
    )
    for group in ("batch_group_count = 1", "feature_group_count = 1"):
        assert group in extra_attrs, (
            f"convolution requires {group} (no grouped/batched conv): {extra_attrs!r}."
        )

    in_shape = _tensor_shape(in_ty)  # [N, Cin, H, W]
    kernel_shape = _tensor_shape(kernel_ty)  # [Cout, Cin, kh, kw]
    out_shape = _tensor_shape(out_ty)  # [N, Cout, oh, ow]
    assert len(in_shape) == 4 and len(kernel_shape) == 4 and len(out_shape) == 4, (
        f"convolution expects 4-D NCHW/OIHW tensors: {in_shape}, {kernel_shape}, {out_shape}."
    )
    _, chans, inh, inw = in_shape
    out_ch, kernel_cin, kh, kw = kernel_shape
    _, out_ch2, outh, outw = out_shape
    assert kernel_cin == chans, f"kernel in-channels {kernel_cin} != feature-map channels {chans}."
    assert out_ch2 == out_ch, f"output channels {out_ch2} != kernel out-channels {out_ch}."
    assert outh == inh - kh + 1 and outw == inw - kw + 1, (
        f"conv output {outh}x{outw} inconsistent with unit-stride/no-pad geometry "
        f"from {inh}x{inw} kernel {kh}x{kw}."
    )

    emit_window((chans, inh, inw, outh, outw, kh, kw, 1, 1, 0, 0))

    K = chans * kh * kw
    N = outh * outw
    col_name = f"{result}_col"
    program.ops.append(
        Im2colOp(result=col_name, src=Operand(resolve(inp), [chans, inh, inw]), out_shape=[K, N])
    )

    weight_name = resolve(kernel)
    M0, n = _weight_requant(weight_name, weight_map)
    program.ops.append(
        MultOp(
            result=result,
            lhs=Operand(weight_name, [out_ch, K]),
            rhs=Operand(col_name, [K, N]),
            out_shape=[out_ch, N],
            M0=M0,
            n=n,
        )
    )


def _lower_maxpool(match, program, resolve, emit_window) -> None:
    """Lower a collapsed `tpu_maxpool` placeholder to `window` + `max`."""
    result, src, wd_body, ws_body, in_ty, out_ty = match.groups()
    wd = [int(v) for v in wd_body.split(",")]
    ws = [int(v) for v in ws_body.split(",")]
    in_shape = _tensor_shape(in_ty)  # [N, C, H, W]
    out_shape = _tensor_shape(out_ty)  # [N, C, oh, ow]
    assert len(in_shape) == 4 and len(out_shape) == 4, (
        f"reduce_window expects 4-D NCHW tensors: {in_shape}, {out_shape}."
    )
    assert wd[0] == 1 and wd[1] == 1 and ws[0] == 1 and ws[1] == 1, (
        f"pooling across batch/channel is unsupported: window {wd}, stride {ws}."
    )
    _, chans, inh, inw = in_shape
    _, chans_out, outh, outw = out_shape
    assert chans_out == chans, f"pool changed channel count {chans} -> {chans_out}."
    winh, winw = wd[2], wd[3]
    strh, strw = ws[2], ws[3]
    assert outh == (inh - winh) // strh + 1 and outw == (inw - winw) // strw + 1, (
        f"pool output {outh}x{outw} inconsistent with {inh}x{inw} "
        f"window {winh}x{winw} stride {strh}x{strw}."
    )

    emit_window((chans, inh, inw, outh, outw, winh, winw, strh, strw, 0, 0))
    program.ops.append(
        MaxPoolOp(
            result=result,
            src=Operand(resolve(src), [chans, inh, inw]),
            out_shape=[chans, outh, outw],
        )
    )


def _last_shape(program: Program, value: str) -> list[int]:
    for op in reversed(program.ops):
        if getattr(op, "result", None) == value:
            return getattr(op, "out_shape", [1, 1])
    return [1, 1]


def legalize(input_mlir: Path, output_tpu: Path, weight_map: dict | None = None) -> Program:
    """Read optimized StableHLO from `input_mlir`, write TPU dialect to `output_tpu`.

    `weight_map` (from `weight_map.json`) supplies each weight's `M0`/`n` so the
    emitted `tpu.mult` ops carry their requantization pair.
    """
    assert input_mlir.is_file(), f"Legalization input does not exist: {input_mlir}"

    program = legalize_text(input_mlir.read_text(), weight_map)
    output_tpu.write_text(serialize(program))
    return program
