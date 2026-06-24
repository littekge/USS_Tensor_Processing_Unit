"""Legalization pass: StableHLO MLIR -> *Functional TPU* dialect.

This is the second MLIR pass in the lowering pipeline (it runs on the output of
the StableHLO optimization pass). For v0.1 it only needs to handle the operations
that survive optimization in `examples/v0.1_optimized_example.mlir`:

  * `stablehlo.reshape`     -- folded away (shape-only; no data movement)
  * `stablehlo.dot_general` -- lowered to `tpu.mult`
  * `stablehlo.add`         -- lowered to `tpu.add`

Reshapes are folded rather than emitted as their own op: a reshape does not move
data (tensors live in memory as flat row-major arrays per the ISA), so the only
thing it changes is the shape a *consumer* sees. We record that rebinding and
substitute it into consuming ops, keeping the dialect free of synthesized-away
operations (a stated design goal).
"""

from __future__ import annotations

import re
from pathlib import Path

from .dialect import AddOp, EndOp, MultOp, Operand, Program, ReturnOp, serialize


def _tensor_shape(type_body: str) -> list[int]:
    """Turn the inside of a `tensor<...>` type into a list of dimensions.

    `4x1xf32` -> [4, 1]; `4xf32` -> [4]; `f32` (scalar) -> [].
    """
    tokens = type_body.split("x")
    dims = tokens[:-1]  # last token is the element type (e.g. f32)
    return [int(d) for d in dims]


_FUNC_RE = re.compile(r"func\.func.*?@main\s*\((.*?)\)\s*->", re.DOTALL)
_ARG_RE = re.compile(r"(%arg\d+)\s*:\s*tensor<([^>]*)>")

_RESHAPE_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.reshape\s+(%[\w]+)\s*:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_DOT_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.dot_general\s+(%[\w]+)\s*,\s*(%[\w]+).*?:\s*"
    r"\(tensor<([^>]*)>\s*,\s*tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)
_ADD_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.add\s+(%[\w]+)\s*,\s*(%[\w]+)\s*:\s*tensor<([^>]*)>"
)
_RETURN_RE = re.compile(r"return\s+(%[\w]+)\s*:")
_MODULE_RE = re.compile(r"module\s+@([\w\.]+)")


def legalize_text(stablehlo_text: str) -> Program:
    """Parse optimized StableHLO text and build a TPU dialect `Program`."""
    module_match = _MODULE_RE.search(stablehlo_text)
    name = module_match.group(1) if module_match else "module"
    program = Program(name=name)

    # Map every reshape result to (source value, new shape) so consumers can be
    # rewritten to reference the underlying data directly.
    reshape_map: dict[str, tuple[str, list[int]]] = {}

    def resolve(value: str) -> str:
        """Follow folded reshapes back to the value that actually holds the data."""
        while value in reshape_map:
            value = reshape_map[value][0]
        return value

    func_match = _FUNC_RE.search(stablehlo_text)
    assert func_match is not None, "Could not find @main function signature in StableHLO input."

    # Process the body line by line so op ordering is preserved.
    for raw_line in stablehlo_text.splitlines():
        line = raw_line.strip()

        m = _RESHAPE_RE.match(line)
        if m:
            result, src, _src_ty, dst_ty = m.groups()
            reshape_map[result] = (resolve(src), _tensor_shape(dst_ty))
            continue

        m = _DOT_RE.match(line)
        if m:
            result, lhs, rhs, lhs_ty, rhs_ty, out_ty = m.groups()
            program.ops.append(
                MultOp(
                    result=result,
                    lhs=Operand(resolve(lhs), _tensor_shape(lhs_ty)),
                    rhs=Operand(resolve(rhs), _tensor_shape(rhs_ty)),
                    out_shape=_tensor_shape(out_ty),
                )
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

        m = _RETURN_RE.match(line)
        if m:
            value = resolve(m.group(1))
            # Find the shape from the matching producer if available; otherwise
            # fall back to the last op's output shape.
            program.ops.append(ReturnOp(value=value, shape=_last_shape(program, value)))
            continue

    program.ops.append(EndOp())
    return program


def _last_shape(program: Program, value: str) -> list[int]:
    for op in reversed(program.ops):
        if getattr(op, "result", None) == value:
            return op.out_shape
    return [1, 1]


def legalize(input_mlir: Path, output_tpu: Path) -> Program:
    """Read optimized StableHLO from `input_mlir`, write TPU dialect to `output_tpu`."""
    assert input_mlir.is_file(), f"Legalization input does not exist: {input_mlir}"

    program = legalize_text(input_mlir.read_text())
    output_tpu.write_text(serialize(program))
    return program
