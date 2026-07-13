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
_RETURN_RE = re.compile(r"return\s+(%[\w]+)\s*:")
_MODULE_RE = re.compile(r"module\s+@([\w\.]+)")


def _weight_requant(name: str, weight_map: dict | None) -> tuple[int | None, int | None]:
    """Return the `(M0, n)` requant pair for a weight operand, else `(None, None)`.

    `name` is the operand's resolved SSA name (reshapes already folded). Only
    weight entries carry `M0`/`n`; inputs and biases do not.
    """
    if weight_map is None:
        return None, None
    entry = weight_map.get(name)
    if entry is None or entry.get("kind") != "weight":
        return None, None
    return entry.get("M0"), entry.get("n")


def legalize_text(stablehlo_text: str, weight_map: dict | None = None) -> Program:
    """Parse optimized StableHLO text and build a TPU dialect `Program`.

    When `weight_map` is supplied, each `dot_general -> tpu.mult` is annotated with
    its weight operand's dyadic requantization pair `M0`/`n`.
    """
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

        m = _TRANSPOSE_RE.match(line)
        if m:
            result, src, _src_ty, dst_ty = m.groups()
            src_name = resolve(src)
            entry = weight_map.get(src_name) if weight_map else None
            assert entry is not None and entry.get("kind") in ("weight", "bias"), (
                f"stablehlo.transpose of {src_name} is not a weight/constant "
                "(it is a runtime/live value); on-device transpose is out of scope "
                "for v0.5. It should have been flagged by the transpose-analysis stage."
            )
            # The weight was physically transposed offline (Process_Weights), so
            # the transpose op is now a pure reinterpretation of that memory: fold
            # it like a reshape (rebind consumers to the already-transposed data).
            reshape_map[result] = (src_name, _tensor_shape(dst_ty))
            continue

        m = _DOT_RE.match(line)
        if m:
            result, lhs, rhs, lhs_ty, rhs_ty, out_ty = m.groups()
            lhs_name, rhs_name = resolve(lhs), resolve(rhs)
            # The requant multiplier belongs to whichever operand is a weight
            # (the other is the activation/intermediate); check both.
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


def legalize(input_mlir: Path, output_tpu: Path, weight_map: dict | None = None) -> Program:
    """Read optimized StableHLO from `input_mlir`, write TPU dialect to `output_tpu`.

    `weight_map` (from `weight_map.json`) supplies each weight's `M0`/`n` so the
    emitted `tpu.mult` ops carry their requantization pair.
    """
    assert input_mlir.is_file(), f"Legalization input does not exist: {input_mlir}"

    program = legalize_text(input_mlir.read_text(), weight_map)
    output_tpu.write_text(serialize(program))
    return program
