"""Output-staging pass: dialect -> dialect (v0.7).

The final dialect pass in the lowering pipeline (after legalization, bias
removal, and matmul partitioning). It stages the network output into the
reserved I/O address so the programmer can read it back.

Rationale (retires LeNet-5 bring-up "Bug 2"): previously the assembler pinned the
returned value directly to the isolated I/O address 0x1. When that output was a
tiled matmul (e.g. LeNet's 10 logits, N-tiled into 8 + 2), the tiles past the
first were emitted at flat bases like 0x1 + 8 = 0x9, spilling out of the isolated
buffer and into resident weight memory -- corrupting weights across Programmer
re-runs. The fix writes the output to main memory (0x2+) as a full tensor like
any other intermediate, then copies it into 0x1 with a single `move`.

This pass inserts one `tpu.move %out -> @io` immediately before the terminator.
The destination is symbolic (`@io`, resolved to 0x1 by the assembler) and the
length is inferred from the returned value's shape, so no physical address enters
the IR. The assembler no longer special-cases the returned value, so it is
allocated in main memory by the normal allocator.
"""

from __future__ import annotations

from .dialect import MoveOp, Operand, Program, ReturnOp


def stage_output(program: Program) -> Program:
    """Insert a `tpu.move %out -> @io` before the `tpu.return` terminator.

    Locates the returned value and stages it into the I/O address for readback.
    A program with no `tpu.return` (e.g. a fragment with no declared output) is
    left unchanged. Idempotency guard: if the value is already staged (a MoveOp of
    the returned value immediately precedes the return), the pass is a no-op.
    """
    return_index = next(
        (i for i, op in enumerate(program.ops) if isinstance(op, ReturnOp)), None
    )
    if return_index is None:
        return program

    returned = program.ops[return_index]
    if return_index > 0:
        prior = program.ops[return_index - 1]
        if isinstance(prior, MoveOp) and prior.src.name == returned.value:
            return program

    move = MoveOp(Operand(returned.value, list(returned.shape)))
    program.ops.insert(return_index, move)
    return program
