"""Matmul-partitioning pass: dialect -> dialect (v0.5).

The final dialect pass in the lowering pipeline (after legalization and bias
removal). It decomposes any `tpu.mult` whose logical dimensions exceed the
systolic array (`MAX_MATMUL_SIZE`) into array-sized tiles addressed *in place*
via leading dimensions -- no blocked memory layout is needed.

For an `M x K` by `K x N` matmul that exceeds the array, the pass emits:

  1. one `tpu.stride ld1 = K, ld2 = N` -- the parent row strides, shared by every
     tile until the next matmul's stride;
  2. for each output tile `(mi, ni)` (row-parallel `M`-tile x `N`-tile), a walk
     over the contraction `K` in array-sized steps, emitting `tpu.multip` for
     every K-tile but the last and `tpu.mult` for the last, so the run
     accumulates the full contraction in-array and only the final `mult` writes
     back a meaningful (requantized) result.

Every emitted tile addresses the corner of its sub-block via a row-major element
offset into the parent matrix (`offset = row_off * ld + col_off`); the assembler
adds that to the parent's base address. Ragged edge tiles carry their real
(smaller) dimensions. Each tile carries the parent matmul's `M0`/`n`.

A matmul already within the array passes through as a single `mult`, but it is
preceded by a `tpu.stride ld1 = 0, ld2 = 0` (contiguous) reset. The `ld1`/`ld2`
registers persist across instructions until the next `stride`, so a pass-through
matmul following a partitioned one would otherwise inherit stale leading
dimensions; the reset guarantees an in-array `mult` never runs with strides left
over from an earlier partitioned matmul (approved rule: "an in-array matmul
following a partitioned matmul should always reset stride to (0,0)").
"""

from __future__ import annotations

from .dialect import MultipOp, MultOp, Operand, Program, StrideOp


def _mkn(op: MultOp) -> tuple[int, int, int]:
    """Return `(M, K, N)` for a `MultOp` whose operands are 2-D (M x K, K x N)."""
    assert len(op.lhs.shape) == 2 and len(op.rhs.shape) == 2, (
        f"Partitioning expects 2-D matmul operands, got lhs {op.lhs.shape}, "
        f"rhs {op.rhs.shape} for {op.result}."
    )
    M, K = op.lhs.shape
    K2, N = op.rhs.shape
    assert K == K2, f"Contraction mismatch for {op.result}: lhs K={K}, rhs K={K2}."
    return M, K, N


def _fits(op: MultOp, max_size: int) -> bool:
    """True if every dimension of the matmul fits in one array tile."""
    if len(op.lhs.shape) != 2 or len(op.rhs.shape) != 2:
        return True  # non-2-D (e.g. degenerate 1-D) ops pass through untouched
    M, K, N = _mkn(op)
    return M <= max_size and K <= max_size and N <= max_size


def _tile_matmul(op: MultOp, max_size: int) -> list:
    """Emit the stride + tiled `multip`/`mult` sequence for one oversized matmul."""
    M, K, N = _mkn(op)
    ops: list = [StrideOp(ld1=K, ld2=N)]

    for mi in range(0, M, max_size):
        rows = min(max_size, M - mi)
        for ni in range(0, N, max_size):
            cols = min(max_size, N - ni)
            k_starts = list(range(0, K, max_size))
            for position, ki in enumerate(k_starts):
                depth = min(max_size, K - ki)
                is_last_k = position == len(k_starts) - 1
                cls = MultOp if is_last_k else MultipOp
                ops.append(
                    cls(
                        result=op.result,
                        lhs=Operand(op.lhs.name, [rows, depth], offset=mi * K + ki),
                        rhs=Operand(op.rhs.name, [depth, cols], offset=ki * N + ni),
                        out_shape=[rows, cols],
                        M0=op.M0,
                        n=op.n,
                        dst_offset=mi * N + ni,
                        parent_out_shape=[M, N],
                    )
                )
    return ops


def partition_program(program: Program, max_size: int) -> Program:
    """Return a new `Program` with oversized `MultOp`s tiled to the array size."""
    new_ops: list = []
    for op in program.ops:
        if isinstance(op, MultOp) and not _fits(op, max_size):
            new_ops.extend(_tile_matmul(op, max_size))
        elif isinstance(op, MultOp):
            # Pass-through (in-array) matmul: reset the leading dimensions to
            # contiguous first so it never inherits a stale stride from a
            # preceding partitioned matmul (ld1/ld2 persist until the next stride).
            new_ops.append(StrideOp(ld1=0, ld2=0))
            new_ops.append(op)
        else:
            new_ops.append(op)
    program.ops = new_ops
    return program
