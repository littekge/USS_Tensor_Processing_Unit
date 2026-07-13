"""Tests for the matmul-partitioning pass (v0.5, leading-dimension tiling)."""

from nn_assembler.Assembler import MAX_MATMUL_SIZE
from nn_assembler.MLIR.dialect import EndOp, MultipOp, MultOp, Operand, Program, ReturnOp
from nn_assembler.MLIR.partition import partition_program

S = MAX_MATMUL_SIZE  # 8


def _program(M, K, N, M0=128, n=8):
    program = Program(name="m")
    program.ops.append(
        MultOp("%r", Operand("%a", [M, K]), Operand("%b", [K, N]), [M, N], M0=M0, n=n)
    )
    program.ops.append(ReturnOp("%r", [M, N]))
    program.ops.append(EndOp())
    return program


def _tiles(program):
    return [op for op in program.ops if isinstance(op, (MultOp, MultipOp))]


def test_small_matmul_passes_through_unchanged():
    program = _program(1, S, S)  # every dim exactly the array size -> fits
    partition_program(program, S)
    tiles = _tiles(program)
    assert len(tiles) == 1
    assert isinstance(tiles[0], MultOp)
    assert not any(type(op).__name__ == "StrideOp" for op in program.ops)


def test_boundary_one_over_array_tiles():
    # N = S+1 must tile (2 N-tiles); N = S must not.
    fits = _program(1, S, S)
    partition_program(fits, S)
    assert len(_tiles(fits)) == 1

    over = _program(1, S, S + 1)
    partition_program(over, S)
    strides = [op for op in over.ops if type(op).__name__ == "StrideOp"]
    assert len(strides) == 1
    assert len(_tiles(over)) == 2  # 1 K-tile x 2 N-tiles


def test_one_stride_per_matmul_with_correct_leading_dims():
    program = _program(1, 20, 30)
    partition_program(program, S)
    strides = [op for op in program.ops if type(op).__name__ == "StrideOp"]
    assert len(strides) == 1
    assert (strides[0].ld1, strides[0].ld2) == (20, 30)  # ld1 = K, ld2 = N


def test_k_accumulation_multip_run_then_mult():
    # K = 3*S -> 3 K-tiles per output tile: 2 multip then 1 mult, in order.
    program = _program(1, 3 * S, 1)
    partition_program(program, S)
    tiles = _tiles(program)  # single output tile (M=1, N=1)
    assert len(tiles) == 3
    assert isinstance(tiles[0], MultipOp)
    assert isinstance(tiles[1], MultipOp)
    assert isinstance(tiles[2], MultOp)  # final K-tile terminates the run
    # Contraction offsets step by S along K (lhs cols, rhs rows).
    assert [t.lhs.offset for t in tiles] == [0, S, 2 * S]
    assert [t.rhs.offset for t in tiles] == [0, S, 2 * S]


def test_row_parallel_m_and_n_tiling_offsets():
    # M=2*S, N=2*S, K=S (single K-tile, so every tile is a terminating mult).
    program = _program(2 * S, S, 2 * S)
    partition_program(program, S)
    tiles = _tiles(program)
    assert len(tiles) == 4  # 2 M-tiles x 2 N-tiles x 1 K-tile
    assert all(isinstance(t, MultOp) for t in tiles)

    N = 2 * S
    K = S
    # dst offset = mi*N + ni ; lhs offset = mi*K + ki ; rhs offset = ki*N + ni.
    expected_dst = {0, S, S * N, S * N + S}  # (mi,ni) in {0,S}^2
    assert {t.dst_offset for t in tiles} == expected_dst
    assert {t.lhs.offset for t in tiles} == {0, S * K}  # mi in {0,S}, ki=0
    assert {t.rhs.offset for t in tiles} == {0, S}  # ki=0, ni in {0,S}
    assert all(t.parent_out_shape == [2 * S, 2 * S] for t in tiles)


def test_ragged_edge_tiles_use_real_sizes():
    program = _program(1, S + 2, S + 3)  # K=10, N=11 with S=8
    partition_program(program, S)
    tiles = _tiles(program)
    # Tile dims never exceed the array; ragged remainders (2 and 3) appear.
    for t in tiles:
        assert t.lhs.shape[1] <= S and t.rhs.shape[0] <= S and t.rhs.shape[1] <= S
    assert {t.lhs.shape[1] for t in tiles} == {S, 2}  # K remainder
    assert {t.rhs.shape[1] for t in tiles} == {S, 3}  # N remainder


def test_every_tile_carries_parent_requant():
    program = _program(1, 20, 20, M0=200, n=9)
    partition_program(program, S)
    for t in _tiles(program):
        assert (t.M0, t.n) == (200, 9)
