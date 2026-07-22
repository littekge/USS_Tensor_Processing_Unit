"""Serializer tests and a full end-to-end pipeline test (build step 8)."""

from pathlib import Path

import numpy as np
import pytest

from nn_assembler.Assembler import (
    MAX_MATMUL_SIZE,
    OPCODE_ADD,
    OPCODE_MULT,
    OPCODE_STRIDE,
    Assemble,
)
from nn_assembler.Convert import NN_import
from nn_assembler.MLIR.dialect import (
    Im2colOp,
    MaxPoolOp,
    MultipOp,
    MultOp,
    ReluOp,
    StrideOp,
    WindowOp,
    parse_program,
)
from nn_assembler.MLIR.transpose_analysis import analyze_transposes_file
from nn_assembler.Process_MLIR import Process_MLIR
from nn_assembler.Process_Weights import Process_Weights, decompose_multiplier
from nn_assembler.Protocol import FLASH, MEM, PROGRAM, STOP
from nn_assembler.Serializer import Serialize

# The real, exported Bigger_NN artifacts (three Linear layers, 1->1000->100->1).
# Its middle matmul (1x1000 @ 1000x100) tiles in both K and N, exercising the full
# partitioning path against a genuine torchax export rather than a synthetic graph.
_NETWORKS_DIR = Path(__file__).resolve().parents[2] / "Neural_Networks"
_BIGGER_NN_DIR = _NETWORKS_DIR / "Bigger_NN"
_have_bigger_nn = (_BIGGER_NN_DIR / "Bigger_NN_Recent.mlir").is_file() and (
    _BIGGER_NN_DIR / "Bigger_NN_Recent.weights.npz"
).is_file()

_LENET_DIR = _NETWORKS_DIR / "LeNet_5"
_have_lenet = (_LENET_DIR / "LeNet_5_Recent.mlir").is_file() and (
    _LENET_DIR / "LeNet_5_Recent.weights.npz"
).is_file()


def test_serialize_wraps_mem_then_program(tmp_path):
    (tmp_path / "MEM.bin").write_bytes(bytes([MEM, 0x02, 0x00, 0x01, 0x00, 0xAB]))
    (tmp_path / "PROGRAM.bin").write_bytes(bytes([PROGRAM]) + (0).to_bytes(16, "little"))

    out_dir = tmp_path / "out"
    result = Serialize(tmp_path, out_dir)
    data = result.read_bytes()

    assert data[0] == FLASH
    assert data[-1] == STOP
    # MEM payload comes before the PROGRAM payload.
    assert data.index(MEM) < data.index(PROGRAM)


def _have_stablehlo_opt():
    import shutil

    return shutil.which("stablehlo-opt") is not None


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
def test_full_pipeline(tmp_path):
    # Recreate the Tiny_NN inputs from the optimized example + a small weights set.
    example = (
        "module @jit_func {\n"
        "  func.func public @main("
        '%arg0: tensor<4x1xf32> loc("states[0]"), '
        '%arg1: tensor<4xf32> loc("states[1]"), '
        '%arg2: tensor<1x4xf32> loc("states[2]"), '
        '%arg3: tensor<1xf32> loc("states[3]"), '
        '%arg4: tensor<1x1xf32> loc("inputs[0][0]")) '
        "-> (tensor<1x1xf32>) {\n"
        "    %0 = stablehlo.reshape %arg0 : (tensor<4x1xf32>) -> tensor<1x4xf32>\n"
        "    %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [1] x [0] : "
        "(tensor<1x1xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>\n"
        "    %2 = stablehlo.reshape %arg1 : (tensor<4xf32>) -> tensor<1x4xf32>\n"
        "    %3 = stablehlo.add %2, %1 : tensor<1x4xf32>\n"
        "    %4 = stablehlo.reshape %arg2 : (tensor<1x4xf32>) -> tensor<4x1xf32>\n"
        "    %5 = stablehlo.dot_general %3, %4, contracting_dims = [1] x [0] : "
        "(tensor<1x4xf32>, tensor<4x1xf32>) -> tensor<1x1xf32>\n"
        "    %6 = stablehlo.reshape %arg3 : (tensor<1xf32>) -> tensor<1x1xf32>\n"
        "    %7 = stablehlo.add %6, %5 : tensor<1x1xf32>\n"
        "    return %7 : tensor<1x1xf32>\n"
        "  }\n"
        "}\n"
    )
    (tmp_path / "initial.mlir").write_text(example)
    S_w = 1.0 / 127
    np.savez(
        tmp_path / "weights.npz",
        **{
            "hidden.weight": np.zeros((4, 1), dtype=np.float32),
            "hidden.bias": np.zeros((4,), dtype=np.float32),
            "output.weight": np.zeros((1, 4), dtype=np.float32),
            "output.bias": np.zeros((1,), dtype=np.float32),
            "__order__": np.array(["hidden.weight", "hidden.bias", "output.weight", "output.bias"]),
            # v0.3 requant metadata: weights carry M (biases do not).
            "__M__hidden.weight": np.float64(0.75),
            "__scales__hidden.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
            "__M__output.weight": np.float64(0.5),
            "__scales__output.weight": np.array([1.0, S_w, 1.0], dtype=np.float64),
        },
    )

    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    # Bias adds are dropped. Both matmuls are in-array (pass-through), so each is
    # preceded by a stride(0,0) contiguous reset; v0.7 stages the output with a
    # final move: 2 strides + 2 mults + move + end.
    assert len(instructions) == 6
    mults = [i for i in instructions if _bits(i, 127, 124) == OPCODE_MULT]
    strides = [i for i in instructions if _bits(i, 127, 124) == OPCODE_STRIDE]
    assert len(mults) == 2 and len(strides) == 2
    # Every pass-through stride reset is contiguous (ld1 = ld2 = 0).
    assert all(_bits(s, 123, 100) == 0 and _bits(s, 99, 76) == 0 for s in strides)

    # No ADD opcodes survive, and each mult carries its layer's M0/n (v0.4: M0 at
    # bits 35-28, n at bits 27-20).
    assert all(_bits(instr, 127, 124) != OPCODE_ADD for instr in instructions)
    assert (_bits(mults[0], 35, 28), _bits(mults[0], 27, 20)) == (192, 8)  # M=0.75
    assert (_bits(mults[1], 35, 28), _bits(mults[1], 27, 20)) == (128, 8)  # M=0.5

    # The final output (second mult's result) now lands in main memory, and a
    # single move to 0x1 (SHAPE opcode, funct3 MOVE) is the last instruction before
    # end, staging it for readback.
    output_addr = _bits(mults[1], 59, 36)
    assert output_addr >= 2 and output_addr != 1
    move = instructions[-2]
    assert _bits(move, 127, 124) == 0b1101 and _bits(move, 2, 0) == 0x1
    assert _bits(move, 123, 100) == output_addr  # src = output base
    assert _bits(move, 99, 76) == 1  # dest = 0x1
    assert instructions[-1] == 0  # end

    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP


def _k_run_covers_contraction(tiles, K, S):
    """A per-output-tile K-run must be (multip x (k-1), mult) covering all of K."""
    k_offsets = list(range(0, K, S))
    for start in range(0, len(tiles), len(k_offsets)):
        run = tiles[start : start + len(k_offsets)]
        assert [t.lhs.offset for t in run] == k_offsets  # in-order K walk
        assert all(isinstance(t, MultipOp) for t in run[:-1])
        assert isinstance(run[-1], MultOp)  # terminating mult


def _tiles_between_strides(program, stride):
    """All mult/multip tiles emitted for a matmul: from its stride to the next."""
    idx = program.ops.index(stride)
    tiles = []
    for op in program.ops[idx + 1 :]:
        if isinstance(op, StrideOp):
            break
        if isinstance(op, (MultOp, MultipOp)):
            tiles.append(op)
    return tiles


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
@pytest.mark.skipif(not _have_bigger_nn, reason="Bigger_NN_Recent artifacts not present")
def test_full_pipeline_bigger_nn_real(tmp_path):
    """Full pipeline on the REAL exported Bigger_NN (1->1000->100->1) artifact.

    Uses the checked-in, torchax-exported `Bigger_NN_Recent.{mlir,weights.npz}`
    directly -- no synthesized requant metadata. Exercises all three matmuls,
    with the middle one (1x1000 @ 1000x100) tiling in both K and N.
    """
    NN_import("Bigger_NN", "Recent", tmp_dir=tmp_path, nn_dir=_NETWORKS_DIR)

    # The regenerated npz carries the v0.3 requant metadata (__M__/__scales__ per
    # weight); the pre-v0.3 file did not, forcing the earlier synthesized workaround.
    weights = np.load(tmp_path / "weights.npz", allow_pickle=True)
    assert any(k.startswith("__M__") for k in weights.files)
    assert any(k.startswith("__scales__") for k in weights.files)

    manifest = analyze_transposes_file(tmp_path)
    assert len(manifest["weight_transposes"]) == 3  # all three weights transposed
    assert manifest["runtime_transposes"] == []

    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    # Framed transmission.
    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP

    program = parse_program((tmp_path / "optimized.partitioned.tpu.mlir").read_text())
    strides = [op for op in program.ops if isinstance(op, StrideOp)]
    # One stride per matmul, each (ld1 = K, ld2 = N) for the three layers.
    assert [(s.ld1, s.ld2) for s in strides] == [(1, 1000), (1000, 100), (100, 1)]

    S = MAX_MATMUL_SIZE

    # --- Middle matmul (1x1000 @ 1000x100): tiles in BOTH K and N. ---
    both_tiled = next(s for s in strides if (s.ld1, s.ld2) == (1000, 100))
    tiles = _tiles_between_strides(program, both_tiled)
    K, N = 1000, 100
    n_tiles = len(range(0, N, S))  # 13 (100 -> 12 full + ragged 4)
    k_tiles = len(range(0, K, S))  # 125 (1000 = 125*8 exactly)
    assert len(tiles) == n_tiles * k_tiles  # 1625 tiles

    # Each output tile's K-run accumulates via multip and terminates with a mult.
    _k_run_covers_contraction(tiles, K, S)
    # N-tiles cover the full 100-wide output via their destination offsets.
    assert sorted({t.dst_offset for t in tiles}) == list(range(0, N, S))
    # Sub-block corners: lhs offset = ki (M=1); rhs offset = ki*N + ni.
    assert {t.lhs.offset for t in tiles} == set(range(0, K, S))
    assert {t.rhs.offset for t in tiles} == {ki * N + ni for ki in range(0, K, S) for ni in range(0, N, S)}
    # Ragged edge: N remainder 4 (100 - 12*8) appears; K divides evenly so no K remainder.
    assert {t.rhs.shape[1] for t in tiles} == {S, N - 12 * S}
    assert {t.lhs.shape[1] for t in tiles} == {S}
    # Every tile carries the layer's requant pair (hidden2.weight: M0=172, n=19).
    assert all((t.M0, t.n) == (172, 19) for t in tiles)

    # --- Last matmul (1x100 @ 100x1): ragged K remainder 4 (100 - 12*8). ---
    last_stride = next(s for s in strides if (s.ld1, s.ld2) == (100, 1))
    last_tiles = _tiles_between_strides(program, last_stride)
    _k_run_covers_contraction(last_tiles, 100, S)
    assert {t.lhs.shape[1] for t in last_tiles} == {S, 100 - 12 * S}

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    last_mult = next(
        instructions[i]
        for i in range(len(instructions) - 1, -1, -1)
        if _bits(instructions[i], 127, 124) == 0b1000
    )
    # v0.7: the returned result lands in main memory; a final move stages it to 0x1.
    output_addr = _bits(last_mult, 59, 36)
    assert output_addr >= 2 and output_addr != 1
    move = instructions[-2]
    assert _bits(move, 127, 124) == 0b1101 and _bits(move, 2, 0) == 0x1  # SHAPE/MOVE
    assert _bits(move, 123, 100) == output_addr  # src = output base
    assert _bits(move, 99, 76) == 1  # dest = I/O address 0x1
    assert instructions[-1] == 0  # end


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
@pytest.mark.skipif(not _have_lenet, reason="LeNet_5_Recent artifacts not present")
def test_full_pipeline_lenet5_real(tmp_path):
    """Full pipeline on the REAL re-exported LeNet-5 artifact (v0.6 E2E).

    Two conv layers (im2col + tiled matmul), two max-pools, three FC layers, and
    ReLU throughout. Verifies windowed geometry, channel-major im2col, conv-matmul
    tiling, per-channel pooling, ReLU emission, CHW chaining, and M0/n on tiles.
    """
    NN_import("LeNet_5", "Recent", tmp_dir=tmp_path, nn_dir=_NETWORKS_DIR)

    weights = np.load(tmp_path / "weights.npz", allow_pickle=True)
    # Per-layer requant metadata is present for every conv/fc weight.
    for layer in ("conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight", "fc3.weight"):
        assert f"__M__{layer}" in weights.files
        assert f"__scales__{layer}" in weights.files

    analyze_transposes_file(tmp_path)
    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    instructions = Assemble(tmp_path)
    out_path = Serialize(tmp_path, tmp_path / "out")

    # Framed transmission.
    data = out_path.read_bytes()
    assert data[0] == FLASH and data[-1] == STOP

    program = parse_program((tmp_path / "optimized.partitioned.tpu.mlir").read_text())

    # --- One window per conv/pool with correct geometry (deduped, in order). ---
    windows = [op for op in program.ops if isinstance(op, WindowOp)]
    geometry = [
        (w.chans, w.inh, w.inw, w.outh, w.outw, w.winh, w.winw, w.strh, w.strw, w.padh, w.padw)
        for w in windows
    ]
    assert geometry == [
        (1, 28, 28, 26, 26, 3, 3, 1, 1, 0, 0),  # conv1
        (6, 26, 26, 13, 13, 2, 2, 2, 2, 0, 0),  # pool1
        (6, 13, 13, 11, 11, 3, 3, 1, 1, 0, 0),  # conv2
        (16, 11, 11, 5, 5, 2, 2, 2, 2, 0, 0),  # pool2
    ]

    # --- im2col columns in channel-major K order (K = chans*kh*kw), N = outh*outw. ---
    im2cols = [op for op in program.ops if isinstance(op, Im2colOp)]
    assert [op.out_shape for op in im2cols] == [[9, 676], [54, 121]]

    # --- Per-channel max-pool outputs (CHW). ---
    pools = [op for op in program.ops if isinstance(op, MaxPoolOp)]
    assert [op.out_shape for op in pools] == [[6, 13, 13], [16, 5, 5]]

    # --- ReLU emitted (one per conv + one per FC before the last). ---
    relus = [op for op in program.ops if isinstance(op, ReluOp)]
    assert len(relus) == 4

    # --- CHW intermediates chain with no reshape: each windowed op reads the
    #     previous layer's CHW output. ---
    assert im2cols[0].src.name == "%arg10"  # network input feeds conv1 im2col
    assert pools[0].src.name == relus[0].result  # relu(conv1) -> pool1
    assert im2cols[1].src.name == pools[0].result  # pool1 -> conv2 im2col
    assert pools[1].src.name == relus[1].result  # relu(conv2) -> pool2

    # --- One stride per matmul: (ld1 = K, ld2 = N) for conv1/conv2/fc1/fc2/fc3. ---
    strides = [op for op in program.ops if isinstance(op, StrideOp)]
    assert [(s.ld1, s.ld2) for s in strides] == [
        (9, 676),  # conv1: K=1*3*3, N=26*26
        (54, 121),  # conv2: K=6*3*3, N=11*11
        (400, 120),  # fc1
        (120, 84),  # fc2
        (84, 10),  # fc3
    ]

    S = MAX_MATMUL_SIZE

    # --- conv1 matmul (6x9 @ 9x676): tiled in K (9 -> 8+1) and N (676 -> 84*8+4).
    #     Weight (out_ch x K) is the LHS, im2col columns the RHS. ---
    conv1_stride = next(s for s in strides if (s.ld1, s.ld2) == (9, 676))
    conv1_tiles = _tiles_between_strides(program, conv1_stride)
    K1, N1 = 9, 676
    # M = 6 <= S so a single M-tile; K-run per output tile is (multip, mult).
    _k_run_covers_contraction(conv1_tiles, K1, S)
    # LHS sub-block corner: offset = mi*K + ki = ki (mi=0). RHS: ki*N + ni. dst: ni.
    assert {t.lhs.offset for t in conv1_tiles} == set(range(0, K1, S))  # {0, 8}
    assert sorted({t.dst_offset for t in conv1_tiles}) == list(range(0, N1, S))
    assert {t.rhs.offset for t in conv1_tiles} == {
        ki * N1 + ni for ki in range(0, K1, S) for ni in range(0, N1, S)
    }
    # Ragged edges: K remainder 1, N remainder 4.
    assert {t.lhs.shape[1] for t in conv1_tiles} == {S, K1 - S}
    assert {t.rhs.shape[1] for t in conv1_tiles} == {S, N1 - 84 * S}

    # --- M0/n on every conv tile matches the exported multiplier decomposition. ---
    conv1_M0, conv1_n = decompose_multiplier(float(weights["__M__conv1.weight"]))
    assert all((t.M0, t.n) == (conv1_M0, conv1_n) for t in conv1_tiles)

    conv2_stride = next(s for s in strides if (s.ld1, s.ld2) == (54, 121))
    conv2_tiles = _tiles_between_strides(program, conv2_stride)
    conv2_M0, conv2_n = decompose_multiplier(float(weights["__M__conv2.weight"]))
    assert all((t.M0, t.n) == (conv2_M0, conv2_n) for t in conv2_tiles)

    def _bits(value, hi, lo):
        return (value >> lo) & ((1 << (hi - lo + 1)) - 1)

    # --- Bug 2 regression: the FC3 output [1x10] tiles in N (10 -> 8 + 2). The
    #     output now writes to CONTIGUOUS main memory (base + {0, 8}), never to the
    #     old 0x1 + 8 = 0x9 (which spilled into weight memory), and a single move to
    #     0x1 stages it for readback as the last instruction before end. ---
    fc3_stride = next(s for s in strides if (s.ld1, s.ld2) == (84, 10))
    fc3_tiles = _tiles_between_strides(program, fc3_stride)
    assert sorted({t.dst_offset for t in fc3_tiles}) == [0, 8]

    move = instructions[-2]
    assert _bits(move, 127, 124) == 0b1101 and _bits(move, 2, 0) == 0x1  # SHAPE/MOVE
    assert _bits(move, 99, 76) == 1  # dest = I/O address 0x1
    assert _bits(move, 75, 60) == 10  # len = 10 logits
    assert instructions[-1] == 0  # end
    output_base = _bits(move, 123, 100)  # source = output's main-memory base
    assert output_base >= 2 and output_base != 1

    mult_rds = {_bits(i, 59, 36) for i in instructions if _bits(i, 127, 124) == 0b1000}
    assert {output_base, output_base + 8} <= mult_rds  # tiles contiguous in main mem
    assert 9 not in mult_rds  # never the old 0x9 weight collision

    # The staged dialect artifact is inspectable and holds exactly one move -> @io.
    staged = (tmp_path / "optimized.staged.tpu.mlir").read_text()
    assert staged.count("tpu.move") == 1 and "-> @io" in staged


def _load_interpreter():
    """Import the out-of-package Interpreter module (Interpreter/Interpreter.py)."""
    import importlib.util

    interp_path = Path(__file__).resolve().parents[1] / "Interpreter" / "Interpreter.py"
    spec = importlib.util.spec_from_file_location("_tpu_interpreter", interp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interpreter_executes_move_and_leaves_weights_intact():
    """Interpreter Step 5: `move` copies src -> 0x1, honoring 0x1 isolation.

    Dataset-free unit check of the move opcode: an output tensor sitting in main
    memory alongside resident weights is staged into the isolated 0x1 buffer; the
    weight words are never touched (Bug 2 in miniature).
    """
    from nn_assembler.Assembler import encode_end, encode_move

    interp = _load_interpreter()

    tpu = interp.TPU()
    # Resident weights at low addresses; the network output further out in main mem.
    weights = {2: 11, 3: 22, 4: 33, 9: 44}  # 0x9 is the old Bug-2 collision address
    tpu.main.update(weights)
    output_base = 100
    output = [7, -8, 9]
    for i, v in enumerate(output):
        tpu.main[output_base + i] = v

    program = [encode_move(rs1=output_base, rd=1, length=len(output)), encode_end()]
    tpu.run(program)

    # The output is staged into the isolated 0x1 buffer...
    assert [tpu.io[i] for i in range(len(output))] == output
    # ...and every resident weight word is byte-identical (never overwritten).
    for addr, value in weights.items():
        assert tpu.main[addr] == value


@pytest.mark.skipif(not _have_stablehlo_opt(), reason="stablehlo-opt not on PATH")
@pytest.mark.skipif(not _have_lenet, reason="LeNet_5_Recent artifacts not present")
def test_lenet5_interpreter_double_run_preserves_weights(tmp_path):
    """Bug 2 regression: re-running the program never corrupts weight memory.

    Executes the emitted LeNet-5 program twice back-to-back (Programmer re-run
    semantics) in the instruction-level interpreter. Because v0.7 writes the tiled
    output to main memory and only stages it into the isolated 0x1 buffer with a
    move, no store ever lands in the weight region -- so weight memory is
    byte-identical after each run, and the argmax matches the golden model.
    """
    interp = _load_interpreter()
    emulator = interp.Emulator

    try:
        npz = np.load(emulator._NPZ_PATH, allow_pickle=False)
        _, image_int8, _label = emulator.load_mnist(0, npz)
    except Exception as exc:  # torchvision/dataset unavailable in this environment
        pytest.skip(f"MNIST input unavailable: {exc}")

    NN_import("LeNet_5", "Recent", tmp_dir=tmp_path, nn_dir=_NETWORKS_DIR)
    analyze_transposes_file(tmp_path)
    Process_Weights(tmp_path)
    Process_MLIR(tmp_path)
    Assemble(tmp_path)

    # Golden reference logits (intended int8 math).
    qw = emulator._quantized_weights(npz)
    pairs = emulator._requant_pairs(npz)
    golden = [int(v) for v in emulator.forward_int8(image_int8, qw, pairs)]
    golden_argmax = int(np.argmax(golden))

    program = interp.parse_program(tmp_path / "PROGRAM.bin")

    tpu = interp.TPU()
    tpu.load_weights(tmp_path / "MEM.bin")
    pristine_weights = dict(tpu.main)  # weight region: address -> signed int8
    assert pristine_weights, "expected resident weights loaded from MEM.bin"

    for _run in range(2):  # execute twice back-to-back
        tpu.load_input(image_int8)
        tpu.run(program)
        # Every pristine weight address is byte-identical after the run.
        for addr, value in pristine_weights.items():
            assert tpu.main[addr] == value, f"weight at {addr:#x} corrupted after run"
        out = [tpu.io[d] for d in range(10)]  # output read from the staged 0x1 buffer
        assert out == golden, f"interpreter logits {out} != golden {golden}"
        assert int(np.argmax(out)) == golden_argmax
