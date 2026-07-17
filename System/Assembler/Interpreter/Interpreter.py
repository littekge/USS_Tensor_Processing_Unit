"""Instruction-level interpreter: executes the ACTUAL emitted TPU program.

Where Emulator.py models the intended integer math layer by layer, this executes
the real 128-bit instruction stream (PROGRAM.bin) against a modeled memory, using
the exact ISA addressing/requant semantics (Functional_TPU_ISA.md). It is the A/B
splitter for a hardware misclassification:

    golden (Emulator) == interpreter  ->  the emitted program is correct
                                           => any hardware divergence is RTL (B)
    golden (Emulator) != interpreter  ->  the emitted program is wrong
                                           => bug is in the assembler (A); the
                                              per-op trace shows the first divergence

Memory model (per ISA "Reserved Addresses"): address 0x0 is hardwired zero;
0x1 is an architecturally ISOLATED staging buffer reached ONLY when the base
address is exactly 0x1 (offsets index within it); every base >= 0x2 is flat main
memory. Weights load from MEM.bin at their addresses; the input image loads into
the 0x1 buffer exactly as the INPUT-header flow would; the final output is read
back from the 0x1 buffer.

Usage:
    python Interpreter/Interpreter.py [digit_index]     # default 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Same-directory Emulator (golden reference) reused for input prep + comparison.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import Emulator  # noqa: E402

_ASM_DIR = Path(__file__).resolve().parents[1]  # .../System/Assembler
_TMP = _ASM_DIR / "tmp"

IO_BUF_WORDS = 1024  # 0x1 staging buffer depth (Programmer.v: 10-bit offset)

# Opcodes (bits 127-124) and funct3 (bits 2-0), matching Assembler.py / the ISA.
OP_MULT, OP_RELU, OP_ADD = 0b1000, 0b1001, 0b1010
OP_MAX, OP_IM2COL, OP_WINDOW, OP_STRIDE, OP_SYSTEM = 0b1100, 0b1101, 0b1110, 0b1111, 0b0000
F3_MULTIP = 0x1

_OPNAME = {OP_MULT: "MUL", OP_RELU: "RELU", OP_ADD: "ADD", OP_MAX: "MAX",
           OP_IM2COL: "IM2COL", OP_WINDOW: "WINDOW", OP_STRIDE: "STRIDE", OP_SYSTEM: "SYS"}


def _bits(word: int, hi: int, lo: int) -> int:
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


def _signed8(byte: int) -> int:
    return byte - 256 if byte >= 128 else byte


def parse_program(path: Path) -> list[int]:
    """PROGRAM.bin -> list of 128-bit instruction words (strips the 0x50 tags)."""
    data = path.read_bytes()
    instrs, i = [], 0
    while i < len(data):
        assert data[i] == 0x50, f"expected PROGRAM tag 0x50 at {i}, got {data[i]:#x}"
        instrs.append(int.from_bytes(data[i + 1:i + 17], "little"))
        i += 17
    return instrs


class TPU:
    """A software TPU: flat main memory + isolated 0x1 buffer, executing the ISA."""

    def __init__(self) -> None:
        self.main: dict[int, int] = {}          # address (>=2) -> signed int8
        self.io = [0] * IO_BUF_WORDS            # the 0x1 staging buffer
        self.ld1 = self.ld2 = 0
        self.win: tuple | None = None           # (chans,inh,inw,outh,outw,winh,winw,strh,strw,padh,padw)
        self.acc: np.ndarray | None = None       # matrix accumulator across a multip run
        self.log: list[tuple] = []

    # --- memory: base address selects the region, offset indexes within it -----
    def read(self, base: int, off: int) -> int:
        if base == 0:
            return 0
        if base == 1:
            return self.io[off]
        return self.main.get(base + off, 0)

    def write(self, base: int, off: int, val: int) -> None:
        if base == 0:
            return  # store to 0x0 discards (ISA); assembler never does this
        if base == 1:
            self.io[off] = val
        else:
            self.main[base + off] = val

    def load_weights(self, mem_path: Path) -> None:
        """Load MEM.bin (weights) into main memory at their emitted addresses."""
        data = mem_path.read_bytes()
        i = 0
        while i < len(data):
            assert data[i] == 0x4D, f"expected MEM tag 0x4D at {i}, got {data[i]:#x}"
            addr = int.from_bytes(data[i + 1:i + 4], "little")
            length = int.from_bytes(data[i + 4:i + 6], "little")
            payload = data[i + 6:i + 6 + length]
            # Weights are always in main memory (addr >= 2 per the layout).
            for j, byte in enumerate(payload):
                self.main[addr + j] = _signed8(byte)
            i += 6 + length

    def load_input(self, image_int8: np.ndarray) -> None:
        """Load the quantized image into the 0x1 buffer (the INPUT-header flow)."""
        flat = image_int8.reshape(-1)
        for j, v in enumerate(flat):
            self.io[j] = int(v)

    # --- requant (Vector_Processor.v:251-264 / ISA mult equation) --------------
    @staticmethod
    def requant(acc: int, M0: int, n: int) -> int:
        bias = (1 << (n - 1)) if n > 0 else 0
        shifted = (M0 * acc + bias) >> n
        return max(-128, min(127, shifted))

    def _exec_mul(self, x: int) -> None:
        rs1, sz11, sz12 = _bits(x, 123, 100), _bits(x, 99, 96), _bits(x, 95, 92)
        rs2, sz21, sz22 = _bits(x, 91, 68), _bits(x, 67, 64), _bits(x, 63, 60)
        rd, M0, n, f3 = _bits(x, 59, 36), _bits(x, 35, 28), _bits(x, 27, 20), _bits(x, 2, 0)
        assert sz12 == sz21, f"contraction mismatch: {sz12} vs {sz21}"
        s1_stride = self.ld1 if self.ld1 else sz12   # rows of src1 spaced ld1
        s2_stride = self.ld2 if self.ld2 else sz22   # rows of src2 spaced ld2
        d_stride = self.ld2 if self.ld2 else sz22    # dest rows also spaced ld2

        if self.acc is None:
            self.acc = np.zeros((sz11, sz22), dtype=np.int64)
        for i in range(sz11):
            for j in range(sz22):
                partial = 0
                for k in range(sz12):
                    partial += self.read(rs1, i * s1_stride + k) * self.read(rs2, k * s2_stride + j)
                self.acc[i, j] += partial
        for i in range(sz11):
            for j in range(sz22):
                self.write(rd, i * d_stride + j, self.requant(int(self.acc[i, j]), M0, n))
        if f3 != F3_MULTIP:  # terminating mult clears the accumulator
            self.acc = None

    def _exec_windowed(self, x: int, is_pool: bool) -> None:
        rs1, rd = _bits(x, 123, 100), _bits(x, 99, 76)
        chans, inh, inw, outh, outw, winh, winw, strh, strw, padh, padw = self.win
        for c in range(chans):
            for oy in range(outh):
                for ox in range(outw):
                    if is_pool:
                        best = -128
                        for wr in range(winh):
                            for wc in range(winw):
                                ih, iw = oy * strh + wr - padh, ox * strw + wc - padw
                                if 0 <= ih < inh and 0 <= iw < inw:
                                    best = max(best, self.read(rs1, c * inh * inw + ih * inw + iw))
                        self.write(rd, c * outh * outw + oy * outw + ox, best)
                    else:
                        p = oy * outw + ox
                        for wr in range(winh):
                            for wc in range(winw):
                                ih, iw = oy * strh + wr - padh, ox * strw + wc - padw
                                val = (self.read(rs1, c * inh * inw + ih * inw + iw)
                                       if 0 <= ih < inh and 0 <= iw < inw else 0)
                                k = c * (winh * winw) + wr * winw + wc
                                self.write(rd, k * (outh * outw) + p, val)

    def run(self, instrs: list[int]) -> None:
        for x in instrs:
            op = _bits(x, 127, 124)
            if op == OP_MULT:
                self._exec_mul(x)
            elif op == OP_STRIDE:
                self.ld1, self.ld2 = _bits(x, 123, 100), _bits(x, 99, 76)
            elif op == OP_WINDOW:
                self.win = (_bits(x, 99, 88), _bits(x, 123, 112), _bits(x, 111, 100),
                            _bits(x, 87, 76), _bits(x, 75, 64), _bits(x, 63, 60), _bits(x, 59, 56),
                            _bits(x, 55, 52), _bits(x, 51, 48), _bits(x, 47, 44), _bits(x, 43, 40))
            elif op == OP_IM2COL:
                self._exec_windowed(x, is_pool=False)
            elif op == OP_MAX:
                self._exec_windowed(x, is_pool=True)
            elif op == OP_RELU:
                rs1, rd, length = _bits(x, 123, 100), _bits(x, 99, 76), _bits(x, 75, 60)
                for i in range(length):
                    self.write(rd, i, max(0, self.read(rs1, i)))
            elif op == OP_ADD:
                rs1, rs2 = _bits(x, 123, 100), _bits(x, 99, 76)
                sz1, sz2, rd = _bits(x, 75, 68), _bits(x, 67, 60), _bits(x, 59, 36)
                for off in range(sz1 * sz2):
                    self.write(rd, off, self.read(rs1, off) + self.read(rs2, off))
            elif op == OP_SYSTEM:
                break
            else:
                raise AssertionError(f"unknown opcode {op:#06b}")


def main() -> None:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    npz = np.load(Emulator._NPZ_PATH, allow_pickle=False)
    _, image_int8, label = Emulator.load_mnist(index, npz)

    # Golden reference (intended math) for comparison.
    qw, pairs = Emulator._quantized_weights(npz), Emulator._requant_pairs(npz)
    golden = [int(v) for v in Emulator.forward_int8(image_int8, qw, pairs)]

    tpu = TPU()
    tpu.load_weights(_TMP / "MEM.bin")
    tpu.load_input(image_int8)
    tpu.run(parse_program(_TMP / "PROGRAM.bin"))
    interp = [tpu.io[d] for d in range(10)]  # final output read from the 0x1 buffer

    print(f"MNIST test[{index}] -- true label: {label}\n")
    print(f"{'idx':>3} {'golden':>7} {'interp':>7}   match")
    for d in range(10):
        print(f"{d:>3} {golden[d]:>7} {interp[d]:>7}   {'ok' if golden[d] == interp[d] else 'DIFF'}")
    print(f"\nargmax  golden={int(np.argmax(golden))}  interp={int(np.argmax(interp))}  (true={label})")
    print(f"golden : {golden}")
    print(f"interp : {interp}")
    verdict = ("MATCH -> emitted program is correct; hardware divergence is an RTL bug (B)."
               if golden == interp else
               "DIFFER -> the emitted program is wrong; bug is in the assembler (A).")
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
