"""Unit tests for the LeNet-5 demo's pure preprocessing/framing functions.

The GUI, serial link, and run.sh programming are hardware/manual; only the
deterministic pure functions are covered here. Framing is checked byte-exact
against nn_assembler's MEM encoder so the demo can never drift from the assembler.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import LeNet5_Demo as demo  # noqa: E402
from nn_assembler.Protocol import build_mem_block, STOP  # noqa: E402


def test_build_input_transmission_is_byte_exact():
    image = bytes(range(0, 200)) * 4  # 800 arbitrary payload bytes
    tx = demo.build_input_transmission(image)
    assert tx == bytes([0x49]) + build_mem_block(demo.INPUT_ADDRESS, image) + bytes([STOP])
    assert tx[0] == 0x49              # INPUT header
    assert tx[-1] == STOP             # STOP trailer
    assert demo.INPUT_ADDRESS == 0x1  # staged to the reserved I/O address


def test_quantize_length_and_dtype():
    gray = np.zeros((demo.GRID, demo.GRID), dtype=np.uint8)
    out = demo.quantize(gray, s_in=0.02)
    assert len(out) == demo.GRID * demo.GRID  # 784 int8 bytes
    vals = np.frombuffer(out, dtype=np.int8)
    assert vals.min() >= -127 and vals.max() <= 127


def test_quantize_known_values_and_clamp():
    s_in = 0.02
    # All-black: normalized = (0 - mean)/std = -0.4331; /s_in = -21.65 -> round -22.
    black = demo.quantize(np.zeros((demo.GRID, demo.GRID), np.uint8), s_in)
    assert set(np.frombuffer(black, dtype=np.int8).tolist()) == {-22}
    # All-white: (1 - mean)/std = 2.880; /s_in = 144 -> clamps to +127.
    white = demo.quantize(np.full((demo.GRID, demo.GRID), 255, np.uint8), s_in)
    assert set(np.frombuffer(white, dtype=np.int8).tolist()) == {127}


def test_sketch_to_grid_empty_is_zeros():
    blank = demo.Image.new("L", (demo.CANVAS, demo.CANVAS), 0)
    grid = demo.sketch_to_grid(blank)
    assert grid.shape == (demo.GRID, demo.GRID)
    assert not grid.any()


def test_sketch_to_grid_centers_ink():
    # A white square in the canvas -> scaled to 20px longest side, centered in
    # 28x28, leaving a >=4px zero border on every edge.
    img = demo.Image.new("L", (demo.CANVAS, demo.CANVAS), 0)
    demo.ImageDraw.Draw(img).rectangle([100, 100, 236, 236], fill=255)
    grid = demo.sketch_to_grid(img)
    assert grid.any()
    assert grid[0].sum() == 0 and grid[demo.GRID - 1].sum() == 0   # top/bottom border
    assert grid[:, 0].sum() == 0 and grid[:, demo.GRID - 1].sum() == 0  # left/right border
    assert grid[demo.GRID // 2, demo.GRID // 2] > 0                # center has ink
