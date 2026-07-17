"""Minimal interactive LeNet-5 demo: draw a digit, send it to the FPGA.

Opens a small window with a black canvas. Draw a digit with the mouse, press
SEND, and the sketch is downscaled to 28x28, preprocessed EXACTLY as the network
was trained (ToTensor + Normalize((0.1307,), (0.3018,))), quantized to int8 with
conv1's input scale, framed as a Message-Protocol INPUT transmission, and streamed
to the FPGA over the existing PC->Arduino->FPGA link (Send_2_Arduino). The TPU
runs the resident LeNet-5 program and shows the argmax on the VGA/X_OUT output.

This is a deliberately throwaway demo; v0.7.0 replaces it with a proper comms link
and UI.

Prerequisites:
  * LeNet-5 already FLASHed, TPU in EXTERNAL mode (i_mode_select = 1), Arduino
    connected. In device mode the 0x49 INPUT header is silently discarded.
  * Run from the project venv. The GUI needs tkinter:  sudo apt install python3-tk

Usage:
    python System/LeNet-5_Test.py            # draw-and-send GUI
    python System/LeNet-5_Test.py --mnist 0  # headless: send MNIST test[0] (no GUI)

Wire stream:  0x49 (INPUT) | MEM(addr=0x1, len=784, <int8 image>) | 0x53 (STOP)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Loose module (not a package): add its directory before importing.
_HERE = Path(__file__).resolve().parent  # .../System
sys.path.insert(0, str(_HERE / "Communication" / "PC_2_Arduino"))
from Send_2_Arduino import send_2_arduino  # noqa: E402

# nn_assembler is pip-installed (editable); reuse its MEM encoder so the framing
# is byte-identical to the assembler's.
from nn_assembler.Protocol import build_mem_block, STOP  # noqa: E402

INPUT_HEADER = ord("I")  # 0x49 — Functional_TPU_Message_Protocol.md
INPUT_ADDRESS = 0x1      # reserved network I/O staging address

_LENET_DIR = _HERE / "Neural_Networks" / "LeNet_5"
_NPZ_PATH = _LENET_DIR / "LeNet_5_Recent.weights.npz"
_DATASET_ROOT = _LENET_DIR / "dataset.e"
_TX_PATH = _HERE / "LeNet-5_Test_input.bin"  # saved alongside for inspection

# Same normalization as training (Neural_Networks/Start.py). Quantizing anything
# else would put the image on a different scale than S_in was measured against.
_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3018

# Drawing surface: a 28x DIGIT grid shown at SCALE px each; a round brush.
GRID = 28
SCALE = 12
CANVAS = GRID * SCALE  # 336 px
BRUSH = 12             # brush radius in canvas px (~1 grid cell)


def conv1_input_scale(npz_path: Path) -> float:
    """conv1's input scale S_in = __scales__<first-layer>[0] from the export."""
    assert npz_path.is_file(), f"Missing exported weights: {npz_path}"
    npz = np.load(npz_path, allow_pickle=False)
    first_layer = str(npz["__order__"][0])
    return float(npz[f"__scales__{first_layer}"][0])


def sketch_to_grid(canvas_img: Image.Image) -> np.ndarray:
    """Downscale a white-on-black sketch to a 28x28 uint8 array, MNIST-style.

    Crops to the ink bounding box, scales the longest side to 20 px, and centers
    it in a 28x28 field (mirroring MNIST's 20-in-28 layout) so a roughly-drawn
    digit lands in the distribution LeNet-5 was trained on.
    """
    arr = np.asarray(canvas_img, dtype=np.uint8)
    ys, xs = np.where(arr > 32)
    if xs.size == 0:
        return np.zeros((GRID, GRID), dtype=np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = canvas_img.crop((x0, y0, x1 + 1, y1 + 1))
    w, h = crop.size
    scale = 20.0 / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    crop = crop.resize((nw, nh), Image.BILINEAR)
    out = Image.new("L", (GRID, GRID), 0)
    out.paste(crop, ((GRID - nw) // 2, (GRID - nh) // 2))
    return np.asarray(out, dtype=np.uint8)


def quantize(gray28: np.ndarray, s_in: float) -> bytes:
    """28x28 uint8 (white=high) -> 784 int8 two's-complement bytes, CHW row-major."""
    normalized = (gray28.astype(np.float64) / 255.0 - _MNIST_MEAN) / _MNIST_STD
    q = np.clip(np.round(normalized / s_in), -127, 127).astype(np.int8)
    return q.reshape(-1).tobytes()


def build_input_transmission(image_bytes: bytes) -> bytes:
    """INPUT header + one MEM command targeting 0x1 + STOP trailer."""
    return bytes([INPUT_HEADER]) + build_mem_block(INPUT_ADDRESS, image_bytes) + bytes([STOP])


def send_image(gray28: np.ndarray, s_in: float) -> None:
    """Quantize, frame, stage the .bin alongside this script, and stream to the FPGA."""
    transmission = build_input_transmission(quantize(gray28, s_in))
    _TX_PATH.write_bytes(transmission)
    print(f"Wrote {_TX_PATH.name} ({len(transmission)} bytes); streaming...")
    send_2_arduino(_TX_PATH)


def _load_mnist(index: int) -> tuple[np.ndarray, int]:
    """MNIST test[index] as a 28x28 uint8 array (white digit on black) + label."""
    from torchvision import datasets
    dataset = datasets.MNIST(root=str(_DATASET_ROOT), train=False, download=False)
    image, label = dataset[index]  # PIL 'L', 0..255
    return np.asarray(image, dtype=np.uint8), int(label)


class DrawDemo:
    """A black canvas you draw on with the mouse, plus Send/Clear buttons."""

    def __init__(self, root, s_in: float) -> None:
        import tkinter as tk

        self.tk = tk
        self.s_in = s_in
        self.image = Image.new("L", (CANVAS, CANVAS), 0)  # offscreen truth
        self.draw = ImageDraw.Draw(self.image)
        self.last = None
        self._result: list = []  # background-send outcome, polled by _poll

        root.title("LeNet-5 draw demo")
        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "last", None))

        self.send_btn = tk.Button(root, text="SEND", command=self._send)
        self.send_btn.grid(row=1, column=0, sticky="ew")
        tk.Button(root, text="Clear", command=self._clear).grid(row=1, column=1, sticky="ew")
        self.status = tk.StringVar(value="Draw a digit, then press SEND.")
        tk.Label(root, textvariable=self.status, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew")
        self.root = root

    def _press(self, event) -> None:
        self.last = (event.x, event.y)
        self._dot(event.x, event.y)

    def _drag(self, event) -> None:
        if self.last is not None:
            self.canvas.create_line(*self.last, event.x, event.y, fill="white",
                                    width=2 * BRUSH, capstyle=self.tk.ROUND, smooth=True)
            self.draw.line([self.last, (event.x, event.y)], fill=255, width=2 * BRUSH)
        self._dot(event.x, event.y)  # round cap so joints stay solid
        self.last = (event.x, event.y)

    def _dot(self, x: int, y: int) -> None:
        self.canvas.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH,
                                fill="white", outline="white")
        self.draw.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill=255)

    def _clear(self) -> None:
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        self.last = None
        self.status.set("Cleared. Draw a digit, then press SEND.")

    def _send(self) -> None:
        import threading
        grid = sketch_to_grid(self.image)
        if not grid.any():
            self.status.set("Canvas is empty — draw a digit first.")
            return
        self.send_btn.config(state="disabled")
        self.status.set("Sending to FPGA...")
        self._result.clear()

        def worker():
            try:
                send_image(grid, self.s_in)
                self._result.append(("ok", None))
            except Exception as exc:  # serial/link errors surface in the status line
                self._result.append(("err", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self) -> None:
        if not self._result:
            self.root.after(100, self._poll)
            return
        kind, detail = self._result[0]
        self.send_btn.config(state="normal")
        self.status.set("Sent. Read the argmax on the VGA output."
                        if kind == "ok" else f"Send failed: {detail}")


def _run_gui(s_in: float) -> None:
    try:
        import tkinter as tk
    except ModuleNotFoundError:
        raise SystemExit(
            "tkinter is not installed. Install it with:  sudo apt install python3-tk\n"
            "(or run headless without the GUI:  python System/LeNet-5_Test.py --mnist 0)"
        )
    root = tk.Tk()
    DrawDemo(root, s_in)
    root.mainloop()


def main() -> None:
    s_in = conv1_input_scale(_NPZ_PATH)
    if len(sys.argv) > 2 and sys.argv[1] == "--mnist":
        gray28, label = _load_mnist(int(sys.argv[2]))
        print(f"MNIST test[{sys.argv[2]}] -- true label: {label}; conv1 S_in={s_in:.8f}")
        send_image(gray28, s_in)
        print(f"Input sent. Compare the VGA argmax against the true label ({label}).")
    else:
        _run_gui(s_in)


if __name__ == "__main__":
    main()
