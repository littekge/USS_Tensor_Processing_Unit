"""LeNet-5 live demo: program the FPGA, draw a digit, stream it, read the VGA.

Opens a fullscreen window with a black canvas and a live console. First press
**Program device** to assemble and FLASH LeNet-5 onto the TPU (via
`System/run.sh -a -p`) — assembly and flash progress stream into the console.
Then flip the DE1-SoC mode switch to EXTERNAL and confirm. Now draw a digit and
press **SEND**: the sketch is downscaled to 28x28, preprocessed EXACTLY as the
network was trained (ToTensor + Normalize((0.1307,), (0.3018,))), quantized to
int8 with conv1's input scale, framed as a Message-Protocol INPUT transmission,
and streamed to the FPGA over the PC->Arduino->FPGA link. The TPU runs the
resident LeNet-5 program and shows the predicted digit on its VGA output.

There is no PC-side result: the FPGA's MISO line is not wired, so the VGA screen
is the result surface by design.

Prerequisites:
  * Arduino connected (USB) and DE1-SoC powered with a VGA display attached.
  * Run from the project venv. The GUI needs tkinter:  sudo apt install python3-tk
  * `System/run.sh` line 2 (venv path) must match this machine.

Usage:
    python Demo/LeNet5_Demo.py            # fullscreen GUI (Esc to quit)
    python Demo/LeNet5_Demo.py --mnist 0  # headless: send MNIST test[0] (no GUI;
                                          #   assumes already programmed + external)

Wire stream (per SEND):  0x49 (INPUT) | MEM(addr=0x1, len=784, <int8 image>) | 0x53 (STOP)
"""

from __future__ import annotations

import contextlib
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Resolve dependencies under System/ (this file lives in the sibling Demo/ folder).
_HERE = Path(__file__).resolve().parent   # .../Demo
_REPO = _HERE.parent                       # repo root
_SYSTEM = _REPO / "System"

# Send_2_Arduino is a loose module (not a package): add its directory to import it.
sys.path.insert(0, str(_SYSTEM / "Communication" / "PC_2_Arduino"))
from Send_2_Arduino import send_2_arduino, find_arduino_port  # noqa: E402

# nn_assembler is pip-installed (editable); reuse its MEM encoder so the framing
# is byte-identical to the assembler's.
from nn_assembler.Protocol import build_mem_block, STOP  # noqa: E402

INPUT_HEADER = ord("I")  # 0x49 — Functional_TPU_Message_Protocol.md
INPUT_ADDRESS = 0x1      # reserved network I/O staging address

_RUN_SH = _SYSTEM / "run.sh"
_MODEL, _RUN = "LeNet_5", "Recent"

_LENET_DIR = _SYSTEM / "Neural_Networks" / "LeNet_5"
_NPZ_PATH = _LENET_DIR / "LeNet_5_Recent.weights.npz"
_DATASET_ROOT = _LENET_DIR / "dataset.e"
_TX_PATH = _HERE / "LeNet5_Demo_input.bin"       # saved alongside for inspection
_PREVIEW_PATH = _HERE / "LeNet5_Demo_preview.png"  # the 28x28 the network sees
PREVIEW_SCALE = 10  # upscale the tiny preview so pixels are visible

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

    Crops to the ink bounding box, scales the longest side to 20 px, and places
    it in a 28x28 field so the ink's intensity CENTER OF MASS lands at the field
    center -- the centering MNIST itself uses. Centering by the bounding-box
    center instead shifts the digit off MNIST's distribution: on clean MNIST test
    images that single change drops the deployed pipeline from ~94% to ~80%.
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
    crop = np.asarray(crop.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)

    mass = float(crop.sum())
    if mass > 0.0:
        row_idx, col_idx = np.indices(crop.shape)
        cy, cx = (row_idx * crop).sum() / mass, (col_idx * crop).sum() / mass
    else:  # resample wiped the ink (shouldn't happen post-crop): fall back to bbox center
        cy, cx = (nh - 1) / 2.0, (nw - 1) / 2.0
    oy = max(0, min(GRID - nh, int(round(GRID / 2 - cy))))
    ox = max(0, min(GRID - nw, int(round(GRID / 2 - cx))))

    out = np.zeros((GRID, GRID), dtype=np.uint8)
    out[oy:oy + nh, ox:ox + nw] = crop
    return out


def quantize(gray28: np.ndarray, s_in: float) -> bytes:
    """28x28 uint8 (white=high) -> 784 int8 two's-complement bytes, CHW row-major."""
    normalized = (gray28.astype(np.float64) / 255.0 - _MNIST_MEAN) / _MNIST_STD
    q = np.clip(np.round(normalized / s_in), -127, 127).astype(np.int8)
    return q.reshape(-1).tobytes()


def build_input_transmission(image_bytes: bytes) -> bytes:
    """INPUT header + one MEM command targeting 0x1 + STOP trailer."""
    return bytes([INPUT_HEADER]) + build_mem_block(INPUT_ADDRESS, image_bytes) + bytes([STOP])


def save_preview(gray28: np.ndarray) -> None:
    """Save the exact 28x28 the network receives (white digit on black), upscaled."""
    Image.fromarray(gray28, mode="L").resize(
        (GRID * PREVIEW_SCALE, GRID * PREVIEW_SCALE), Image.NEAREST).save(_PREVIEW_PATH)


def send_image(gray28: np.ndarray, s_in: float) -> None:
    """Save a preview, quantize, frame, stage the .bin, and stream to the FPGA."""
    save_preview(gray28)
    transmission = build_input_transmission(quantize(gray28, s_in))
    _TX_PATH.write_bytes(transmission)
    print(f"Saved {_PREVIEW_PATH.name}; wrote {_TX_PATH.name} "
          f"({len(transmission)} bytes); streaming...")
    send_2_arduino(_TX_PATH)


def program_device(on_line=None) -> None:
    """Assemble + FLASH the resident LeNet-5 program via System/run.sh (-a -p).

    Streams the script's combined stdout/stderr line-by-line to `on_line` (so the
    UI can show live assembly + flash progress). Runs with cwd=System/ because
    run.sh uses `./`-relative paths. Raises RuntimeError on a non-zero exit.
    """
    assert _RUN_SH.is_file(), f"Missing programming script: {_RUN_SH}"
    proc = subprocess.Popen(
        ["bash", str(_RUN_SH), "-m", _MODEL, "-r", _RUN, "-a", "-p"],
        cwd=str(_SYSTEM), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:              # live stream until the process closes stdout
        if on_line is not None:
            on_line(line.rstrip("\n"))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"run.sh exited {proc.returncode} (see console above).")


def _load_mnist(index: int) -> tuple[np.ndarray, int]:
    """MNIST test[index] as a 28x28 uint8 array (white digit on black) + label."""
    from torchvision import datasets
    dataset = datasets.MNIST(root=str(_DATASET_ROOT), train=False, download=False)
    image, label = dataset[index]  # PIL 'L', 0..255
    return np.asarray(image, dtype=np.uint8), int(label)


class _LineWriter:
    """A stdout-like sink that forwards completed lines to a callback.

    Used to capture what send_2_arduino prints (progress + result) into the demo
    console. Treats both '\\n' and '\\r' as line breaks so the serial progress
    (which uses carriage returns) shows as discrete lines.
    """

    def __init__(self, emit) -> None:
        self._emit = emit
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while True:
            breaks = [i for i in (self._buf.find("\n"), self._buf.find("\r")) if i >= 0]
            if not breaks:
                break
            cut = min(breaks)
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            if line:
                self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


class DrawDemo:
    """Program the device, then draw digits and stream them to the FPGA.

    A small state machine gates input so the presenter can only draw/send once the
    program is flashed and external mode is confirmed (the INPUT header is silently
    discarded in device mode, so sending earlier would look like a no-op). A
    console panel streams the programming and send output live.
    """

    def __init__(self, root, s_in: float) -> None:
        import tkinter as tk

        self.tk = tk
        self.s_in = s_in
        self.image = Image.new("L", (CANVAS, CANVAS), 0)  # offscreen truth
        self.draw = ImageDraw.Draw(self.image)
        self.last = None
        self.ready = False            # external mode confirmed -> draw/send allowed
        self._result: list = []       # background-op outcome, polled by _poll
        self._log_q: queue.Queue = queue.Queue()  # thread-safe console lines

        root.title("LeNet-5 Live Demo")
        self.root = root

        # Centered content frame so the layout sits mid-screen in fullscreen.
        wrap = tk.Frame(root)
        wrap.place(relx=0.5, rely=0.5, anchor="center")

        instructions = ("1. Program device    2. Flip DE1-SoC to EXTERNAL mode + Confirm    "
                        "3. Draw a digit and SEND  ->  read the prediction on the VGA"
                        "        (Esc to quit)")
        tk.Label(wrap, text=instructions, anchor="w", padx=8, pady=8,
                 font=("TkDefaultFont", 13)).grid(row=0, column=0, columnspan=3, sticky="ew")

        # Drawing canvas in its own labeled frame so it top-aligns with the
        # preview (whose label would otherwise push its canvas lower).
        draw_frame = tk.Frame(wrap)
        draw_frame.grid(row=1, column=0, padx=8, pady=4)  # centered vertically in the row
        tk.Label(draw_frame, text="Draw a digit (28x28)").pack()
        self.canvas = tk.Canvas(draw_frame, width=CANVAS, height=CANVAS, bg="black",
                                highlightthickness=1, highlightbackground="#444",
                                cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        # Live preview of the exact 28x28 the network will receive.
        prev_px = GRID * PREVIEW_SCALE
        prev_frame = tk.Frame(wrap)
        prev_frame.grid(row=1, column=1, padx=8, pady=4)  # centered vertically in the row
        tk.Label(prev_frame, text="Network input (28x28)").pack()
        self.preview = tk.Canvas(prev_frame, width=prev_px, height=prev_px, bg="black",
                                 highlightthickness=1, highlightbackground="#444")
        self.preview.pack()

        # Console: live stream of programming (assembly + flash) and send output.
        term_frame = tk.Frame(wrap)
        term_frame.grid(row=1, column=2, padx=8, pady=4, sticky="ns")
        tk.Label(term_frame, text="Console").pack(anchor="w")
        scroll = tk.Scrollbar(term_frame)
        scroll.pack(side="right", fill="y")
        self.term = tk.Text(term_frame, width=58, height=22, wrap="none",
                            bg="#101010", fg="#d0d0d0", insertbackground="#d0d0d0",
                            font=("TkFixedFont", 10), state="disabled",
                            yscrollcommand=scroll.set)
        self.term.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.term.yview)

        self.program_btn = tk.Button(wrap, text="Program device", command=self._program)
        self.program_btn.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        self.confirm_btn = tk.Button(wrap, text="Confirm external mode",
                                     command=self._confirm_external, state="disabled")
        self.confirm_btn.grid(row=2, column=1, sticky="ew", padx=8, pady=2)

        self.send_btn = tk.Button(wrap, text="SEND", command=self._send, state="disabled")
        self.send_btn.grid(row=3, column=0, sticky="ew", padx=8, pady=2)
        self.clear_btn = tk.Button(wrap, text="Clear", command=self._clear, state="disabled")
        self.clear_btn.grid(row=3, column=1, sticky="ew", padx=8, pady=2)

        self.status = tk.StringVar()
        tk.Label(wrap, textvariable=self.status, anchor="w", padx=8, pady=8,
                 font=("TkDefaultFont", 13)).grid(row=4, column=0, columnspan=3, sticky="ew")

        port = find_arduino_port()
        self.status.set("Arduino detected. Step 1: press Program device."
                        if port else
                        "No Arduino detected — connect it, then press Program device.")

        self.root.after(100, self._drain_log)  # start pumping console lines

    # --- console ---------------------------------------------------------------
    def _log(self, text: str) -> None:
        """Thread-safe: queue a console line (workers call this)."""
        self._log_q.put(text)

    def _drain_log(self) -> None:
        """Main-thread: flush queued console lines into the Text widget."""
        try:
            while True:
                line = self._log_q.get_nowait()
                self.term.config(state="normal")
                self.term.insert("end", line + "\n")
                self.term.see("end")
                self.term.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    # --- drawing ---------------------------------------------------------------
    def _press(self, event) -> None:
        if not self.ready:
            return
        self.last = (event.x, event.y)
        self._dot(event.x, event.y)

    def _drag(self, event) -> None:
        if not self.ready:
            return
        if self.last is not None:
            self.canvas.create_line(*self.last, event.x, event.y, fill="white",
                                    width=2 * BRUSH, capstyle=self.tk.ROUND, smooth=True)
            self.draw.line([self.last, (event.x, event.y)], fill=255, width=2 * BRUSH)
        self._dot(event.x, event.y)  # round cap so joints stay solid
        self.last = (event.x, event.y)

    def _release(self, _event) -> None:
        self.last = None
        if self.ready:
            self._update_preview()

    def _dot(self, x: int, y: int) -> None:
        self.canvas.create_oval(x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH,
                                fill="white", outline="white")
        self.draw.ellipse([x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH], fill=255)

    def _update_preview(self) -> None:
        grid = sketch_to_grid(self.image)
        self.preview.delete("all")
        ps = PREVIEW_SCALE
        for y in range(GRID):
            for x in range(GRID):
                v = int(grid[y, x])
                if v:
                    shade = f"#{v:02x}{v:02x}{v:02x}"
                    self.preview.create_rectangle(x * ps, y * ps, x * ps + ps, y * ps + ps,
                                                  fill=shade, outline="")

    def _clear(self) -> None:
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        self.preview.delete("all")
        self.last = None
        self.status.set("Cleared. Draw a digit, then press SEND.")

    # --- device programming ----------------------------------------------------
    def _program(self) -> None:
        self._set_controls(programming=True)
        self.status.set("Assembling + flashing LeNet-5 (watch the console)...")
        self._log("=== Program device: run.sh -m LeNet_5 -r Recent -a -p ===")
        self._result.clear()

        def worker():
            try:
                program_device(on_line=self._log)
                self._result.append(("programmed", None))
            except BaseException as exc:  # noqa: BLE001 — run.sh/serial errors surface in-UI
                self._log(f"[error] {exc}")
                self._result.append(("err", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(150, self._poll)

    def _confirm_external(self) -> None:
        self.ready = True
        self.confirm_btn.config(state="disabled")
        self.send_btn.config(state="normal")
        self.clear_btn.config(state="normal")
        self.program_btn.config(state="normal")  # allow re-flash if needed
        self.status.set("Ready. Draw a digit and press SEND, then read the VGA.")

    # --- sending ---------------------------------------------------------------
    def _send(self) -> None:
        grid = sketch_to_grid(self.image)
        if not grid.any():
            self.status.set("Canvas is empty — draw a digit first.")
            return
        self._set_controls(sending=True)
        self.status.set("Sending to FPGA (watch the console)...")
        self._log("=== SEND: streaming INPUT to 0x1 ===")
        self._result.clear()

        def worker():
            try:
                # Capture what send_image / send_2_arduino print into the console.
                with contextlib.redirect_stdout(_LineWriter(self._log)):
                    send_image(grid, self.s_in)
                self._result.append(("sent", None))
            except BaseException as exc:  # noqa: BLE001 — send_2_arduino raises SystemExit on framing failure
                self._log(f"[error] {exc}")
                self._result.append(("err", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll)

    # --- shared background-op polling / control state --------------------------
    def _set_controls(self, *, programming: bool = False, sending: bool = False) -> None:
        if programming or sending:
            self.program_btn.config(state="disabled")
            self.send_btn.config(state="disabled")
            self.clear_btn.config(state="disabled")
            self.confirm_btn.config(state="disabled")

    def _poll(self) -> None:
        if not self._result:
            self.root.after(100, self._poll)
            return
        kind, detail = self._result[0]
        if kind == "programmed":
            self.confirm_btn.config(state="normal")
            self.program_btn.config(state="normal")
            self.status.set("Flashed. Flip the DE1-SoC switch to EXTERNAL mode, "
                            "then press Confirm external mode.")
        elif kind == "sent":
            self.send_btn.config(state="normal")
            self.clear_btn.config(state="normal")
            self.program_btn.config(state="normal")
            self.status.set(f"Sent (preview: {_PREVIEW_PATH.name}). "
                            "Read the predicted digit on the VGA.")
        else:  # err
            self.program_btn.config(state="normal")
            if self.ready:
                self.send_btn.config(state="normal")
                self.clear_btn.config(state="normal")
            self.status.set(f"Failed: {detail} (see console).")


def _run_gui(s_in: float) -> None:
    try:
        import tkinter as tk
    except ModuleNotFoundError:
        raise SystemExit(
            "tkinter is not installed. Install it with:  sudo apt install python3-tk\n"
            "(or run headless without the GUI:  python Demo/LeNet5_Demo.py --mnist 0)"
        )
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _e: root.destroy())  # Esc exits the demo
    DrawDemo(root, s_in)
    root.mainloop()


def main() -> None:
    s_in = conv1_input_scale(_NPZ_PATH)
    if len(sys.argv) > 2 and sys.argv[1] == "--mnist":
        # Headless test path: assumes the device is already programmed and in
        # external mode (no run.sh call here).
        gray28, label = _load_mnist(int(sys.argv[2]))
        print(f"MNIST test[{sys.argv[2]}] -- true label: {label}; conv1 S_in={s_in:.8f}")
        send_image(gray28, s_in)
        print(f"Input sent. Compare the VGA prediction against the true label ({label}).")
    else:
        _run_gui(s_in)


if __name__ == "__main__":
    main()
