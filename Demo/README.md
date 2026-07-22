# LeNet-5 Live Demo

Draw a digit on the PC, stream it to the Functional TPU over the
PC → Arduino → FPGA link, and read the prediction on the FPGA's VGA output.

This is the v0.7 presentation demo (`LeNet5_Demo.py`), grown from the earlier
throwaway `LeNet-5_Test.py`.

## Demo story

- **Input:** a digit drawn on the PC (or an MNIST test image, headless).
- **Compute:** the resident LeNet-5 program on the TPU.
- **Output:** the predicted digit on the FPGA's **VGA** screen. There is no
  PC-side result — the MISO line is not physically wired, so the FPGA cannot
  report back. The VGA screen is the result surface, by design.

## Presenter flow

1. **Connect** the Arduino (USB) and the DE1-SoC; power on; VGA display attached.
2. **Launch:** from the repo, `python Demo/LeNet5_Demo.py` (needs the project
   venv and `python3-tk`).
3. **Program device** (button): assembles + FLASHes LeNet-5 by calling
   `System/run.sh -m LeNet_5 -r Recent -a -p`.
4. **Switch to external mode:** when prompted, flip the DE1-SoC mode switch to
   EXTERNAL (`i_mode_select = 1`), then click **Confirm**. The INPUT header
   (`0x49`) is silently discarded in device mode, so drawing is disabled until
   this is confirmed.
5. **Draw & send:** draw a digit, press **SEND**, read the prediction on the VGA.
   Repeat as needed (**Clear** to reset the canvas).

Headless test (no GUI, still needs the link + external mode):
`python Demo/LeNet5_Demo.py --mnist 0` sends MNIST test image 0.

## Preprocessing (must match training)

Sketch → crop to ink bbox → longest side scaled to 20 px → centered in 28×28
(MNIST's 20-in-28 layout) → normalize with the training mean/std
(`0.1307` / `0.3018`) → int8 quantize with conv1's input scale `S_in`
(read from `LeNet_5_Recent.weights.npz`) → frame as
`0x49 INPUT | MEM(0x1, 784) | 0x53 STOP`. Framing reuses `nn_assembler.Protocol`
so the bytes are identical to the assembler's.

Re-assembling on **Program device** guarantees the flashed program and the
demo's `S_in` come from the same artifact.

## Design notes / scope

- **Setup and inference in one window.** A "Program device" button shells out to
  `run.sh` (cwd `System/`); a modal walks the presenter through the manual
  external-mode switch (the PC cannot toggle it) before enabling draw/send.
- **Window:** fullscreen (Esc to quit); live 28×28 preview of what the network
  sees; prominent status line.
- **Console panel:** streams the programming output live (assembly steps, flash
  ACK progress, success) and the per-send output, so the audience sees the work
  happening. Programming is streamed line-by-line from `run.sh` (`Popen`); send
  output is captured from `send_2_arduino`'s prints via a stdout redirect.
- **Robust link handling:** `send_2_arduino` raises `SystemExit` on a framing
  failure and returns `None` port when no Arduino is found — the GUI catches
  `BaseException` and checks the port up front so link problems show in the
  status line instead of crashing.
- **Serial:** reuses `Send_2_Arduino` as-is (opens/closes per send, ~2 s Arduino
  reset each time). Acceptable for the demo; a persistent connection was
  deliberately deferred.
- **Out of scope:** PC-side result readback (MISO unwired) and extra input modes
  (MNIST browser). The `--mnist N` headless path is kept for testing only.

## Layout

- `LeNet5_Demo.py` — the demo (GUI + headless).
- Pure functions (`sketch_to_grid`, `quantize`, `build_input_transmission`) are
  unit-testable; GUI + serial + `run.sh` orchestration are hardware/manual.

## Implementation checklist

1. Rename `LeNet-5_Test.py` → `LeNet5_Demo.py`; fix paths for the `Demo/`
   location (`_REPO = _HERE.parent`, resolve `System/…`); refresh docstring/usage.
2. Add the "Program device" button + `run.sh` worker + external-mode confirm gate
   (draw/SEND disabled until confirmed).
3. Harden: up-front Arduino-connection check; catch `SystemExit`/`BaseException`
   on send; in-UI error surfacing.
4. Presentation polish: large windowed layout, live 28×28 preview, clear
   instructions + status.
5. Tests: pure-function unit tests (framing byte-exactness vs `nn_assembler`),
   plus a manual hardware checklist.
