"""Golden reference: a bit-exact software model of the TPU's integer inference.

Runs LeNet-5 the way the FPGA runs it -- int8 matmul, dyadic requant, ReLU,
max-pool, channel-major im2col -- in pure numpy, to produce the int8 logits the
hardware *should* emit for a given input. This is the ground truth for isolating
a hardware misclassification:

    (1) float model  ->  (2) int8 golden (this)  ->  (3) FPGA hardware
         correctness         intended quantized          actual output
                             computation

  * (1) vs (2) isolates quantization damage (assembler/calibration).
  * (2) vs (3) isolates execution divergence (RTL).

Fidelity: the requant matches Vector_Processor.v:251-264 exactly -- multiply
first at full width, add rounding bias 1<<(n-1) BEFORE an arithmetic right shift
by n, then saturate to [-128, 127]. Weight quantization and the M0/n dyadic
decomposition reuse the assembler's own functions, so this cannot drift from the
real pipeline. Biases are dropped, matching the deployed network.

Topology (from LeNet_5_Recent.mlir): conv1(1->6,3x3) -> relu -> maxpool(2x2/2)
-> conv2(6->16,3x3) -> relu -> maxpool(2x2/2) -> flatten(16*5*5=400) ->
fc1(400->120) -> relu -> fc2(120->84) -> relu -> fc3(84->10). No padding; unit
conv stride; conv->relu->pool order (ReLU/pool commute, so order is immaterial).

Usage:
    python -m nn_assembler.Emulator [digit_index]     # default 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Emulator lives outside the nn_assembler package (in Interpreter/), so it imports
# the assembler's quantization helpers absolutely (nn_assembler is pip-installed
# editable). Reusing them keeps the golden model in lock-step with the pipeline.
from nn_assembler.Process_Weights import (
    SCALES_KEY_PREFIX,
    SCALES_S_W_INDEX,
    decompose_multiplier,
    quantize_per_tensor_int8,
)

# The five requantized weight layers, in forward order, with their kind.
_CONV_LAYERS = ["conv1.weight", "conv2.weight"]
_FC_LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]

_LENET_DIR = Path(__file__).resolve().parents[2] / "Neural_Networks" / "LeNet_5"
_NPZ_PATH = _LENET_DIR / "LeNet_5_Recent.weights.npz"
_DATASET_ROOT = _LENET_DIR / "dataset.e"

_POOL = 2  # window and stride for both max-pools


def requant(acc: np.ndarray, M0: int, n: int) -> np.ndarray:
    """int32 accumulator -> int8, matching Vector_Processor.v:251-264 exactly.

    result = saturate( (M0*acc + round_bias) >>_arith n ), round_bias = 1<<(n-1)
    (0 when n==0). numpy `>>` on signed int64 is an arithmetic shift, so negative
    accumulators shift identically to the RTL's `>>>` on a signed operand.
    """
    acc = acc.astype(np.int64)
    bias = np.int64(1 << (n - 1)) if n > 0 else np.int64(0)
    shifted = (M0 * acc + bias) >> n
    return np.clip(shifted, -128, 127).astype(np.int32)


def _quantized_weights(npz) -> dict[str, np.ndarray]:
    """int8 weights per layer, quantized with the exported S_w (as the assembler does)."""
    out: dict[str, np.ndarray] = {}
    for name in _CONV_LAYERS + _FC_LAYERS:
        s_w = float(npz[f"{SCALES_KEY_PREFIX}{name}"][SCALES_S_W_INDEX])
        s_w = s_w if s_w != 0.0 else 1.0
        out[name] = quantize_per_tensor_int8(npz[name], s_w)
    return out


def _requant_pairs(npz) -> dict[str, tuple[int, int]]:
    """(M0, n) per layer from the exported multiplier M (as legalization derives)."""
    return {name: decompose_multiplier(float(npz[f"__M__{name}"]))
            for name in _CONV_LAYERS + _FC_LAYERS}


def _conv2d_int(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Valid (no-pad, unit-stride) int cross-correlation. x:[C,H,W], w:[O,C,kh,kw] -> [O,oh,ow] int32."""
    C, H, W = x.shape
    O, Cw, kh, kw = w.shape
    assert Cw == C
    oh, ow = H - kh + 1, W - kw + 1
    x64, w64 = x.astype(np.int64), w.astype(np.int64)
    acc = np.zeros((O, oh, ow), dtype=np.int64)
    for ky in range(kh):
        for kx in range(kw):
            patch = x64[:, ky:ky + oh, kx:kx + ow]  # [C, oh, ow]
            # sum over input channels: w64[:, :, ky, kx] is [O, C]
            acc += np.tensordot(w64[:, :, ky, kx], patch, axes=([1], [0]))  # [O, oh, ow]
    return acc


def _maxpool_int(x: np.ndarray) -> np.ndarray:
    """2x2 stride-2 max-pool per channel. x:[C,H,W] -> [C, H//2, W//2] (geometry is exact)."""
    C, H, W = x.shape
    oh, ow = (H - _POOL) // _POOL + 1, (W - _POOL) // _POOL + 1
    out = np.full((C, oh, ow), -128, dtype=x.dtype)
    for dy in range(_POOL):
        for dx in range(_POOL):
            out = np.maximum(out, x[:, dy:dy + oh * _POOL:_POOL, dx:dx + ow * _POOL:_POOL][:, :oh, :ow])
    return out


def forward_int8(image_int8: np.ndarray, qw: dict, pairs: dict) -> np.ndarray:
    """Integer LeNet-5 forward. image_int8: [1,28,28] int8 -> [10] int8 logits."""
    x = image_int8
    for name in _CONV_LAYERS:
        x = requant(_conv2d_int(x, qw[name]), *pairs[name]).astype(np.int8)  # conv + requant
        x = np.maximum(x, np.int8(0))                                        # relu
        x = _maxpool_int(x)                                                  # pool
    v = x.reshape(-1)  # CHW row-major -> 400, matching fc1's flatten
    for i, name in enumerate(_FC_LAYERS):
        w = qw[name]  # [out, in]
        acc = np.tensordot(w.astype(np.int64), v.astype(np.int64), axes=([1], [0]))  # [out]
        v = requant(acc, *pairs[name]).astype(np.int8)
        if i < len(_FC_LAYERS) - 1:
            v = np.maximum(v, np.int8(0))  # relu after fc1, fc2; not fc3
    return v


def forward_float(image: np.ndarray, npz, use_bias: bool) -> np.ndarray:
    """Reference float forward from the exported float weights (optionally with bias)."""
    def bias(name):
        b = npz[name.replace(".weight", ".bias")]
        return b if use_bias else np.zeros_like(b)

    x = image.astype(np.float64)
    for name in _CONV_LAYERS:
        w = npz[name].astype(np.float64)
        O, _, kh, kw = w.shape
        C, H, W = x.shape
        oh, ow = H - kh + 1, W - kw + 1
        acc = np.zeros((O, oh, ow))
        for ky in range(kh):
            for kx in range(kw):
                acc += np.tensordot(w[:, :, ky, kx], x[:, ky:ky + oh, kx:kx + ow], axes=([1], [0]))
        acc += bias(name)[:, None, None]
        x = np.maximum(acc, 0.0)
        # float 2x2/2 max-pool
        C2, H2, W2 = x.shape
        oh2, ow2 = H2 // 2, W2 // 2
        pooled = np.full((C2, oh2, ow2), -np.inf)
        for dy in range(2):
            for dx in range(2):
                pooled = np.maximum(pooled, x[:, dy:dy + oh2 * 2:2, dx:dx + ow2 * 2:2][:, :oh2, :ow2])
        x = pooled
    v = x.reshape(-1)
    for i, name in enumerate(_FC_LAYERS):
        v = npz[name].astype(np.float64) @ v + bias(name)
        if i < len(_FC_LAYERS) - 1:
            v = np.maximum(v, 0.0)
    return v


def load_mnist(index: int, npz):
    """Return (normalized float image [1,28,28], int8 image, true_label), quantized with conv1 S_in."""
    from torchvision import datasets, transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3018,))]
    )
    dataset = datasets.MNIST(root=str(_DATASET_ROOT), train=False, download=False, transform=transform)
    image, label = dataset[index]
    normalized = image.numpy().astype(np.float64)  # [1,28,28]
    s_in = float(npz[f"{SCALES_KEY_PREFIX}{_CONV_LAYERS[0]}"][0])
    image_int8 = np.clip(np.round(normalized / s_in), -127, 127).astype(np.int8)
    return normalized, image_int8, int(label)


def main() -> None:
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    assert _NPZ_PATH.is_file(), f"Missing {_NPZ_PATH}"
    npz = np.load(_NPZ_PATH, allow_pickle=False)
    qw, pairs = _quantized_weights(npz), _requant_pairs(npz)

    normalized, image_int8, label = load_mnist(index, npz)

    fl_bias = forward_float(normalized, npz, use_bias=True)
    fl_nobias = forward_float(normalized, npz, use_bias=False)
    q = forward_int8(image_int8, qw, pairs)

    print(f"MNIST test[{index}] -- true label: {label}\n")
    print(f"{'idx':>3} {'float+bias':>11} {'float-bias':>11} {'int8 golden':>12}")
    for d in range(10):
        print(f"{d:>3} {fl_bias[d]:>11.3f} {fl_nobias[d]:>11.3f} {int(q[d]):>12}")
    print(f"\nargmax  float+bias={int(np.argmax(fl_bias))}  "
          f"float-bias={int(np.argmax(fl_nobias))}  int8_golden={int(np.argmax(q))}  (true={label})")
    print(f"int8 golden logits: {[int(v) for v in q]}")


if __name__ == "__main__":
    main()
