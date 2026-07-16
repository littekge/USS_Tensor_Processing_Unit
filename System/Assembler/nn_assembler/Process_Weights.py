"""Step 2 of the lowering pipeline: weight processing.

Reads the imported neural network (`/tmp/initial.mlir` and `/tmp/weights.npz`),
quantizes the weights f32 -> per-tensor symmetric int8, decomposes each layer's
requantization multiplier `M` into the ISA dyadic pair `M0 * 2^-n`, maps every
tensor to an ISA memory address, and writes:

  * `/tmp/MEM.bin`         -- quantized weights in Message Protocol MEM format.
  * `/tmp/weight_map.json` -- argument-name -> memory address mapping (plus each
                              weight layer's `M0`/`n`), used later by the
                              assembler.

Quantization (v0.3): weights use per-tensor symmetric int8 with scale
`S_w = absmax(W)/127`, read from the npz `__scales__<name>` array so the
quantized weight values stay consistent with the exported requantization
multiplier `M` (which was computed with that same `S_w`). Biases are still
quantized and mapped so weight addresses stay stable, but their scale is
irrelevant -- they are dropped from the instruction stream downstream.

Address layout (per `Functional_TPU_ISA.md`, v0.4):
  * 0x0 is hardwired zero; 0x1 is the reserved network I/O designation.
  * The network input is mapped to 0x1, and the network's final output is written
    back to 0x1 (see `FINAL_OUTPUT_ADDRESS`) so the programmer can read it.
  * Weights and biases are laid out contiguously starting at 0x2, in `__order__`.
  * Intermediate results live in main data memory past the weight region; the
    assembler assigns them via `IntermediateAllocator`, reusing freed regions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .MLIR.transpose_analysis import MANIFEST_NAME
from .Protocol import build_mem_blocks

# Per-tensor symmetric int8: values map through S so that round(v/S) lands in
# [-127, 127]. -128 is intentionally excluded to keep the range symmetric about
# zero (matching the exported scale S = absmax/127).
INT8_SYMMETRIC_MIN = -127
INT8_SYMMETRIC_MAX = 127

# Width of the ISA MUL instruction's M0 (mantissa) field, in bits. The dyadic
# decomposition of M targets this precision, so it must match the ISA MUL format
# (bits 59-52). See Functional_TPU_ISA.md.
REQUANT_MANTISSA_BITS = 8

INPUT_LABEL = "__input__"
INPUT_ADDRESS = 0x1
# The network's final output is written back to the reserved I/O designation so
# the programmer can read it. 0x1 is architecturally isolated (ISA), and by the
# time the last op runs the input it aliases is already consumed.
FINAL_OUTPUT_ADDRESS = 0x1
FIRST_WEIGHT_ADDRESS = 0x2

# npz metadata key prefixes written by the neural-network export
# (Neural_Networks/Operations.py). `__M__<name>` is the scalar requantization
# multiplier; `__scales__<name>` is [S_in, S_w, S_out]. Their presence is what
# distinguishes a requantized weight tensor from a bias tensor.
M_KEY_PREFIX = "__M__"
SCALES_KEY_PREFIX = "__scales__"
SCALES_S_W_INDEX = 1  # __scales__ layout is [S_in, S_w, S_out]

_ARG_RE = re.compile(r"(%arg\d+)\s*:\s*tensor<([^>]*)>(?:\s*loc\(\"([^\"]*)\"\))?")
_FUNC_RE = re.compile(r"func\.func.*?@main\s*\((.*?)\)\s*->", re.DOTALL)


def quantize_per_tensor_int8(values: np.ndarray, scale: float) -> np.ndarray:
    """Quantize an f32 array to symmetric int8: `clamp(round(v/scale), -127, 127)`.

    `scale` is the tensor's `S = absmax/127`. Round-to-nearest preserves accuracy;
    values outside the representable range are clamped to the nearest int8.
    """
    assert scale > 0.0, f"Quantization scale must be positive, got {scale}."
    scaled = np.round(values.astype(np.float64) / scale)
    clamped = np.clip(scaled, INT8_SYMMETRIC_MIN, INT8_SYMMETRIC_MAX)
    return clamped.astype(np.int8)


def decompose_multiplier(M: float, mantissa_bits: int = REQUANT_MANTISSA_BITS) -> tuple[int, int]:
    """Decompose a requantization multiplier `M` into the ISA dyadic pair `(M0, n)`.

    The ISA applies `result = round(M0 * x / 2^n)`, so `M ~= M0 * 2^-n`. The
    mantissa `M0` is normalized to `[2^(B-1), 2^B - 1]` (i.e. `[128, 255]` for
    B=8), matching the MUL instruction's unsigned 8-bit `M0`/`n` fields.
    """
    assert M > 0.0, (
        f"Requantization multiplier must be positive, got M={M}. "
        "M == 0 indicates a dead layer (zero output range over the calibration set)."
    )
    B = mantissa_bits

    # Normalize M into m in [0.5, 1): M = m * 2^e, so M0 = round(m * 2^B) lands in
    # [2^(B-1), 2^B] and n = B - e is the accompanying right-shift.
    e = math.floor(math.log2(M)) + 1
    m = M * (2.0**-e)
    M0 = round(m * (2**B))
    n = B - e

    # Renormalize the boundary case m -> 1.0 (M0 rounds up to 2^B, one bit too
    # wide) back into the mantissa range. Halving M0 and decrementing n is lossless.
    if M0 == (1 << B):
        M0 = 1 << (B - 1)
        n -= 1

    assert (1 << (B - 1)) <= M0 <= (1 << B) - 1, (
        f"Decomposed mantissa M0={M0} outside [{1 << (B - 1)}, {(1 << B) - 1}] for M={M}."
    )
    assert 0 <= n <= 255, (
        f"Decomposed shift n={n} outside the 8-bit ISA field for M={M}. "
        "n < 0 implies M >= 256, i.e. a degenerate/mis-calibrated layer."
    )
    return M0, n


def _tensor_shape(type_body: str) -> list[int]:
    """`4x1xf32` -> [4, 1]; `4xf32` -> [4]; `f32` -> []."""
    tokens = type_body.split("x")
    return [int(d) for d in tokens[:-1]]


def flatten_conv_weight_shape(shape: list[int]) -> list[int]:
    """Flatten a 4-D conv weight shape `(out_ch, in_ch, kh, kw)` to `[out_ch, K]`.

    `K = in_ch * kh * kw`. The row-major flatten of a PyTorch `(out, in, kh, kw)`
    kernel already orders K channel-major (`k = c*(kh*kw) + ki*kw + kj`), which is
    exactly the order `im2col` gathers its columns -- so the flatten is a pure
    reshape with NO permutation. A mismatch here would silently compute the wrong
    convolution, so the order must not change.
    """
    assert len(shape) == 4, f"Expected a 4-D conv weight shape, got {shape}."
    out_ch, in_ch, kh, kw = shape
    return [out_ch, in_ch * kh * kw]


def _parse_main_args(mlir_text: str) -> list[tuple[str, list[int], str]]:
    """Return [(arg_name, shape, loc_label), ...] for @main's arguments."""
    func_match = _FUNC_RE.search(mlir_text)
    assert func_match is not None, "Could not find @main signature in initial.mlir."

    args: list[tuple[str, list[int], str]] = []
    for name, type_body, label in _ARG_RE.findall(func_match.group(1)):
        args.append((name, _tensor_shape(type_body), label))
    assert args, "No arguments found in @main signature."
    return args


def _num_words(shape: list[int]) -> int:
    """Number of memory words a tensor occupies (its flattened element count)."""
    count = 1
    for dim in shape:
        count *= dim
    return count


def first_free_address(weight_map: dict) -> int:
    """First main-memory address past the contiguous weight/bias region.

    Intermediate results are allocated from here upward so they never collide
    with resident weights or biases. Falls back to `FIRST_WEIGHT_ADDRESS` when
    the network has no mapped weights.
    """
    end = FIRST_WEIGHT_ADDRESS
    for entry in weight_map.values():
        if entry.get("kind") in ("weight", "bias"):
            words = entry.get("num_words", _num_words(entry.get("shape", [])))
            end = max(end, entry["address"] + words)
    return end


class IntermediateAllocator:
    """Assigns main-memory addresses to intermediate tensor results.

    Regions returned via `free` are reused by later `allocate` calls (simple
    scope-based reuse -- the caller frees a value once its last consuming op has
    run; there is no cross-branch liveness analysis yet). A fresh high-water
    address is used only when no freed region is large enough.
    """

    def __init__(self, base_address: int) -> None:
        assert base_address >= FIRST_WEIGHT_ADDRESS, (
            f"Intermediate base {base_address} overlaps reserved addresses (<0x2)."
        )
        self._next = base_address
        self._free: list[tuple[int, int]] = []  # (address, num_words), address-sorted

    def allocate(self, num_words: int) -> int:
        """Reserve `num_words` contiguous words; reuse a freed region if one fits."""
        assert num_words > 0, f"Cannot allocate {num_words} words."
        for index, (address, size) in enumerate(self._free):
            if size >= num_words:
                self._free.pop(index)
                if size > num_words:
                    self._free.append((address + num_words, size - num_words))
                    self._merge()
                return address
        address = self._next
        self._next += num_words
        return address

    def free(self, address: int, num_words: int) -> None:
        """Return a region to the free list for reuse by later allocations."""
        self._free.append((address, num_words))
        self._merge()

    def _merge(self) -> None:
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for address, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == address:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((address, size))
        self._free = merged


def _load_transpose_manifest(tmp_dir: Path) -> dict[str, dict]:
    """Map arg name -> transpose entry for weights resolved offline (v0.5).

    Reads `/tmp/transpose_manifest.json` (written by the transpose-analysis
    stage). Absent manifest means no transposes were flagged (e.g. networks with
    only trivial transposes, which the StableHLO optimizer folds to reshapes).
    """
    manifest_path = tmp_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    return {entry["operand"]: entry for entry in manifest.get("weight_transposes", [])}


def _weight_scale(npz, weight_name: str) -> float:
    """Per-tensor symmetric int8 scale `S_w` for a requantized weight tensor.

    Read from the exported `__scales__<name>` (index 1) so the quantized weights
    stay consistent with the exported multiplier `M`. An all-zero weight has
    `S_w == 0`; fall back to 1.0 (every value quantizes to 0 regardless).
    """
    scales_key = f"{SCALES_KEY_PREFIX}{weight_name}"
    assert scales_key in npz.files, (
        f"Weight {weight_name} has a {M_KEY_PREFIX}{weight_name} entry but no "
        f"{scales_key}; the export is inconsistent."
    )
    S_w = float(npz[scales_key][SCALES_S_W_INDEX])
    return S_w if S_w != 0.0 else 1.0


def _self_scale(values: np.ndarray) -> float:
    """Symmetric int8 self-scale `absmax/127` for a tensor (used for biases)."""
    absmax = float(np.abs(values).max()) if values.size else 0.0
    return absmax / 127.0 if absmax != 0.0 else 1.0


def Process_Weights(tmp_dir: Path | None = None) -> dict:
    """Quantize and map weights, writing MEM.bin and weight_map.json.

    Returns the weight map dict (also written to disk) for convenience/testing.
    """
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"

    mlir_path = tmp_dir / "initial.mlir"
    weights_path = tmp_dir / "weights.npz"
    assert mlir_path.is_file(), f"Missing {mlir_path}"
    assert weights_path.is_file(), f"Missing {weights_path}"

    args = _parse_main_args(mlir_path.read_text())
    npz = np.load(weights_path, allow_pickle=False)
    order = [str(name) for name in npz["__order__"]]
    transpose_by_arg = _load_transpose_manifest(tmp_dir)

    # State args carry a loc label like "states[i]"; the i indexes into __order__.
    # The remaining arg(s) (loc "inputs[...]" or unlabeled) are the network input.
    weight_map: dict[str, dict] = {}
    regions: list[tuple[int, bytes]] = []  # (address, data) for MEM emission
    next_address = FIRST_WEIGHT_ADDRESS

    for name, shape, label in args:
        state_match = re.match(r"states\[(\d+)\]", label or "")
        if not state_match:
            # Network input: lives at the scratch address, supplied at runtime.
            weight_map[name] = {
                "kind": "input",
                "label": INPUT_LABEL,
                "address": INPUT_ADDRESS,
                "shape": shape,
                "num_words": _num_words(shape),
            }
            continue

        weight_name = order[int(state_match.group(1))]
        array = npz[weight_name]

        # A weight fed through a graph transpose is stored transposed offline
        # (Wᵀ, row-major) so the device reads the correct layout without a runtime
        # transpose. Per-tensor symmetric int8 scale is transpose-invariant, so
        # transposing before quantization is numerically safe.
        transpose = transpose_by_arg.get(name)
        if transpose is not None:
            array = np.transpose(array, transpose["perm"])
            shape = transpose["out_shape"]

        # A 4-D conv kernel is a matmul operand once im2col produces the columns:
        # store it as the 2-D matrix (out_ch) x (in_ch*kh*kw). The row-major flatten
        # is channel-major with no permutation (see flatten_conv_weight_shape), so
        # the physical bytes are unchanged -- only the recorded shape becomes 2-D.
        if len(shape) == 4:
            shape = flatten_conv_weight_shape(shape)

        values = array.reshape(-1)
        words = _num_words(shape)
        assert len(values) == words, (
            f"{weight_name} has {len(values)} elements but arg shape {shape} expects {words}."
        )

        # A tensor is a requantized weight iff the export wrote its multiplier M.
        # Everything else in __order__ (the *.bias tensors) is quantized only to
        # keep addresses stable and carries no M0/n.
        is_weight = f"{M_KEY_PREFIX}{weight_name}" in npz.files
        if is_weight:
            scale = _weight_scale(npz, weight_name)
            quantized = quantize_per_tensor_int8(values, scale)
            M = float(npz[f"{M_KEY_PREFIX}{weight_name}"])
            M0, n = decompose_multiplier(M)
            entry = {
                "kind": "weight",
                "label": weight_name,
                "address": next_address,
                "shape": shape,
                "num_words": words,
                "M0": M0,
                "n": n,
            }
        else:
            quantized = quantize_per_tensor_int8(values, _self_scale(values))
            entry = {
                "kind": "bias",
                "label": weight_name,
                "address": next_address,
                "shape": shape,
                "num_words": words,
            }

        data = quantized.astype(np.uint8).tobytes()  # two's complement bytes
        regions.append((next_address, data))
        weight_map[name] = entry
        next_address += words

    _write_mem(tmp_dir / "MEM.bin", regions)
    (tmp_dir / "weight_map.json").write_text(json.dumps(weight_map, indent=2))
    return weight_map


def _write_mem(mem_path: Path, regions: list[tuple[int, bytes]]) -> None:
    """Merge contiguous (address, data) regions and write them as MEM blocks."""
    regions = sorted(regions, key=lambda r: r[0])

    merged: list[tuple[int, bytearray]] = []
    for address, data in regions:
        if merged and merged[-1][0] + len(merged[-1][1]) == address:
            merged[-1][1].extend(data)
        else:
            merged.append((address, bytearray(data)))

    blob = bytearray()
    for address, data in merged:
        blob += build_mem_blocks(address, bytes(data))
    mem_path.write_bytes(bytes(blob))
