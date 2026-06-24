"""Step 2 of the lowering pipeline: weight processing.

Reads the imported neural network (`/tmp/initial.mlir` and `/tmp/weights.npz`),
quantizes the weights f32 -> Q0.7, maps them to ISA memory addresses, and writes:

  * `/tmp/MEM.bin`        -- quantized weights in Message Protocol MEM format.
  * `/tmp/weight_map.json` -- argument-name -> memory address mapping, used later
                              by the assembler.

Address layout (per `Functional_TPU_ISA_v0.2.md`):
  * 0x0 is hardwired zero, 0x1 is the scratch/temporary designation.
  * The network input is mapped to 0x1 (it is the first temporary value, and the
    assembler reuses 0x1 for every intermediate result).
  * Weights are laid out contiguously starting at 0x2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from Protocol import build_mem_block

# Q0.7 fixed point: 1 sign bit + 7 fractional bits. A value of 1.0 would be 128,
# which overflows int8, so the representable range is [-1.0, 127/128].
Q07_SCALE = 128
Q07_MIN = -128
Q07_MAX = 127

INPUT_LABEL = "__input__"
INPUT_ADDRESS = 0x1
FIRST_WEIGHT_ADDRESS = 0x2

_ARG_RE = re.compile(r"(%arg\d+)\s*:\s*tensor<([^>]*)>(?:\s*loc\(\"([^\"]*)\"\))?")
_FUNC_RE = re.compile(r"func\.func.*?@main\s*\((.*?)\)\s*->", re.DOTALL)


def quantize_q07(values: np.ndarray) -> np.ndarray:
    """Quantize an f32 array to Q0.7, returning int8 values (two's complement).

    Round-to-nearest preserves as much accuracy as possible; values outside the
    representable range are clamped (the closest the format can express).
    """
    scaled = np.round(values.astype(np.float64) * Q07_SCALE)
    clamped = np.clip(scaled, Q07_MIN, Q07_MAX)
    return clamped.astype(np.int8)


def _tensor_shape(type_body: str) -> list[int]:
    """`4x1xf32` -> [4, 1]; `4xf32` -> [4]; `f32` -> []."""
    tokens = type_body.split("x")
    return [int(d) for d in tokens[:-1]]


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

    # State args carry a loc label like "states[i]"; the i indexes into __order__.
    # The remaining arg(s) (loc "inputs[...]" or unlabeled) are the network input.
    weight_map: dict[str, dict] = {}
    regions: list[tuple[int, bytes]] = []  # (address, data) for MEM emission
    next_address = FIRST_WEIGHT_ADDRESS

    for name, shape, label in args:
        state_match = re.match(r"states\[(\d+)\]", label or "")
        if state_match:
            weight_name = order[int(state_match.group(1))]
            quantized = quantize_q07(npz[weight_name].reshape(-1))
            words = _num_words(shape)
            assert len(quantized) == words, (
                f"{weight_name} has {len(quantized)} elements but arg shape {shape} expects {words}."
            )
            data = quantized.astype(np.uint8).tobytes()  # two's complement bytes
            regions.append((next_address, data))
            weight_map[name] = {
                "kind": "weight",
                "label": weight_name,
                "address": next_address,
                "shape": shape,
                "num_words": words,
            }
            next_address += words
        else:
            # Network input: lives at the scratch address, supplied at runtime.
            weight_map[name] = {
                "kind": "input",
                "label": INPUT_LABEL,
                "address": INPUT_ADDRESS,
                "shape": shape,
                "num_words": _num_words(shape),
            }

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
        blob += build_mem_block(address, bytes(data))
    mem_path.write_bytes(bytes(blob))
