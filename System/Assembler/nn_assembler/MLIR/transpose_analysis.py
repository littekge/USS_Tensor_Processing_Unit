"""Post-import transpose analysis (lowering step 1.5, v0.5).

Scans the freshly imported `/tmp/initial.mlir` for `stablehlo.transpose` ops and
classifies each one:

  * **Weight / constant transpose** -- the transposed value is a network
    *parameter* (a `@main` argument whose loc is `states[...]`). These are known
    offline, so the transpose is *resolved offline*: `Process_Weights` physically
    stores the transposed weight (`Wᵀ`, row-major) and the legalizer folds the
    transpose op away (it becomes a pure reinterpretation of already-transposed
    memory).
  * **Runtime / live transpose** -- the transposed value is the network input
    (`inputs[...]`) or an intermediate result (`%N`). Transposing runtime data on
    the device is out of scope for v0.5; such a transpose is *flagged* (recorded
    in the manifest) rather than silently dropped, so the limitation is explicit.

The result is written to `/tmp/transpose_manifest.json` and consumed by
`Process_Weights` (which transposes the flagged weights) and, indirectly, by the
legalizer (which folds resolved transposes).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_FUNC_RE = re.compile(r"func\.func.*?@main\s*\((.*?)\)\s*->", re.DOTALL)
_ARG_RE = re.compile(r"(%arg\d+)\s*:\s*tensor<([^>]*)>(?:\s*loc\(\"([^\"]*)\"\))?")
_TRANSPOSE_RE = re.compile(
    r"(%[\w]+)\s*=\s*stablehlo\.transpose\s+(%[\w]+)\s*,\s*dims\s*=\s*\[([\d,\s]*)\]\s*:\s*"
    r"\(tensor<([^>]*)>\)\s*->\s*tensor<([^>]*)>"
)

MANIFEST_NAME = "transpose_manifest.json"


def _tensor_shape(type_body: str) -> list[int]:
    """`100x1000xf32` -> [100, 1000]; `4xf32` -> [4]; `f32` -> []."""
    return [int(d) for d in type_body.split("x")[:-1]]


def _parse_arg_locs(mlir_text: str) -> dict[str, str]:
    """Map each `@main` argument SSA name to its loc label (may be empty)."""
    func_match = _FUNC_RE.search(mlir_text)
    assert func_match is not None, "Could not find @main signature in initial.mlir."
    return {name: (label or "") for name, _ty, label in _ARG_RE.findall(func_match.group(1))}


def analyze_transposes(mlir_text: str) -> dict:
    """Classify every `stablehlo.transpose` in `mlir_text` into the manifest dict.

    A transpose whose operand is a parameter argument (`states[...]` loc) is a
    weight transpose (resolved offline); anything else (the network input, or an
    intermediate `%N`) is a runtime transpose (flagged, out of scope).
    """
    arg_locs = _parse_arg_locs(mlir_text)

    weight_transposes: list[dict] = []
    runtime_transposes: list[dict] = []
    for result, operand, perm_body, in_ty, out_ty in _TRANSPOSE_RE.findall(mlir_text):
        perm = [int(p) for p in perm_body.split(",") if p.strip()]
        entry = {
            "result": result,
            "operand": operand,
            "perm": perm,
            "in_shape": _tensor_shape(in_ty),
            "out_shape": _tensor_shape(out_ty),
        }
        loc = arg_locs.get(operand)
        is_parameter = loc is not None and loc.startswith("states[")
        (weight_transposes if is_parameter else runtime_transposes).append(entry)

    return {"weight_transposes": weight_transposes, "runtime_transposes": runtime_transposes}


def analyze_transposes_file(tmp_dir: Path | None = None) -> dict:
    """Analyze `/tmp/initial.mlir` and write `/tmp/transpose_manifest.json`."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent.parent / "tmp"

    initial_path = tmp_dir / "initial.mlir"
    assert initial_path.is_file(), f"Missing {initial_path}"

    manifest = analyze_transposes(initial_path.read_text())
    (tmp_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest
