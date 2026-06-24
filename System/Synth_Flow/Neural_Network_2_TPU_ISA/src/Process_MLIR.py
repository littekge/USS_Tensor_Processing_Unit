"""First half of lowering step 3: MLIR lowering.

Runs the two MLIR passes in sequence:
  1. StableHLO target-independent optimization (`/tmp/initial.mlir` ->
     `/tmp/optimized.mlir`) via the `stablehlo-opt` binary.
  2. Legalization to the *Functional TPU* dialect (`/tmp/optimized.mlir` ->
     `/tmp/optimized.tpu.mlir`), defined in `src/MLIR/legalize.py`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from MLIR.legalize import legalize


def Process_MLIR(tmp_dir: Path | None = None) -> None:
    """Optimize the StableHLO graph, then legalize it to the TPU dialect."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"

    initial_path = tmp_dir / "initial.mlir"
    optimized_path = tmp_dir / "optimized.mlir"
    tpu_path = tmp_dir / "optimized.tpu.mlir"
    assert initial_path.is_file(), f"Missing {initial_path}"

    try:
        subprocess.run(
            [
                "stablehlo-opt",
                "--stablehlo-target-independent-optimization",
                str(initial_path),
                "-o",
                str(optimized_path),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "stablehlo-opt not found on PATH. Install it or add it to PATH "
            "(see CLAUDE.md: ~/Programs/stablehlo/build/bin/)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"stablehlo-opt failed on {initial_path}: {exc}") from exc

    assert optimized_path.is_file(), f"stablehlo-opt did not produce {optimized_path}"
    legalize(optimized_path, tpu_path)
