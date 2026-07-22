"""First half of lowering step 3: MLIR lowering.

Runs the MLIR passes in sequence:
  1. StableHLO target-independent optimization (`/tmp/initial.mlir` ->
     `/tmp/optimized.mlir`) via the `stablehlo-opt` binary.
  2. Legalization to the *Functional TPU* dialect (`/tmp/optimized.mlir` ->
     `/tmp/optimized.tpu.mlir`), annotating each `tpu.mult` with its layer's
     `M0`/`n` from `/tmp/weight_map.json`. Defined in `MLIR/legalize.py`.
  3. Bias removal (`/tmp/optimized.tpu.mlir` -> `/tmp/optimized.nobias.tpu.mlir`),
     dropping bias `add`s. Defined in `MLIR/bias_removal.py`.
  4. Matmul partitioning (`/tmp/optimized.nobias.tpu.mlir` ->
     `/tmp/optimized.partitioned.tpu.mlir`), tiling oversized matmuls into
     array-sized `mult`/`multip` runs with a leading `stride`. Defined in
     `MLIR/partition.py`.
  5. Output staging (`/tmp/optimized.partitioned.tpu.mlir` ->
     `/tmp/optimized.staged.tpu.mlir`), inserting a `tpu.move %out -> @io` before
     the terminator so the network output is copied into the I/O address 0x1 for
     readback. Defined in `MLIR/stage_output.py`; this is the final dialect the
     assembler consumes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .Assembler import MAX_MATMUL_SIZE
from .MLIR.bias_removal import remove_bias_adds
from .MLIR.dialect import serialize
from .MLIR.legalize import legalize
from .MLIR.partition import partition_program
from .MLIR.stage_output import stage_output


def Process_MLIR(tmp_dir: Path | None = None) -> None:
    """Optimize the StableHLO graph, legalize it, then strip bias adds."""
    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"

    initial_path = tmp_dir / "initial.mlir"
    optimized_path = tmp_dir / "optimized.mlir"
    tpu_path = tmp_dir / "optimized.tpu.mlir"
    nobias_path = tmp_dir / "optimized.nobias.tpu.mlir"
    partitioned_path = tmp_dir / "optimized.partitioned.tpu.mlir"
    staged_path = tmp_dir / "optimized.staged.tpu.mlir"
    weight_map_path = tmp_dir / "weight_map.json"
    assert initial_path.is_file(), f"Missing {initial_path}"
    assert weight_map_path.is_file(), f"Missing {weight_map_path} (run Process_Weights first)."

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

    weight_map = json.loads(weight_map_path.read_text())
    program = legalize(optimized_path, tpu_path, weight_map)
    remove_bias_adds(program, weight_map)
    nobias_path.write_text(serialize(program))

    partition_program(program, MAX_MATMUL_SIZE)
    partitioned_path.write_text(serialize(program))

    stage_output(program)
    staged_path.write_text(serialize(program))
