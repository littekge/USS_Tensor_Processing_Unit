"""Top-level entry point for the Neural Network Assembler.

Binds the four lowering stages together:
  1. Neural Network Import   (NN_import)
  2. Weight Processing       (Process_Weights)
  3. MLIR Lowering           (Process_MLIR) + Assembler (Assemble)
  4. Final Conversion        (Serialize)
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    # Allow running this file directly (`python ./src/Convert.py ...`) even though
    # the package now uses relative imports: put the parent of `src/` on sys.path,
    # import the package's entry point, and hand off to it.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.Convert import main as _main

    _main()
    raise SystemExit(0)

import argparse
import shutil
from pathlib import Path

from .Assembler import Assemble
from .Process_MLIR import Process_MLIR
from .Process_Weights import Process_Weights
from .Serializer import Serialize


def NN_import(model_name: str, run_name: str, tmp_dir: Path | None = None) -> None:
    """Copy a trained network's MLIR graph and weights into the tmp workspace."""
    nn_dir = Path(__file__).parent.parent.parent.parent / "Neural_Networks"
    mlir_path = nn_dir / model_name / f"{model_name}_{run_name}.mlir"
    weight_path = nn_dir / model_name / f"{model_name}_{run_name}.weights.npz"
    assert mlir_path.is_file(), f"Missing network graph: {mlir_path}"
    assert weight_path.is_file(), f"Missing network weights: {weight_path}"

    if tmp_dir is None:
        tmp_dir = Path(__file__).parent.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(str(mlir_path), str(tmp_dir / "initial.mlir"))
    shutil.copy(str(weight_path), str(tmp_dir / "weights.npz"))


def Convert(model_name: str, run_name: str) -> Path:
    """Run the full pipeline for one network, returning the transmission path."""
    NN_import(model_name, run_name)
    Process_Weights()
    Process_MLIR()
    Assemble()
    return Serialize()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: assemble a network named on the command line."""
    parser = argparse.ArgumentParser(description="Assemble a neural network into a Functional TPU transmission.")
    parser.add_argument("model_name")
    parser.add_argument("run_name")
    args = parser.parse_args(argv)

    output_path = Convert(args.model_name, args.run_name)
    print(f"Wrote transmission to {output_path}")


if __name__ == "__main__":
    main()
