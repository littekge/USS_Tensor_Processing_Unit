#!/usr/bin/env bash
# One-command environment bootstrap: creates the Python venv, installs pinned
# dependencies, and installs the nn_assembler package in editable mode.
# Safe to re-run; existing installs are left untouched.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/USS_TPU.venv"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment at $VENV"
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install -r "$REPO_ROOT/requirements.txt"
# editable_mode=compat writes a static path so IDEs/Pylance resolve the package
pip install -e "$REPO_ROOT/System/Assembler" --config-settings editable_mode=compat

echo
echo "=== Environment checks ==="
python -c "import tkinter" 2>/dev/null \
  && echo "  [ok]      tkinter available (Demo GUI)" \
  || echo "  [warning] tkinter missing — Demo GUI needs it: sudo apt install python3-tk"
command -v stablehlo-opt >/dev/null \
  && echo "  [ok]      stablehlo-opt on PATH (Assembler stage)" \
  || echo "  [warning] stablehlo-opt not on PATH — required by the Assembler; not
            needed for the pre-built /Benchmarks transmissions. Prebuilt binary
            (Linux x86-64, glibc >= 2.38):
            https://github.com/littekge/USS_Tensor_Processing_Unit/releases/tag/StableHLO_Binaries
            or build from https://github.com/openxla/stablehlo"

echo
echo "Setup complete. Activate the environment with:"
echo "  source ${VENV#"$REPO_ROOT/"}/bin/activate"
