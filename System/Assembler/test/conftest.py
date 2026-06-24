"""Make the `nn_assembler` package importable from tests.

Prefers the installed (editable) package; if it is not installed, falls back to
importing the source tree under its public name so `python -m pytest test/` works
straight from a checkout.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import nn_assembler  # noqa: F401  (editable install present)
except ModuleNotFoundError:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import src as nn_assembler

    # Register the source package under its public name so that
    # `from nn_assembler... import ...` resolves to the same modules.
    sys.modules.setdefault("nn_assembler", nn_assembler)
