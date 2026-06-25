"""Neural Network Assembler.

Lowers a StableHLO computation graph plus quantized weights into a complete
*Functional TPU* message-protocol transmission.

Public API::

    from nn_assembler import Convert
    Convert("Tiny_NN", "Recent")   # writes out/TRANSMISSION.bin
"""

from .Convert import Convert, NN_import, main

__all__ = ["Convert", "NN_import", "main"]
__version__ = "0.1.0"
