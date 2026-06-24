import subprocess
from pathlib import Path

def Process_MLIR():
    
    # StableHLO base optimization pass
    TMP_DIR = Path(__file__).parent.parent / "tmp"
    OPTIM_PATH = TMP_DIR / "optimized.mlir"
    MLIR_PATH = TMP_DIR / "initial.mlir"
    try:
        subprocess.run(["stablehlo-opt", "--stablehlo-target-independent-optimization", str(MLIR_PATH), "-o", str(OPTIM_PATH)])
    except subprocess.CalledProcessError as e:
        print("Failed: stablehlo-opt either does not exist or is not registered in PATH")
