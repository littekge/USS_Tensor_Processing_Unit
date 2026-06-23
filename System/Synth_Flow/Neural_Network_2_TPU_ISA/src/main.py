import argparse
from pathlib import Path
import subprocess
from NN_Import import NN_import

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("run_name")

    args = parser.parse_args()
    
    NN_import(args.model_name, args.run_name) # import NN to ../tmp/
    
    # StableHLO base optimization pass
    TMP_DIR = Path(__file__).parent.parent / "tmp"
    OPTIM_PATH = TMP_DIR / "optimized.mlir"
    MLIR_PATH = TMP_DIR / "initial.mlir"
    try:
        subprocess.run(["stablehlo-opt", "--stablehlo-target-independent-optimization", str(MLIR_PATH), "-o", str(OPTIM_PATH)])
    except subprocess.CalledProcessError as e:
        print("Failed: stablehlo-opt either does not exist or is not registered in PATH")