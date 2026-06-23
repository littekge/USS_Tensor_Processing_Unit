import argparse
from pathlib import Path
import subprocess
import shutil


def NN_import(model_name, run_name):
    NN_DIR = Path(__file__).parent.parent.parent.parent / "Neural_Networks"
    MLIR_PATH = NN_DIR / model_name / str(model_name + "_" + run_name + ".mlir")
    WEIGHT_PATH = NN_DIR / model_name / str(model_name + "_" + run_name + ".weights.npz")
    assert MLIR_PATH.is_file()
    assert WEIGHT_PATH.is_file()

    CURRENT_DIR = Path(__file__).parent
    TMP_DIR = CURRENT_DIR.parent / "tmp"
    TMP_MLIR_PATH = TMP_DIR / "initial.mlir"
    TMP_WEIGHT_PATH = TMP_DIR / "weights.npz"
    
    shutil.copy(str(MLIR_PATH), str(TMP_MLIR_PATH))
    shutil.copy(str(WEIGHT_PATH), str(TMP_WEIGHT_PATH))

def Convert(model_name, run_name):
    NN_import(model_name, run_name) # import NN to ../tmp/

    # StableHLO base optimization pass
    TMP_DIR = Path(__file__).parent.parent / "tmp"
    OPTIM_PATH = TMP_DIR / "optimized.mlir"
    MLIR_PATH = TMP_DIR / "initial.mlir"
    try:
        subprocess.run(["stablehlo-opt", "--stablehlo-target-independent-optimization", str(MLIR_PATH), "-o", str(OPTIM_PATH)])
    except subprocess.CalledProcessError as e:
        print("Failed: stablehlo-opt either does not exist or is not registered in PATH")

# main guard
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("run_name")

    args = parser.parse_args()

    Convert(args.model_name, args.run_name)
    
    
    
    