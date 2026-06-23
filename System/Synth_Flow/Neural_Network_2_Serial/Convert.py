
import argparse
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("run_name")

    args = parser.parse_args()
    
    NN_DIR = Path(__file__).parent.parent.parent / "Neural_Networks" # getting Neural_Networks directory
    MLIR_PATH = NN_DIR / args.model_name / str(args.model_name + "_" + args.run_name + ".mlir") # setting save path
    assert MLIR_PATH.is_file()

    CURRENT_DIR = Path(__file__).parent
    OPTIM_PATH = CURRENT_DIR / "tmp" / "current.optim.mlir"
    try:
        subprocess.run(["stablehlo-opt", "--stablehlo-target-independent-optimization", str(MLIR_PATH), "-o", str(OPTIM_PATH)])
    except subprocess.CalledProcessError as e:
        print("Failed: stablehlo-opt either does not exist or is not registered in PATH")