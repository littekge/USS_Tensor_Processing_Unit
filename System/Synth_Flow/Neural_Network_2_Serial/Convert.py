'''
from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import stablehlo
'''
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
    OPTIM_PATH = CURRENT_DIR / "current.optim.mlir"
    try:
        subprocess.run(["stablehlo-opt", "--stablehlo-target-independent-optimization", str(MLIR_PATH), "-o", str(OPTIM_PATH)])
    except subprocess.CalledProcessError as e:
        print("Failed: stablehlo-opt either does not exist or is not registered in PATH")


    '''
    with ir.Context() as ctx, ir.Location.unknown():
        stablehlo.register_dialect(ctx)
        stablehlo.register_stablehlo_passes()
        
        optim_text = OPTIM_PATH.read_text(encoding="utf-8")
        module = ir.Module.parse(optim_text)
        pm = PassManager.parse(
        "builtin.module(func.func("
        "canonicalize"
        "))"
        )
        
        pm.run(module.operation)
        CANON_PATH = CURRENT_DIR / "current.canon.mlir"
        with open(CANON_PATH, "w") as out:
            out.write(str(module))

        canon_text = CANON_PATH.read_text(encoding="utf-8")
        module = ir.Module.parse(canon_text)
        pm = PassManager.parse(
        "builtin.module(func.func("
        "cse"
        "))"
        )
        pm.run(module.operation)
        CANON_PATH = CURRENT_DIR / "current.cse.mlir"
        with open(CANON_PATH, "w") as out:
            out.write(str(module))
    '''