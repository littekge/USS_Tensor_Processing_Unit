from pathlib import Path
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





