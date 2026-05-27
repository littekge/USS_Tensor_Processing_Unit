import torch
from pathlib import Path
import torchax.export as exp

def convert(MODEL_NAME, MODEL_RUN):
    TOP_DIR = Path(__file__).parent.parent.parent # determining path of top level directory
    LOAD_PATH = TOP_DIR / "Neural_Networks" / MODEL_NAME / str(MODEL_NAME + "_" + MODEL_RUN + ".pt2") # determing exact load path
    assert(LOAD_PATH.is_file()) # checking that the load file exists
    model = torch.export.load(LOAD_PATH)
    

if __name__ == "__main__":
    MODEL_NAME = "Tiny_NN"
    MODEL_RUN = "Recent"
    convert(MODEL_NAME, MODEL_RUN)




