import Operations

# define other imports
from LeNet_5.LeNet_5 import LeNet_5
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

if __name__ == "__main__":
    # model parameters
    model_params = Operations.MODEL_PARAMS(
        MODEL=LeNet_5(),
        MODEL_NAME="LeNet_5",
        RUN_NAME="Recent",
        TRAIN = True,
        RUN = True,
        EXPORT = True
    )

    # training parameters
    training_params = Operations.TRAINING_PARAMS(
        SUBSET_SIZE=1000, 
        NUM_EPOCHS=2, 
        BATCH_SIZE=4, 
        LOSS_FUNCTION="CrossEntropy", 
        CLASSIFICATION_MODE="classification" 
    )

    # ---------- DEFINED PER MODEL ----------
    CURRENT_DIR = Path(__file__).parent # determining file path
    DATA_PATH = CURRENT_DIR / model_params.MODEL_NAME / "dataset.e" # setting dataset path
    # converts images to normalized tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3018,))
    ])
    trainSet = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform) # defining training dataset
    testSet = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform) # defining test dataset
    exportSet = (torch.randn(1, 1, 28, 28),)
    # ---------- END DEFINED PER MODEL ----------
    
    Operations.Start(
        model_params=model_params,
        training_params=training_params,
        testSet=testSet,
        trainSet=trainSet,
        exportSet=exportSet
    )

    