
# Neural network imports
from Tiny_NN.Tiny_NN import Tiny_NN
from LeNet_5.LeNet_5 import LeNet_5
from Bigger_NN.Bigger_NN import Bigger_NN
# define other imports
import Operations
import torch
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import argparse

def Get_LeNet_5_Data():
    CURRENT_DIR = Path(__file__).parent # determining file path
    DATA_PATH = CURRENT_DIR / "LeNet_5" / "dataset.e" # setting dataset path
    # converts images to normalized tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3018,))
    ])
    trainSet = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform) # defining training dataset
    testSet = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform) # defining test dataset
    exportSet = (torch.randn(1, 1, 28, 28),)
    return trainSet, testSet, exportSet


def Get_Tiny_NN_Data():
    DATASET_SIZE = 1000 # define dataset size
    # create basic training data
    train_x = torch.randn(DATASET_SIZE, 1)
    train_y = train_x.clone() - 2
    trainSet = TensorDataset(train_x, train_y)

    # create basic test data
    test_x = torch.randn(DATASET_SIZE, 1)
    test_y = test_x.clone() - 2
    testSet = TensorDataset(test_x, test_y)

    # create sample input
    exportSet = (torch.tensor([[0.5]], dtype=torch.float32),)
    return trainSet, testSet, exportSet
   
def Get_Bigger_NN_Data():
    DATASET_SIZE = 1000 # define dataset size
    # create basic training data
    train_x = torch.randn(DATASET_SIZE, 1)
    train_y = train_x.clone() - 2
    trainSet = TensorDataset(train_x, train_y)

    # create basic test data
    test_x = torch.randn(DATASET_SIZE, 1)
    test_y = test_x.clone() - 2
    testSet = TensorDataset(test_x, test_y)

    # create sample input
    exportSet = (torch.tensor([[0.5]], dtype=torch.float32),)
    return trainSet, testSet, exportSet

def Start(model_name, run_name, train, export, run):
    model_params = Operations.MODEL_PARAMS(
        MODEL_NAME=model_name,
        RUN_NAME=run_name,
        TRAIN = train,
        RUN = run,
        EXPORT = export
    )

    match model_name:
        case "LeNet_5":
            model_params.MODEL = LeNet_5()
            trainSet, testSet, exportSet = Get_LeNet_5_Data()
            training_params = Operations.TRAINING_PARAMS(
                SUBSET_SIZE=5000, 
                NUM_EPOCHS=5, 
                BATCH_SIZE=4, 
                LOSS_FUNCTION="CrossEntropy", 
                CLASSIFICATION_MODE="classification" 
            )
        case "Tiny_NN":
            model_params.MODEL = Tiny_NN()
            trainSet, testSet, exportSet = Get_Tiny_NN_Data()
            training_params = Operations.TRAINING_PARAMS(
                SUBSET_SIZE=1000,
                NUM_EPOCHS=5, 
                BATCH_SIZE=4, 
                LOSS_FUNCTION="MSE",
                CLASSIFICATION_MODE="regression"
            )
        case "Bigger_NN":
            model_params.MODEL = Bigger_NN()
            trainSet, testSet, exportSet = Get_Bigger_NN_Data()
            training_params = Operations.TRAINING_PARAMS(
                SUBSET_SIZE=1000,
                NUM_EPOCHS=5, 
                BATCH_SIZE=4, 
                LOSS_FUNCTION="MSE",
                CLASSIFICATION_MODE="regression"
            )
        case _:
            print("Invalid model name, exiting...")
            quit()

    Operations.Start(
        model_params=model_params,
        training_params=training_params,
        testSet=testSet,
        trainSet=trainSet,
        exportSet=exportSet
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("run_name")
    parser.add_argument("-t", "--train", action='store_true')
    parser.add_argument("-e", "--export", action='store_true')
    parser.add_argument("-r", "--run", action='store_true')
    
    args = parser.parse_args()

    Start(
        model_name=args.model_name,
        run_name=args.run_name,
        train=args.train,
        export=args.export,
        run=args.run
    )

    
    
    




