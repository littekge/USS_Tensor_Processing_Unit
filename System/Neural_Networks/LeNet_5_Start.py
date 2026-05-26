from LeNet_5.LeNet_5 import LeNet_5
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import Operations

if __name__ == "__main__":
    # Parameters
    NN_NAME = "LeNet_5"
    SAVE_FILE_NAME = "LeNet_5.pth" 
    TRAIN = False
    params = Operations.TRAINING_PARAMS( # training params
        SUBSET_SIZE=1000, # size of the training subset for each epoch
        NUM_EPOCHS=2, # number of epochs to train on
        BATCH_SIZE=4, # number of images processed before calculating loss and updating weights 
    )
    RUN = True

    # determining path variables
    CURRENT_DIR = Path(__file__).parent # determining file path
    DATA_PATH = CURRENT_DIR / NN_NAME / "dataset.e" # setting path
    SAVE_PATH = CURRENT_DIR / NN_NAME / SAVE_FILE_NAME

    # ---------- DEFINED PER MODEL ----------
    # converts images to normalized tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3018,))
    ])
    trainSet = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform) # defining training dataset
    testSet = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform) # defining test dataset
    net = LeNet_5()
    # ---------- END DEFINED PER MODEL ----------

    if TRAIN: Operations.Train(model=net, dataset=trainSet, save_file_path=SAVE_PATH, params=params)
    if RUN: Operations.Run(model=net, dataset=testSet, load_file_path=SAVE_PATH, BATCH_SIZE=params.BATCH_SIZE)
        

    