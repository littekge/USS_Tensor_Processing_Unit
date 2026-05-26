from Tiny_NN.Tiny_NN import Tiny_NN
import torch
from pathlib import Path
import Operations
from torch.utils.data import TensorDataset

if __name__ == "__main__":
    # User options
    NN_NAME = "Tiny_NN"
    SAVE_FILE_NAME = "Tiny_NN_Recent.pth"
    TRAIN = True
    RUN = True

    # training parameters
    params = Operations.TRAINING_PARAMS(
        SUBSET_SIZE=100, # size of the training subset for each epoch
        NUM_EPOCHS=2, # number of epochs to train on
        BATCH_SIZE=4, # number of images processed before calculating loss and updating weights 
        LOSS_FUNCTION="MSE", # loss function for numbers
        CLASSIFICATION_MODE="regression"
    )

    # setting default save file name if unspecified
    if SAVE_FILE_NAME == None:
        SAVE_FILE_NAME = (
            NN_NAME + "_" + 
            str(params.SUBSET_SIZE) + "_" + 
            str(params.NUM_EPOCHS) + "_" + 
            str(params.BATCH_SIZE) + "_" + 
            str(params.LEARNING_RATE) + "_" + 
            str(params.MOMENTUM) + "_" +
            str(params.LOSS_FUNCTION) + ".pth" 
        )

    # determining path variables
    CURRENT_DIR = Path(__file__).parent # determining file path
    DATA_PATH = CURRENT_DIR / NN_NAME / "dataset.e" # setting path
    SAVE_PATH = CURRENT_DIR / NN_NAME / SAVE_FILE_NAME

    # ---------- DEFINED PER MODEL ----------
    DATASET_SIZE = 1000 # define dataset size
    # create basic training data
    train_x = torch.randn(DATASET_SIZE, 1)
    train_y = train_x.clone()
    trainSet = TensorDataset(train_x, train_y)

    # create basic test data
    test_x = torch.randn(DATASET_SIZE, 1)
    test_y = test_x.clone()
    testSet = TensorDataset(test_x, test_y)
    net = Tiny_NN()
    # ---------- END DEFINED PER MODEL ----------

    if TRAIN: Operations.Train(model=net, dataset=trainSet, save_file_path=SAVE_PATH, params=params)
    if RUN: Operations.Run(model=net, dataset=testSet, load_file_path=SAVE_PATH, params=params)

