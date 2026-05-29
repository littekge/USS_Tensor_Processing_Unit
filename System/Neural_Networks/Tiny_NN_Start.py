import Operations

# define other imports
from Tiny_NN.Tiny_NN import Tiny_NN
import torch
from torch.utils.data import TensorDataset

if __name__ == "__main__":
    # model parameters
    model_params = Operations.MODEL_PARAMS(
        MODEL=Tiny_NN(),
        MODEL_NAME="Tiny_NN",
        RUN_NAME="Recent",
        TRAIN = True,
        RUN = True,
        EXPORT = True
    )

    # training parameters
    training_params = Operations.TRAINING_PARAMS(
        SUBSET_SIZE=100,
        NUM_EPOCHS=2, 
        BATCH_SIZE=4, 
        LOSS_FUNCTION="MSE",
        CLASSIFICATION_MODE="regression"
    )

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

    # create sample input
    exportSet = (torch.tensor([[0.5]], dtype=torch.float32),)
    # ---------- END DEFINED PER MODEL ----------

    Operations.Start(
        model_params=model_params,
        training_params=training_params,
        testSet=testSet,
        trainSet=trainSet,
        exportSet=exportSet
    )