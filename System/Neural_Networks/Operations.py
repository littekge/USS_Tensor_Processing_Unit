import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler as sample
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MODEL_PARAMS:
    MODEL: torch.nn.Module
    MODEL_NAME: str
    RUN_NAME: str
    TRAIN: bool = False
    RUN: bool = True
    EXPORT: bool = False

@dataclass
class TRAINING_PARAMS:
    SUBSET_SIZE: int # size of the training subset for each epoch
    NUM_EPOCHS: int = 10 # number of epochs to train on
    BATCH_SIZE: int = 4 # number of images processed before calculating loss and updating weights
    LEARNING_RATE: float = 0.001 # learning rate
    MOMENTUM: float = 0.9 # momentum
    LOSS_FUNCTION: str = "MSE" # which loss function to use in training
    CLASSIFICATION_MODE: str = "regression" # type of correctness classification to use in testing

def Start(model_params, training_params, trainSet=None, testSet=None, exportSet=None):
    # assertions
    assert isinstance(model_params, MODEL_PARAMS) # asserts that model_params is correct type
    if model_params.TRAIN: assert trainSet != None # assert that trainSet exists if training
    if model_params.RUN: assert testSet != None # assert that testSet exisits if running
    if model_params.EXPORT: assert exportSet != None # assert that exportSet exisits if exporting

    # determining path variables
    CURRENT_DIR = Path(__file__).parent # determining file path
    SAVE_PATH = CURRENT_DIR / model_params.MODEL_NAME / str(model_params.MODEL_NAME + "_" + model_params.RUN_NAME + ".pth") # setting save path
    LOAD_PATH = SAVE_PATH # setting load path
    EXPORT_PATH = CURRENT_DIR / model_params.MODEL_NAME / str(model_params.MODEL_NAME + "_" + model_params.RUN_NAME + ".pt2") # setting export path

    if model_params.TRAIN: Train(model=model_params.MODEL, dataset=trainSet, save_file_path=SAVE_PATH, params=training_params)
    if model_params.EXPORT: Save(model=model_params.MODEL, sample_input=exportSet, save_file_path=EXPORT_PATH)
    if model_params.RUN: Run(model=model_params.MODEL, dataset=testSet, load_file_path=LOAD_PATH, params=training_params)
    
    

def Train(model, dataset, save_file_path, params):
    # assertions
    assert isinstance(params, TRAINING_PARAMS) # assert that params is correct type
    assert 0 < params.SUBSET_SIZE <= len(dataset) # subset size must be less than or equal to dataset size
    assert 0 < params.BATCH_SIZE <= params.SUBSET_SIZE # batch size must be less than or equal to subset size
    # constants def
    PRINT_INTERVAL = max(1, int(params.SUBSET_SIZE/params.BATCH_SIZE/10)) # how often training data should be printed

    # different loss functions
    match params.LOSS_FUNCTION:
        case "MSE": criterion = nn.MSELoss()
        case "CrossEntropy": criterion = nn.CrossEntropyLoss() 
        case _: 
            print("invalid loss_type, defaulting to MSELoss...")
            criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), params.LEARNING_RATE, params.MOMENTUM) # learning driver to adjust weights

    for epoch in range(params.NUM_EPOCHS):
        running_loss = 0.0 # loss accumulated during epoch
        indices = torch.randperm(len(dataset))[:params.SUBSET_SIZE].tolist() # generate a 
        sampler = sample(indices)
        trainLoader = torch.utils.data.DataLoader(dataset, batch_size=params.BATCH_SIZE, sampler=sampler, num_workers=2)

        for i, data in enumerate(trainLoader, 0): # loop through training data
            inputs, labels = data # get input data and labels
            optimizer.zero_grad() # zeroing gradient in between iterations
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # backwards propogation to calculate gradients
            optimizer.step() # adjust weights

            # print statistics
            running_loss += loss.item()
            if i % PRINT_INTERVAL == PRINT_INTERVAL-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / PRINT_INTERVAL))
                running_loss = 0.0

    print('Finished Training')

    # saving data
    currentDir = Path(__file__).parent # determining file path
    path = currentDir / save_file_path # determining save path
    torch.save(model.state_dict(), path)
    print("Model saved successfully!")
    


# runs the neural network with relevant parameters
def Run(model, dataset, load_file_path, params):
    # assertions
    assert isinstance(params, TRAINING_PARAMS) # assert that params is correct type
    assert load_file_path.is_file() # assert that the weights file exists
    assert 0 < params.BATCH_SIZE < len(dataset) # Batch size cannot be larger than dataset

    # constants def
    PRINT_INTERVAL = max(1, int(len(dataset) / params.BATCH_SIZE / 10)) # determining print interval

    model.load_state_dict(torch.load(load_file_path, weights_only=True)) # loading model
    testLoader = torch.utils.data.DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=2) # loading test dataset
    # calculating and printing the accuracy of the model
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(testLoader, 0):
            inputs, labels = data
            outputs = model(inputs)

            match params.CLASSIFICATION_MODE:
                case "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                case "regression":
                    tolerance = 0.1
                    correct += (torch.abs(outputs - labels) < tolerance).sum().item()
                    total += labels.numel()

                case _:
                    print("invalid classification mode, defaulting to regression...")
                    tolerance = 0.1
                    correct += (torch.abs(outputs - labels) < tolerance).sum().item()
                    total += labels.numel()

            if i % PRINT_INTERVAL == PRINT_INTERVAL - 1:
                print("[%d] correct: %d total: %d" % (i + 1, correct, total))

    print('\nAccuracy of the network: %d %%' % (
    100 * correct / total))
   
def Save(model, sample_input, save_file_path):
    exportedModel = torch.export.export(model, sample_input) # converted to ExportedProgram
    torch.export.save(exportedModel, save_file_path) # save as pt2 archive
    print("Model exported to .pt2 successfully!")
    

    