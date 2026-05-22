import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler as sample
from pathlib import Path
from dataclasses import dataclass
from torch.fx import symbolic_trace

@dataclass
class TRAINING_PARAMS:
    SUBSET_SIZE: int # size of the training subset for each epoch
    NUM_EPOCHS: int = 10 # number of epochs to train on
    BATCH_SIZE: int = 4 # number of images processed before calculating loss and updating weights
    LEARNING_RATE: float = 0.001 # learning rate
    MOMENTUM: float = 0.9 # momentum

def Train(model, dataset, save_file_path, params):
    # assertions
    assert isinstance(params, TRAINING_PARAMS) # assert that params is correct type
    assert 0 < params.SUBSET_SIZE <= len(dataset) # subset size must be less than or equal to dataset size
    assert 0 < params.BATCH_SIZE <= params.SUBSET_SIZE # batch size must be less than or equal to subset size
    # constants def
    PRINT_INTERVAL = params.SUBSET_SIZE/params.BATCH_SIZE/10 # how often training data should be printed

    criterion = nn.CrossEntropyLoss() # loss function
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
    
def Run(model, dataset, load_file_path, BATCH_SIZE=4):
    # assertions
    assert load_file_path.is_file() # assert that the weights file exists
    assert 0 < BATCH_SIZE < len(dataset) # Batch size cannot be larger than dataset

    # constants def
    PRINT_INTERVAL = len(dataset) / BATCH_SIZE / 10 # determining print interval

    model.load_state_dict(torch.load(load_file_path, weights_only=True)) # loading model
    testLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) # loading test dataset
    # calculating and printing the accuracy of the model
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testLoader, 0):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % PRINT_INTERVAL == PRINT_INTERVAL-1:
                print("[%d] correct: %d total: %d" % (i+1, correct, total))
            

    print('\nAccuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
   
    # printing graph of function calls and model parameters
    with torch.no_grad():
        model.eval()
        gm = symbolic_trace(model)
        print("\n==== Graph ====")
        gm.graph.print_tabular()

        for layer in model.modules():
            print(layer)

        for name, param in model.named_parameters():
            print(name, param)
    