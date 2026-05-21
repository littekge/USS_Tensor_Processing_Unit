from LeNet_5 import LeNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler as sample
from torch.fx import symbolic_trace

if __name__ == "__main__":
    # adjustable parameters
    SUBSET_SIZE = 1000 # size of the training subset for each epoch
    NUM_EPOCHS = 10 # number of epochs to train on
    BATCH_SIZE = 10 # number of images processed before calculating loss and updating weights
    PRINT_INTERVAL = SUBSET_SIZE / BATCH_SIZE / 10 # number of batches per print statment
    # converts images to tensors normalized around x = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3018,))
    ])

    trainSet = torchvision.datasets.MNIST(root="./LeNet_5/datasets/", train=True, download=True, transform=transform) # defining training dataset
    dataset_size = len(trainSet)
   
    # trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4, shuffle=True, num_workers=2) # loading training dataset

    testSet = torchvision.datasets.MNIST(root="./LeNet_5/datasets/", train=False, download=True, transform=transform) # defining test dataset
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) # loading test dataset

    net = LeNet() # initializing neural network
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # learning driver to adjust weights

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0 # loss accumulated during epoch
        indices = torch.randperm(dataset_size)[:SUBSET_SIZE].tolist() # generate a 
        sampler = sample(indices)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

        for i, data in enumerate(trainLoader, 0): # loop through training data
            inputs, labels = data # get input data and labels
            optimizer.zero_grad() # zeroing gradient in between iterations
            outputs = net(inputs) # forward pass
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
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    
    net.eval()
    gm = symbolic_trace(net)
    print("\n\n==== Graph ====")
    print(gm.graph)

