import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    # defines the structure of the neural network
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels,
        # 3x3 square convolution kernel.
        self.conv1 = nn.Conv2d(1, 6, 3) # convolutional layer 1
        self.conv2 = nn.Conv2d(6, 16, 3) # convolutional layer 2
        # 16*6*6 input featuers, 120 output features.
        self.fc1 = nn.Linear(16*5*5, 120) # linear layer 1
        self.fc2 = nn.Linear(120, 84) # linear layer 2
        self.fc3 = nn.Linear(84, 10) # linear layer 3

    # defines forward propagation of network
    # x is the input to the network
    def forward(self, x):
        # Passes x through the first convolutional layer, applies relu
        # activation function, and then samples x.
        # Max pooling over a (2, 2) window.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # Second convolutional layer
        x = x.view(-1, self.num_flat_features(x)) # Resizing x
        x = F.relu(self.fc1(x)) # 1st linear layer
        x = F.relu(self.fc2(x)) # 2nd linear layer
        x = self.fc3(x) # 3rd linear layer
        return x
    
    # determines the number of flat features in x
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features