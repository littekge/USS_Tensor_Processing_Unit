import torch
import torch.nn as nn

class Bigger_NN(nn.Module):
    def __init__(self):
        super(Bigger_NN, self).__init__()
        self.hidden1 = nn.Linear(1, 1000)
        self.hidden2 = nn.Linear(1000, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        return x

