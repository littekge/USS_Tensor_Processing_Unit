import torch.nn as nn

class Tiny_NN(nn.Module):
    def __init__(self):
        super(Tiny_NN, self).__init__()
        self.hidden = nn.Linear(1, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x

