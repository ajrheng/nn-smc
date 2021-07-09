import torch
import torch.nn as nn


class model(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Linear(500, 500)
        self.l2 = nn.Linear(500, 250)
        self.l3 = nn.Linear(250, 100)
        self.l4 = nn.Linear(100, 50)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):   

        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.softmax(self.l4(x))

        return x

