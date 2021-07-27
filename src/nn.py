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


class neural_network(nn.Module):
    def __init__(
        self,
        units,
        nonlinearity = nn.ReLU
    ):
        super().__init__()
        assert len(units) > 1
        self.in_dim = units[0]
        self.out_dim = units[-1]

        module_list = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            module_list.append(nn.Linear(in_dim, out_dim))
            module_list.append(nonlinearity())
        module_list = module_list[:-1]
        module_list.append(nn.Softmax(dim=-1))
        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)
