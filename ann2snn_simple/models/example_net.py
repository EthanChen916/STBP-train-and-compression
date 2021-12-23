import torch.nn as nn
import torch.nn.functional as F
import torch

"""
ExampleNet0: 
ExampleNet0bias: 
"""


class ExampleNet0(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.avgpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.avgpool2(out)
        out = self.relu3(self.conv3(out))
        out = out.view(-1, 4*4*32)
        out = self.fc1(out)
        return out


class ExampleNet0bias(ExampleNet0):
    def __init__(self, n_extra_layers=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
