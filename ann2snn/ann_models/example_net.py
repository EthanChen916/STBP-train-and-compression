import torch.nn as nn
import torch.nn.functional as F
import torch


class ExampleNet(nn.Module):
    def __init__(self):
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
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 4*4*32)
        out = self.fc1(x)
        return out


class MNISTNet(nn.Module):  # Example net for MNIST
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1, 2, bias=None)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(15, 40, 5, 1, 2, bias=None)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 40, 300, bias=None)
        self.fc2 = nn.Linear(300, 10, bias=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 40)
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class Cifar10Net(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(Cifar10Net, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)  # 64 * 32 * 32
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)  # 64 * 32* 32
        self.avgpool1 = nn.AvgPool2d(2)  # pooling 16 * 16 * 16

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)  # 128 * 16 * 16
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)  # 128 * 16 * 16
        self.avgpool2 = nn.AvgPool2d(2)  # pooling 128 * 8 * 8

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, bias=False)  # 256 * 8 * 8
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)  # 256 * 8 * 8
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)  # 256 * 8 * 8
        self.avgpool3 = nn.AvgPool2d(2)  # pooling 256 * 4 * 4

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)  # 512 * 4 * 4
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)  # 512 * 4 * 4
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)  # 512 * 4 * 4
        self.avgpool4 = nn.AvgPool2d(2)  # pooling 100 * 2 * 2

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1, bias=False)  # 512 * 2 * 2
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)  # 512 * 2 * 2
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)  # 512 * 2 * 2
        self.avgpool5 = nn.AvgPool2d(2)  # pooling 512 * 1 * 1

        self.fc1 = nn.Linear(512 * 1 * 1, 256, bias=False)
        self.fc2 = nn.Linear(256, 256, bias=False)
        self.fc3 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.avgpool1(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.avgpool2(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = F.relu(x)
        x = self.avgpool3(x)

        x = self.conv4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = F.relu(x)
        x = self.avgpool4(x)

        x = self.conv5_1(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x)
        x = self.avgpool5(x)

        x = x.view(-1, 512 * 1 * 1)
        nn.Dropout(0.5)
        x = self.fc1(x)
        x = F.relu(x)
        nn.Dropout(0.5)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        return out
