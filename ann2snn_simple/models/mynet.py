import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 8, 3, padding=1)  # 8 * 32 * 32
        self.bn1_1 = nn.BatchNorm2d(8)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(8, 8, 3, padding=1)  # 8 * 32* 32
        self.bn1_2 = nn.BatchNorm2d(8)
        self.relu1_2 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2)  # pooling 16 * 16 * 16

        self.conv2_1 = nn.Conv2d(8, 16, 3, padding=1)  # 32 * 16 * 16
        self.bn2_1 = nn.BatchNorm2d(16)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=1)  # 32 * 16 * 16
        self.bn2_2 = nn.BatchNorm2d(16)
        self.relu2_2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2)  # pooling 128 * 8 * 8

        self.conv3_1 = nn.Conv2d(16, 28, 3, padding=1)  # 64 * 8 * 8
        self.bn3_1 = nn.BatchNorm2d(28)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(28, 28, 3, padding=1)  # 64 * 8 * 8
        self.bn3_2 = nn.BatchNorm2d(28)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(28, 28, 3, padding=1)  # 64 * 8 * 8
        self.bn3_3 = nn.BatchNorm2d(28)
        self.relu3_3 = nn.ReLU()
        self.avgpool3 = nn.AvgPool2d(2)  # pooling 256 * 4 * 4

        # self.conv4_1 = nn.Conv2d(16, 28, 3, padding=1, bias=False)  # 100 * 4 * 4
        # self.conv4_2 = nn.Conv2d(28, 28, 3, padding=1, bias=False)  # 100 * 4 * 4
        # self.conv4_3 = nn.Conv2d(28, 28, 3, padding=1, bias=False)  # 100 * 4 * 4
        # self.avgpool4 = nn.AvgPool2d(2)  # pooling 100 * 2 * 2

        # self.conv5_1 = nn.Conv2d(28, 28, 3, padding=1, bias=False)  # 100 * 2 * 2
        # self.conv5_2 = nn.Conv2d(28, 28, 3, padding=1, bias=False)  # 100 * 2 * 2
        # self.conv5_3 = nn.Conv2d(28, 28, 3, padding=1, bias=False)  # 100 * 2 * 2
        # self.avgpool5 = nn.AvgPool2d(2)  # pooling 100 * 1 * 1

        # view

        self.fc1 = nn.Linear(28 * 4 * 4, 28 * 4 * 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(28 * 4 * 4, 28 * 4 * 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(28 * 4 * 4, 10)

    def forward(self, x):
        # x.size(0)即为batch_size
        # in_size = x.size(0)

        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.avgpool1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu2_1(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.relu2_2(out)
        out = self.avgpool2(out)

        out = self.conv3_1(out)
        out = self.bn3_1(out)
        out = self.relu3_1(out)
        out = self.conv3_2(out)
        out = self.bn3_2(out)
        out = self.relu3_2(out)
        out = self.conv3_3(out)
        out = self.bn3_3(out)
        out = self.relu3_3(out)
        out = self.avgpool3(out)

        # out = self.conv4_1(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.conv4_2(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.conv4_3(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.avgpool4(out)
        #
        # out = self.conv5_1(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.conv5_2(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.conv5_3(out)
        # nn.BatchNorm2d(28)
        # out = F.relu(out)
        # out = self.avgpool5(out)

        # 展平
        out = out.view(-1, 28 * 4 * 4)
        nn.Dropout(0.5)
        out = self.fc1(out)
        out = self.relu1(out)
        nn.Dropout(0.5)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        return out


