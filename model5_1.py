import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
        )  # 26, 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3),
            nn.ReLU(),
        )  # 24, 5
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU(),
        )  # 22, 7
        self.pool1 = nn.MaxPool2d(2, 2)  # 11, 8
        self.ant1 = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
        )  # 11, 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 10, 3),
            nn.ReLU(),
        )  # 9 , 12
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 14, 3),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 16, 3),
            nn.ReLU(),
        )  # 5 , 32
        self.avgpool = nn.AvgPool2d(5)
        self.conv7 = nn.Conv2d(16, 10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.ant1(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.conv6(x)
        x = self.avgpool(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
