import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())  # 28, 3
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())  # 28, 5
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())  # 28, 7
        self.pool1 = nn.MaxPool2d(2, 2)  # 14, 8
        self.ant1 = nn.Sequential(nn.Conv2d(128, 32, 1), nn.ReLU())  # 14, 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )  # 14 , 12
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )  # 14, 16
        self.pool2 = nn.MaxPool2d(2, 2)  # 7, 18
        self.ant2 = nn.Sequential(nn.Conv2d(128, 32, 1), nn.ReLU())  # 7, 18

        self.conv6 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU())  # 5 , 32
        self.avgpool = nn.AvgPool2d(5)
        self.conv7 = nn.Sequential(nn.Conv2d(128, 10, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.ant1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.ant2(x)
        x = self.conv6(x)
        x = self.avgpool(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
