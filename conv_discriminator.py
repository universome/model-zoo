import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDiscriminator(nn.Module):
    "Simple convolutional discriminator"
    def __init__(self, in_channels:int):
        super(ConvDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3), nn.BatchNorm2d(4))
        self.conv2 = nn.Sequential(nn.Conv2d(4, 16, 5), nn.BatchNorm2d(16))
        self.conv3 = nn.Sequential(nn.Conv2d(16, 64, 5, stride=2), nn.BatchNorm2d(64))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2), nn.BatchNorm2d(128))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2), nn.BatchNorm2d(256))
        self.dense = nn.Sequential(nn.Linear(1024, 1))

    def forward(self, x):
        out = x

        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))

        out = out.view(x.size(0), -1)
        out = self.dense(out)

        return out
