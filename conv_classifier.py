import torch
import torch.nn as nn


class ConvClassifier(nn.Module):
    "Simple convolutional classifier"
    def __init__(self, in_channels:int):
        super(ConvClassifier, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 8, 3), nn.BatchNorm2d(8), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 32, 5), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 128, 5), nn.BatchNorm2d(128), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.ffn = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out = x

        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)

        out = out.view(x.size(0), -1)
        out = self.ffn(out)

        return out
