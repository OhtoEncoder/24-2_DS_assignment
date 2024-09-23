import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  #입력크기는 내가 알아서...
        self.fc2 = nn.Linear(128, 3)  #Buy Hold Sell 세개의 클래스

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  #flat
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x