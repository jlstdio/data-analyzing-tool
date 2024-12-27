import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 원하는 크기로 조정
        self.fc = nn.Linear(128 * 4 * 4, num_classes)  # Adaptive Pooling 크기에 맞게 수정

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.adaptive_pool(x)  # (batch_size, 128, 4, 4)

        x = x.view(x.size(0), -1)  # Flatten (batch_size, 128*4*4)
        x = self.fc(x)

        return x
