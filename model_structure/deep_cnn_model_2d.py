import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN2D, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # Output: 32 x 11 x 150

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # Output: 64 x 5 x 75

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # Output: 128 x 2 x 37

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))  # Output: 256 x 1 x 18

        # Layer 5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), padding=2)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2))  # Output: 512 x 1 x 9

        # Adaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output: 512 x 4 x 4

        # Fully Connected Layer
        self.fc = nn.Linear(512 * 4 * 4, num_classes)

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

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool5(x)

        x = self.adaptive_pool(x)  # (batch_size, 512, 4, 4)

        x = x.view(x.size(0), -1)  # Flatten (batch_size, 512*4*4)
        x = self.fc(x)

        return x
