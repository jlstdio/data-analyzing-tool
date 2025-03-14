import numpy as np
import torch
import torch.nn as nn


class testNN_wo_Softmax_3_layer(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_3_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 8 * 8, outputClasses)  # 64는 채널 수, 4*4는 마지막 풀링 레이어의 출력 크기
        # self.softmax = nn.Softmax(dim=1)  # use only with BCE

        self._initialize_weights()  # He 초기화 함수 호출

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        if torch.isnan(x).any():
            print("Conv1 출력에 NaN이 있습니다.")
        x = self.pool(x)  # (None, 32, 16, 16)
        x = self.relu(x)  # (None, 32, 16, 16)

        x = self.conv2(x)
        if torch.isnan(x).any():
            print("Conv2 출력에 NaN이 있습니다.")
        x = self.pool(x)  # (None, 32, 8, 8)
        x = self.relu(x)  # (None, 32, 8, 8)

        x = self.conv3(x)
        if torch.isnan(x).any():
            print("Conv3 출력에 NaN이 있습니다.")
        # x = self.pool(x)  # (None, 32, 8, 8)
        x = self.relu(x)  # (None, 32, 8, 8)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)  # (None, 10)

        return x
