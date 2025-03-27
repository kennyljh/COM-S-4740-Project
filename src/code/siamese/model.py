import torch
import torch.nn as nn
import torch.nn.functional as F

class SignatureEncoder(nn.Module):
    def __init__(self):
        super(SignatureEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x62x62

            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x29x29

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 128x13x13
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 13 * 13, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # embedding size
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = SignatureEncoder()

    def forward(self, input1, input2):
        output1 = self.encoder(input1)
        output2 = self.encoder(input2)
        distance = F.pairwise_distance(output1, output2)
        return distance
