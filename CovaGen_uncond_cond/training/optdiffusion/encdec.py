import torch
from torch import nn

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Tanh()
        )

    def forward(self, x):
        # batchsize = x.size(0)
        # x = x.view(batchsize, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x.view(batchsize, 1, 28, 28)
        return x


