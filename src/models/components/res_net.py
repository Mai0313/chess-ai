from lightning import LightningModule
from torch import nn


class ResidualBlock(LightningModule):
    def __init__(self, in_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = self.relu(x)
        return x


class ChessResNet(LightningModule):
    def __init__(self, num_actions: int, num_blocks: int = 10):
        super().__init__()

        self.initial_layers = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])

        self.fc_value = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        self.fc_policy = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc_policy(x), self.fc_value(x)
