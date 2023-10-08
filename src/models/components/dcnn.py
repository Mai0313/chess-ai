from lightning import LightningModule
from torch import nn


class DeepChessCNN(LightningModule):
    def __init__(self, num_actions: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.fc_value = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh(),
        )

        self.fc_policy = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_policy(x), self.fc_value(x)
