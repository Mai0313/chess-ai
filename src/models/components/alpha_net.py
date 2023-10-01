import torch
import torch.nn as nn


class AlphaGoNet(nn.Module):
    def __init__(self, input_channels, board_size):
        super().__init__()

        # Initial convolutional layers
        self.conv_initial = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([self._make_residual_block() for _ in range(19)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def _make_residual_block(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        x = self.conv_initial(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = nn.ReLU()(x)

        # policy = self.policy_head(x)
        value = self.value_head(x)
        return value
