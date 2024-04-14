import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F


class ChessCNN(LightningModule):
    def __init__(self, num_actions: int):
        super().__init__()

        # Define CNN layers
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)  # input channels: 12 for chess
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        # Fully connected layers for regression prediction
        self.fc1_value = nn.Linear(256 * 8 * 8, 1024)
        self.fc2_value = nn.Linear(1024, 512)
        self.fc3_value = nn.Linear(512, 1)

        # Fully connected layers for policy
        self.fc1_policy = nn.Linear(256 * 8 * 8, 1024)
        self.fc2_policy = nn.Linear(1024, num_actions)

    def forward(self, board_input: torch.Tensor) -> torch.Tensor:
        # Pass through CNN layers
        x = F.relu(self.bn1(self.conv1(board_input)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Flatten the board
        flattened = x.view(x.size(0), -1)

        # Value prediction
        value = F.relu(self.fc1_value(flattened))
        value = F.relu(self.fc2_value(value))
        value = torch.tanh(self.fc3_value(value))

        # Policy prediction
        policy_logits = F.log_softmax(self.fc2_policy(F.relu(self.fc1_policy(flattened))), dim=1)

        return policy_logits, value
