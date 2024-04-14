import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F


class ChessRNN(LightningModule):
    def __init__(self, num_actions: int):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(12, 64)

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

        # Fully connected layers for value prediction
        self.fc1_value = nn.Linear(128, 512)
        self.fc2_value = nn.Linear(512, 256)
        self.fc3_value = nn.Linear(256, 1)

        # Fully connected layers for policy
        self.fc1_policy = nn.Linear(128, 512)
        self.fc2_policy = nn.Linear(512, num_actions)

    def forward(self, board_input: torch.Tensor) -> torch.Tensor:
        # Embed the board
        x = self.embedding(board_input.view(-1, 12)).view(board_input.size(0), 8 * 8, -1)

        # Pass through LSTM layers
        lstm_out, _ = self.lstm(x)
        # Taking the last output of LSTM sequence for further predictions
        x = lstm_out[:, -1, :]

        # Value prediction
        value = F.relu(self.fc1_value(x))
        value = F.relu(self.fc2_value(value))
        value = torch.tanh(self.fc3_value(value))

        # Policy prediction
        policy_logits = F.log_softmax(self.fc2_policy(F.relu(self.fc1_policy(x))), dim=1)

        return policy_logits, value
