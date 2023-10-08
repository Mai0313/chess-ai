import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class PositionalEncoding(LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ChessTransformer(LightningModule):
    def __init__(self, d_model: int, nhead: int, num_layers: int, num_actions: int):
        super().__init__()

        self.embedding = nn.Linear(12, d_model)  # Convert each square's representation into d_model dimensions
        self.pos_encoder = PositionalEncoding(d_model, max_len=8 * 8)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected layers for regression prediction
        self.fc1 = nn.Linear(d_model * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, board_input: torch.Tensor) -> torch.Tensor:
        board_embed = self.embedding(board_input.view(-1, 12)).view(board_input.size(0), 8 * 8, -1)
        board_embed = self.pos_encoder(board_embed)

        encoded_board = self.transformer_encoder(board_embed)

        flattened_encoded_board = encoded_board.view(encoded_board.size(0), -1)
        value = F.relu(self.fc1(flattened_encoded_board))
        value = F.relu(self.fc2(value))
        value = torch.tanh(self.fc3(value))

        policy_logits = self.policy_head(flattened_encoded_board)

        return policy_logits, value
