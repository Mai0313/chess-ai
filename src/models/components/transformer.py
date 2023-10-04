import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class ChessTransformer(LightningModule):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6) -> None:
        super().__init__()

        # Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_layers
        )

        # Positional Encoding with normal distribution initialization
        self.positional_encoding = nn.Parameter(torch.empty(1, 8 * 8, d_model))
        torch.nn.init.normal_(self.positional_encoding)

        # Fully connected layers
        self.fc1 = nn.Linear(d_model * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, board_input: torch.Tensor) -> torch.Tensor:
        # Flatten the board
        board_input = board_input.view(board_input.size(0), 8 * 8, -1)
        board_input += self.positional_encoding

        # Transformer forward
        encoded_board = self.transformer_encoder(board_input)

        # Fully connected layers forward pass
        flattened_encoded_board = encoded_board.view(encoded_board.size(0), -1)
        output = F.relu(self.fc1(flattened_encoded_board))
        output = self.dropout1(output)
        output = F.relu(self.fc2(output))
        output = self.dropout2(output)
        output = F.sigmoid(self.fc3(output))

        return output
