import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class ChessTransformer(LightningModule):
    def __init__(
        self, d_model: int, nhead: int, num_layers: int, num_actions: int, action_type: str
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(
            12, d_model
        )  # Convert each square's representation into d_model dimensions

        # Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_layers
        )

        # Positional Encoding with normal distribution initialization
        self.positional_encoding = nn.Parameter(torch.randn(1, 8 * 8, d_model) * 0.01)

        # Fully connected layers for regression prediction
        self.fc1 = nn.Linear(d_model * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_actions),
            nn.Softmax(dim=1),
        )
        self.action_type = action_type

    def forward(self, board_input: torch.Tensor) -> torch.Tensor:
        # Embed the board
        board_embed = self.embedding(board_input.view(-1, 12)).view(board_input.size(0), 8 * 8, -1)

        # Add positional encoding
        board_embed += self.positional_encoding

        # Transformer encoding
        encoded_board = self.transformer_encoder(board_embed)

        # Fully connected layers for regression
        flattened_encoded_board = encoded_board.view(encoded_board.size(0), -1)
        value = F.relu(self.fc1(flattened_encoded_board))
        value = self.dropout1(value)
        value = F.relu(self.fc2(value))
        value = self.dropout2(value)
        value = torch.tanh(self.fc3(value))  # Use tanh to ensure value is between -1 and 1
        policy_logits = self.policy_head(flattened_encoded_board)

        if self.action_type == "best" or self.action_type == "greedy":
            # This will find the best action for the current board state.
            action = torch.argmax(policy_logits).item()
        elif self.action_type == "random" or self.action_type == "train":
            # This will seek more exploration for training.
            action = torch.multinomial(policy_logits[0], 1).item()

        return policy_logits
