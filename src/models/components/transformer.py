import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()

        # Transformer layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=num_layers
        )

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 8 * 8, d_model))

        # Fully connected layers
        self.fc1 = nn.Linear(d_model * 8 * 8, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        # Flatten the board
        x = x.view(x.size(0), 8 * 8, -1)
        x += self.positional_encoding

        # Transformer forward
        x = self.transformer_encoder(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))

        return x
