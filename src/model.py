"""A small feed-forward neural network for binary classification."""
import torch
from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits. BCEWithLogitsLoss applies the sigmoid internally.
        return self.network(x)
