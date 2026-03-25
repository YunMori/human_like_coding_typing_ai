import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TimingDiscriminator(nn.Module):
    """Bidirectional LSTM + Spectral Norm discriminator (WGAN-GP style)."""

    def __init__(self, input_dim: int = 3, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(hidden_size * 2, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, 3)
        out, _ = self.lstm(x)            # (B, seq_len, hidden*2)
        pooled = out.mean(dim=1)         # (B, hidden*2)
        score = self.fc(pooled)          # (B, 1)
        return score
