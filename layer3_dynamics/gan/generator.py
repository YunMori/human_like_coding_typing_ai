import torch
import torch.nn as nn


class TimingGenerator(nn.Module):
    """LSTM-based generator: noise(64) + context(32) -> timing sequences."""

    def __init__(self, noise_dim: int = 64, context_dim: int = 32,
                 hidden_size: int = 128, num_layers: int = 3, seq_len: int = 32):
        super().__init__()
        self.noise_dim = noise_dim
        self.context_dim = context_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(noise_dim + context_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3),   # [key_down_ms, key_up_ms, gap_ms]
            nn.Softplus(),       # ensure positive outputs
        )

    def forward(self, noise: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # noise: (B, noise_dim), context: (B, context_dim)
        x = torch.cat([noise, context], dim=-1)  # (B, noise+context)
        x = self.input_proj(x)                    # (B, hidden)
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq_len, hidden)
        out, _ = self.lstm(x)                     # (B, seq_len, hidden)
        timing = self.output_proj(out)            # (B, seq_len, 3)
        return timing
