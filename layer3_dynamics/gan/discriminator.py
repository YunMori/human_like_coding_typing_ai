import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TimingDiscriminator(nn.Module):
    """Conditional Bidirectional LSTM + Spectral Norm discriminator.

    Receives both the timing sequence and the context vector so that
    it can evaluate whether the generated timings are plausible *given*
    the conditioning signal (HMM state, complexity, fatigue, etc.).
    """

    def __init__(self, input_dim: int = 3, context_dim: int = 32,
                 hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.ctx_proj = spectral_norm(nn.Linear(context_dim, hidden_size))
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )
        # hidden*2 (bidirectional) + hidden (projected context)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(hidden_size * 2 + hidden_size, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1)),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x:   (B, seq_len, 3)
        # ctx: (B, context_dim)
        out, _ = self.lstm(x)            # (B, seq_len, hidden*2)
        pooled = out.mean(dim=1)         # (B, hidden*2)
        ctx_h = torch.relu(self.ctx_proj(ctx))   # (B, hidden)
        combined = torch.cat([pooled, ctx_h], dim=-1)  # (B, hidden*3)
        score = self.fc(combined)        # (B, 1)
        return score
