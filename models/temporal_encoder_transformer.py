"""Temporal encoder: Transformer over the sequence of per-frame embeddings.

Receives the (batch, seq_len, hidden_dim) tensor produced by applying
KeypointEncoder to all 64 frames, adds sinusoidal positional encodings,
runs 4 Transformer encoder layers, then mean-pools over time to produce
a single (batch, hidden_dim) motion summary vector.
"""

import logging
import math

import torch
import torch.nn as nn

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Not learned — encodes absolute frame position with sin/cos waves at
    exponentially spaced frequencies. Registered as a buffer so it moves
    with the model across devices.

    Args:
        hidden_dim: Model embedding dimension (must be even).
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding encodings.
    """

    def __init__(self, hidden_dim: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Position-encoded tensor of same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """Transformer-based temporal encoder over frame embedding sequences.

    Input shape:  (batch, seq_len=64, hidden_dim=256)
    Output shape: (batch, hidden_dim=256)

    Architecture:
        SinusoidalPositionalEncoding
        → N × TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        → mean pool over sequence dimension

    Args:
        cfg: ModelConfig instance carrying all hyperparameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim=cfg.hidden_dim,
            max_len=cfg.seq_len * 2,  # 2x headroom for variable-length inputs
            dropout=cfg.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,  # (batch, seq, feature) convention throughout
            norm_first=True,   # Pre-LN: more stable training than post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_transformer_layers,
            enable_nested_tensor=False,  # avoid padding-related shape issues
        )

        logger.info(
            "TemporalEncoder: %d layers, d_model=%d, nhead=%d, ff_dim=%d",
            cfg.num_transformer_layers, cfg.hidden_dim, cfg.num_heads, cfg.ff_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frame embedding sequence to a single motion vector.

        Args:
            x: Frame embeddings of shape (batch, seq_len, hidden_dim).

        Returns:
            Motion summary vector of shape (batch, hidden_dim).
        """
        x = self.pos_encoding(x)       # (B, T, D) — adds frame position info
        x = self.transformer(x)        # (B, T, D) — self-attention across frames
        return x.mean(dim=1)           # (B, D) — aggregate over time