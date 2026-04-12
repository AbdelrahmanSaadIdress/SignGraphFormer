"""Temporal encoder: Bidirectional LSTM over per-frame embeddings.

Receives (batch, seq_len, hidden_dim) from KeypointEncoder, runs a
2-layer bidirectional LSTM, projects the concatenated forward/backward
hidden states back to hidden_dim, then mean-pools over time to produce
a single (batch, hidden_dim) motion summary vector.

This is Experiment 1 — the baseline against which the Transformer is compared.
"""

import logging

import torch
import torch.nn as nn

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class TemporalEncoderLSTM(nn.Module):
    """Bidirectional LSTM temporal encoder.

    Input shape:  (batch, seq_len, hidden_dim)   e.g. (B, 16, 256)
    Output shape: (batch, hidden_dim)             e.g. (B, 256)

    Architecture:
        2-layer BiLSTM  → (B, T, hidden_dim * 2)
        Linear projection hidden_dim*2 → hidden_dim
        Mean pool over T → (B, hidden_dim)

    Args:
        cfg: ModelConfig carrying all architecture hyperparameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )

        # Bidirectional doubles the output dimension; project back to hidden_dim
        lstm_out_dim = cfg.hidden_dim * 2 if cfg.bidirectional else cfg.hidden_dim
        self.proj = nn.Linear(lstm_out_dim, cfg.hidden_dim)

        logger.info(
            "TemporalEncoderLSTM: layers=%d, hidden=%d, bidirectional=%s, out_proj=%d→%d",
            cfg.lstm_layers,
            cfg.hidden_dim,
            cfg.bidirectional,
            lstm_out_dim,
            cfg.hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frame embedding sequence to a single motion vector.

        Args:
            x: Frame embeddings of shape (batch, seq_len, hidden_dim).

        Returns:
            Motion summary vector of shape (batch, hidden_dim).
        """
        # lstm_out: (B, T, hidden_dim*2)  [bidirectional]
        lstm_out, _ = self.lstm(x)

        # Project to hidden_dim: (B, T, hidden_dim)
        projected = self.proj(lstm_out)

        # Mean pool over time: (B, hidden_dim)
        return projected.mean(dim=1)
