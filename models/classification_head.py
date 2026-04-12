"""Classification head: maps motion summary vector to class logits.

This is the ONLY module that differs between Phase 1 (100 ASL classes)
and Phase 2 (50 ESL classes). For Phase 2 transfer: freeze KeypointEncoder
and TemporalEncoder, then instantiate a new ClassificationHead(num_classes=50)
and fine-tune it alone.

No Softmax in forward() — CrossEntropyLoss expects raw logits.
"""

import logging

import torch
import torch.nn as nn

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Two-layer MLP classification head with dropout.

    Input shape:  (batch, hidden_dim=256)
    Output shape: (batch, num_classes)  — raw logits, no softmax

    Architecture:
        Linear(hidden_dim → hidden_dim) → ReLU → Dropout(head_dropout)
        → Linear(hidden_dim → num_classes)

    Args:
        cfg: ModelConfig instance. num_classes controls output dimension.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.head_dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )
        logger.info(
            "ClassificationHead: hidden_dim=%d → num_classes=%d",
            cfg.hidden_dim, cfg.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map motion summary vector to class logits.

        Args:
            x: Motion vector of shape (batch, hidden_dim).

        Returns:
            Class logits of shape (batch, num_classes). No softmax applied.
        """
        return self.net(x)