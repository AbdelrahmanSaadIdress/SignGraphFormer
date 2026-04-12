"""Loss functions for SLR training.

Single entry point: build_criterion(cfg, class_weights) returns the
configured loss. All training code imports only from here — never
instantiates nn.CrossEntropyLoss directly.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from configs.model_config import TrainingConfig

logger = logging.getLogger(__name__)


def build_criterion(
    cfg: TrainingConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.CrossEntropyLoss:
    """Build the training loss function.

    Uses PyTorch's built-in label smoothing so the implementation is
    numerically stable and AMP-compatible. Class weights are optional —
    pass them to compensate for WLASL100's class imbalance.

    Args:
        cfg: TrainingConfig carrying label_smoothing value.
        class_weights: Optional float32 tensor of shape (num_classes,).
                       Computed by WLASLDataset.get_class_weights().

    Returns:
        Configured CrossEntropyLoss instance (expects raw logits).
    """
    if class_weights is not None:
        logger.info(
            "Building criterion: label_smoothing=%.2f, class_weights=enabled (shape=%s)",
            cfg.label_smoothing,
            tuple(class_weights.shape),
        )
    else:
        logger.info(
            "Building criterion: label_smoothing=%.2f, class_weights=disabled",
            cfg.label_smoothing,
        )

    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
        reduction="mean",
    )
