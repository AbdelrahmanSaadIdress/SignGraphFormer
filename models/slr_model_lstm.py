"""Full SLR model — Experiment 1: KeypointEncoder + LSTM + ClassificationHead.

SLRModelLSTM is a drop-in counterpart to SLRModelTransformer. Both share
the identical forward signature, checkpoint format, and sub-module names
so the trainer, evaluator, and inference scripts are model-agnostic.

forward(x: Tensor[B, seq_len, 225]) → logits: Tensor[B, num_classes]
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from models.classification_head import ClassificationHead
from models.keypoint_encoder import KeypointEncoder
from models.temporal_encoder_lstm import TemporalEncoderLSTM

logger = logging.getLogger(__name__)


class SLRModelLSTM(nn.Module):
    """End-to-end Sign Language Recognition model with LSTM temporal encoder.

    Pipeline:
        1. Reshape  (B, T, 225)       → (B*T, 75, 3)
        2. KeypointEncoder            → (B*T, hidden_dim)
        3. Reshape  (B*T, hidden_dim) → (B, T, hidden_dim)
        4. TemporalEncoderLSTM        → (B, hidden_dim)
        5. ClassificationHead         → (B, num_classes)

    Args:
        cfg: ModelConfig instance defining all architecture parameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.keypoint_encoder = KeypointEncoder(cfg)
        self.temporal_encoder = TemporalEncoderLSTM(cfg)
        self.classification_head = ClassificationHead(cfg)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("SLRModelLSTM initialized | trainable params: %s", f"{num_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass from raw keypoint sequence to class logits.

        Args:
            x: Keypoint sequences of shape (batch, seq_len, input_dim)
               where input_dim = num_joints * coords_per_joint = 225.

        Returns:
            Class logits of shape (batch, num_classes). No softmax applied.
        """
        batch, seq_len, _ = x.shape

        # (B, T, 225) → (B*T, 75, 3)
        x = x.view(batch * seq_len, self.cfg.num_joints, self.cfg.coords_per_joint)

        # (B*T, 75, 3) → (B*T, hidden_dim)
        frame_embeddings = self.keypoint_encoder(x)

        # (B*T, hidden_dim) → (B, T, hidden_dim)
        frame_embeddings = frame_embeddings.view(batch, seq_len, self.cfg.hidden_dim)

        # (B, T, hidden_dim) → (B, hidden_dim)
        motion_vector = self.temporal_encoder(frame_embeddings)

        # (B, hidden_dim) → (B, num_classes)
        return self.classification_head(motion_vector)

    def save_checkpoint(self, path: str, extra: Dict[str, Any] = None) -> None:
        """Save model weights and config to a checkpoint file.

        Args:
            path: Destination .pt file path.
            extra: Optional metadata dict (e.g., epoch, val_acc).
        """
        payload = {
            "model_state_dict": self.state_dict(),
            "model_config": self.cfg,
            "model_type": "lstm",
            "extra": extra or {},
        }
        torch.save(payload, path)
        logger.info("Checkpoint saved → %s", path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "SLRModelLSTM":
        """Load model from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint file.
            device: Target device string.

        Returns:
            SLRModelLSTM instance in eval mode.
        """
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg: ModelConfig = payload["model_config"]
        model = cls(cfg)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        logger.info("Checkpoint loaded ← %s (device=%s)", path, device)
        return model

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable parameter counts per sub-module.

        Returns:
            Dict mapping sub-module name to parameter count, plus 'total'.
        """
        counts = {}
        for name, module in [
            ("keypoint_encoder", self.keypoint_encoder),
            ("temporal_encoder", self.temporal_encoder),
            ("classification_head", self.classification_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
