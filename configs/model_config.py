"""
Model and training hyperparameter configuration.

All magic numbers live here. Import ModelConfig and TrainingConfig
everywhere else — never hardcode values in model or training files.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the SLR model.

    Attributes:
        hidden_dim: Embedding dimension used throughout spatial and temporal encoders.
        num_heads: Number of attention heads in the Transformer encoder.
        num_transformer_layers: Depth of the Transformer encoder.
        ff_dim: Feed-forward hidden dimension inside each Transformer layer.
        dropout: Dropout probability applied inside encoder layers.
        head_dropout: Dropout probability inside the classification head.
        seq_len: Fixed number of frames per clip after padding/cropping.
                 Must match the value used in extract_keypoints.py (--seq_len).
        num_joints: Total skeleton joints (21 left hand + 21 right hand + 33 pose).
        coords_per_joint: Coordinate dimensions per joint (x, y, z).
        num_classes: Output classes — 100 for WLASL100.
        lstm_layers: Number of stacked LSTM layers (used by TemporalEncoderLSTM).
        bidirectional: Whether the LSTM runs in both temporal directions.
    """
    hidden_dim: int = 256
    num_heads: int = 8
    num_transformer_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    head_dropout: float = 0.3
    seq_len: int = 64          # matches --seq_len 64 used during extraction
    num_joints: int = 75
    coords_per_joint: int = 3
    num_classes: int = 100
    lstm_layers: int = 4
    bidirectional: bool = True

    @property
    def input_dim(self) -> int:
        """Derived: total floats per frame (num_joints * coords_per_joint)."""
        return self.num_joints * self.coords_per_joint


@dataclass
class TrainingConfig:
    """Training loop and optimization hyperparameters.

    Attributes:
        batch_size: Number of clips per gradient step.
        num_epochs: Total training epochs.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        warmup_epochs: Epochs for linear LR warmup before cosine decay.
        grad_clip_norm: Max gradient norm for clipping (0.0 = disabled).
        label_smoothing: Label smoothing epsilon for CrossEntropyLoss.
        num_workers: DataLoader worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        seed: Global random seed for reproducibility.
        checkpoint_dir: Directory to save best and latest checkpoints.
        wandb_project: W&B project name.
        wandb_entity: W&B entity/team name (None = personal account).
        use_amp: Whether to use automatic mixed precision (FP16) training.
    """
    batch_size: int = 256
    num_epochs: int = 200 #80
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "slr-phase1-wlasl100"
    wandb_entity: Optional[str] = None
    use_amp: bool = True
