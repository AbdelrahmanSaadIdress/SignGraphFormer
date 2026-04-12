"""Training engine for the SLR benchmark.

Trainer is model-agnostic: it accepts any nn.Module that satisfies the
SLRModel interface (forward, save_checkpoint, count_parameters).
Pass model_type='lstm' or 'transformer' to tag W&B runs correctly.

Usage (from run_training.py):
    trainer = Trainer(model, train_loader, val_loader, cfg, train_cfg, model_type)
    trainer.train()
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import wandb

from configs.model_config import ModelConfig, TrainingConfig
from .losses import build_criterion
from .metrics import MetricsResult, compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer: AdamW,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
) -> LambdaLR:
    """Build a warmup-then-cosine learning rate schedule.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of epochs for linear warmup from 0 → lr.
        total_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.

    Returns:
        LambdaLR scheduler operating at the step level.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop with AMP, gradient clipping, W&B, and checkpointing.

    Args:
        model: SLR model with forward(x) → logits and save_checkpoint(path, extra).
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        model_cfg: ModelConfig (architecture hyperparameters).
        train_cfg: TrainingConfig (optimization hyperparameters).
        model_type: 'lstm' or 'transformer' — used for W&B run naming and
                    checkpoint file names.
        class_weights: Optional float32 tensor of shape (num_classes,)
                       for weighted CrossEntropyLoss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        model_type: str,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model_type = model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer device: %s", self.device)

        self.model.to(self.device)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = build_criterion(train_cfg, class_weights)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # Scheduler
        self.scheduler = _build_scheduler(
            self.optimizer,
            warmup_epochs=train_cfg.warmup_epochs,
            total_epochs=train_cfg.num_epochs,
            steps_per_epoch=len(train_loader),
        )

        # AMP
        self.scaler = GradScaler(enabled=train_cfg.use_amp)

        # Checkpointing
        self.checkpoint_dir = Path(train_cfg.checkpoint_dir) / model_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_top1: float = 0.0
        self.best_epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop and log results to W&B."""
        self._init_wandb()

        logger.info(
            "Starting training | model=%s | epochs=%d | device=%s",
            self.model_type,
            self.train_cfg.num_epochs,
            self.device,
        )

        for epoch in range(1, self.train_cfg.num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_top1, train_top5 = self._train_epoch(epoch)
            val_metrics = self._eval_epoch(self.val_loader, split="val")

            epoch_time = time.time() - epoch_start

            self._log_epoch(epoch, train_loss, train_top1, train_top5, val_metrics, epoch_time)
            self._maybe_save_checkpoint(epoch, val_metrics)

        logger.info(
            "Training complete | best val top-1: %.4f @ epoch %d",
            self.best_val_top1,
            self.best_epoch,
        )
        wandb.finish()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Run one full training epoch.

        Returns:
            Tuple of (mean_loss, top1_accuracy, top5_accuracy).
        """
        self.model.train()
        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch_idx, (sequences, labels) in enumerate(self.train_loader):
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.train_cfg.use_amp):
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.train_cfg.grad_clip_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.train_cfg.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        mean_loss = total_loss / len(self.train_loader)
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        from training.metrics import compute_topk_accuracy
        top1 = compute_topk_accuracy(logits_cat, labels_cat, k=1)
        top5 = compute_topk_accuracy(logits_cat, labels_cat, k=min(5, self.model_cfg.num_classes))

        logger.info(
            "Epoch %03d [train] loss=%.4f  top1=%.4f  top5=%.4f",
            epoch, mean_loss, top1, top5,
        )
        return mean_loss, top1, top5

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        split: str = "val",
    ) -> MetricsResult:
        """Evaluate the model on a DataLoader.

        Args:
            loader: DataLoader to evaluate.
            split: Label for logging ('val' or 'test').

        Returns:
            MetricsResult with all metrics populated.
        """
        self.model.eval()
        total_loss = 0.0
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for sequences, labels in loader:
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.train_cfg.use_amp):
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        mean_loss = total_loss / len(loader)
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(
            logits=logits_cat,
            labels=labels_cat,
            loss=mean_loss,
            signer_ids=None,  # signer IDs not tracked in DataLoader; handled in evaluate.py
            num_classes=self.model_cfg.num_classes,
        )

        logger.info(
            "Epoch [%s]  loss=%.4f  top1=%.4f  top5=%.4f",
            split, metrics.loss, metrics.top1, metrics.top5,
        )
        return metrics

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_top1: float,
        train_top5: float,
        val_metrics: MetricsResult,
        epoch_time: float,
    ) -> None:
        """Log all metrics to W&B for the current epoch."""
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/top1": train_top1,
                "train/top5": train_top5,
                "val/loss": val_metrics.loss,
                "val/top1": val_metrics.top1,
                "val/top5": val_metrics.top5,
                "lr": self.scheduler.get_last_lr()[0],
                "epoch_time_s": epoch_time,
            },
            step=epoch,
        )

    def _maybe_save_checkpoint(self, epoch: int, val_metrics: MetricsResult) -> None:
        """Save checkpoint if validation top-1 improved.

        Always overwrites 'latest.pt'. Only overwrites 'best.pt' on improvement.

        Args:
            epoch: Current epoch number.
            val_metrics: Validation metrics for this epoch.
        """
        extra = {
            "epoch": epoch,
            "val_top1": val_metrics.top1,
            "val_top5": val_metrics.top5,
            "val_loss": val_metrics.loss,
        }

        # Always save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        self.model.save_checkpoint(str(latest_path), extra=extra)

        # Save best
        if val_metrics.top1 > self.best_val_top1:
            self.best_val_top1 = val_metrics.top1
            self.best_epoch = epoch
            best_path = self.checkpoint_dir / "best.pt"
            self.model.save_checkpoint(str(best_path), extra=extra)
            logger.info(
                "New best checkpoint → %s  (top-1: %.4f @ epoch %d)",
                best_path,
                self.best_val_top1,
                epoch,
            )
            wandb.run.summary["best_val_top1"] = self.best_val_top1
            wandb.run.summary["best_epoch"] = epoch

    def _init_wandb(self) -> None:
        """Initialize W&B run with full config."""
        config = {
            "model_type": self.model_type,
            **vars(self.model_cfg),
            **vars(self.train_cfg),
        }
        # Dataclass properties not in __dict__; add manually
        config["input_dim"] = self.model_cfg.input_dim

        wandb.init(
            project=self.train_cfg.wandb_project,
            entity=self.train_cfg.wandb_entity,
            name=f"{self.model_type}_seed{self.train_cfg.seed}",
            config=config,
            reinit=True,
        )

        # Log parameter count breakdown
        if hasattr(self.model, "count_parameters"):
            param_counts = self.model.count_parameters()
            wandb.run.summary.update(
                {f"params/{k}": v for k, v in param_counts.items()}
            )
