"""Evaluation metrics for the SLR benchmark.

All metrics are computed from accumulated (logits, labels) tensors so they
work identically during validation, test, and standalone evaluation.

Public API:
    compute_topk_accuracy(logits, labels, k) → float
    compute_metrics(logits, labels, signer_ids, k_values) → MetricsResult
    measure_latency(model, device, cfg, n_runs) → LatencyResult
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetricsResult:
    """All evaluation metrics for one model checkpoint.

    Attributes:
        top1: Top-1 accuracy in [0, 1].
        top5: Top-5 accuracy in [0, 1].
        loss: Mean cross-entropy loss over the split.
        cross_signer_top1: Top-1 accuracy computed per signer then averaged.
                           None if signer_ids were not provided.
        per_class_accuracy: Dict mapping class index to per-class accuracy.
        confusion_matrix: (num_classes, num_classes) numpy array.
        num_samples: Total samples evaluated.
    """
    top1: float
    top5: float
    loss: float
    cross_signer_top1: Optional[float] = None
    per_class_accuracy: Dict[int, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Flat dict suitable for W&B logging (excludes confusion matrix)."""
        d: Dict[str, float] = {
            "top1": self.top1,
            "top5": self.top5,
            "loss": self.loss,
            "num_samples": float(self.num_samples),
        }
        if self.cross_signer_top1 is not None:
            d["cross_signer_top1"] = self.cross_signer_top1
        return d


@dataclass
class LatencyResult:
    """Inference latency measurements.

    Attributes:
        cpu_ms: Median single-sample latency on CPU (milliseconds).
        gpu_ms: Median single-sample latency on GPU (milliseconds). None if no GPU.
        param_count: Total trainable parameters.
    """
    cpu_ms: float
    gpu_ms: Optional[float]
    param_count: int


# ---------------------------------------------------------------------------
# Core accuracy helpers
# ---------------------------------------------------------------------------

def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute top-k accuracy over a batch or full split.

    Args:
        logits: Raw model outputs of shape (N, num_classes).
        labels: Ground-truth integer labels of shape (N,).
        k: k for top-k accuracy.

    Returns:
        Top-k accuracy in [0, 1].
    """
    with torch.no_grad():
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = top_k_preds.eq(labels.unsqueeze(1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()


# ---------------------------------------------------------------------------
# Full metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: float,
    signer_ids: Optional[List[str]] = None,
    num_classes: int = 100,
) -> MetricsResult:
    """Compute the full evaluation metric suite from accumulated tensors.

    Args:
        logits: Float tensor of shape (N, num_classes) — raw model outputs.
        labels: Long tensor of shape (N,) — ground-truth class indices.
        loss: Pre-computed scalar loss (averaged over the split).
        signer_ids: Optional list of signer identifier strings, length N.
                    Required for cross-signer accuracy.
        num_classes: Number of output classes.

    Returns:
        MetricsResult with all metrics populated.
    """
    logits = logits.cpu()
    labels = labels.cpu()

    top1 = compute_topk_accuracy(logits, labels, k=1)
    top5 = compute_topk_accuracy(logits, labels, k=min(5, num_classes))

    # --- Per-class accuracy and confusion matrix ---
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(labels_np, preds):
        confusion[true, pred] += 1

    per_class: Dict[int, float] = {}
    for c in range(num_classes):
        total_c = confusion[c].sum()
        per_class[c] = float(confusion[c, c] / total_c) if total_c > 0 else 0.0

    # --- Cross-signer accuracy (optional) ---
    cross_signer_top1: Optional[float] = None
    if signer_ids is not None:
        # Compute per-signer accuracy then average across signers
        signer_map: Dict[str, Tuple[List[int], List[int]]] = {}
        for i, sid in enumerate(signer_ids):
            if sid not in signer_map:
                signer_map[sid] = ([], [])
            signer_map[sid][0].append(int(labels_np[i]))
            signer_map[sid][1].append(int(preds[i]))

        signer_accs: List[float] = []
        for sid, (true_list, pred_list) in signer_map.items():
            correct = sum(t == p for t, p in zip(true_list, pred_list))
            signer_accs.append(correct / len(true_list))

        cross_signer_top1 = float(np.mean(signer_accs)) if signer_accs else None

    return MetricsResult(
        top1=top1,
        top5=top5,
        loss=loss,
        cross_signer_top1=cross_signer_top1,
        per_class_accuracy=per_class,
        confusion_matrix=confusion,
        num_samples=len(labels_np),
    )


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def measure_latency(
    model: nn.Module,
    cfg: ModelConfig,
    n_runs: int = 200,
    warmup_runs: int = 20,
) -> LatencyResult:
    """Measure single-sample CPU and GPU inference latency.

    Uses median over n_runs to be robust to outliers. GPU timing uses
    CUDA events for accuracy (avoids Python-level overhead).

    Args:
        model: The SLR model (eval mode expected).
        cfg: ModelConfig to construct a dummy input of the right shape.
        n_runs: Number of timed forward passes.
        warmup_runs: Untimed warm-up passes before measurement.

    Returns:
        LatencyResult with cpu_ms, gpu_ms, and param_count.
    """
    model.eval()
    dummy = torch.zeros(1, cfg.seq_len, cfg.input_dim)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- CPU latency ---
    cpu_model = model.cpu()
    cpu_input = dummy.cpu()
    with torch.no_grad():
        for _ in range(warmup_runs):
            cpu_model(cpu_input)
        cpu_times: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            cpu_model(cpu_input)
            cpu_times.append((time.perf_counter() - t0) * 1000.0)
    cpu_ms = float(np.median(cpu_times))

    # --- GPU latency (if available) ---
    gpu_ms: Optional[float] = None
    if torch.cuda.is_available():
        gpu_model = model.cuda()
        gpu_input = dummy.cuda()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(warmup_runs):
                gpu_model(gpu_input)
            torch.cuda.synchronize()
            gpu_times: List[float] = []
            for _ in range(n_runs):
                start_event.record()
                gpu_model(gpu_input)
                end_event.record()
                torch.cuda.synchronize()
                gpu_times.append(start_event.elapsed_time(end_event))
        gpu_ms = float(np.median(gpu_times))

    logger.info(
        "Latency — CPU: %.2f ms | GPU: %s | params: %s",
        cpu_ms,
        f"{gpu_ms:.2f} ms" if gpu_ms is not None else "N/A",
        f"{param_count:,}",
    )

    return LatencyResult(cpu_ms=cpu_ms, gpu_ms=gpu_ms, param_count=param_count)
