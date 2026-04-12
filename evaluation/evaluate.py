"""
Standalone evaluator for the SLR benchmark comparison.

Loads a trained checkpoint, runs it against the test split, and prints
the full comparison table including latency and parameter count.
Run this after BOTH experiments are complete to produce the side-by-side table.

Usage:
    # Evaluate LSTM checkpoint
    python evaluation/evaluate.py \\
        --checkpoint checkpoints/lstm/best.pt \\
        --model_type lstm \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json

    # Evaluate Transformer checkpoint
    python evaluation/evaluate.py \\
        --checkpoint checkpoints/transformer/best.pt \\
        --model_type transformer \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json

    # Evaluate both and print comparison table
    python evaluation/evaluate.py \\
        --checkpoint checkpoints/lstm/best.pt \\
        --model_type lstm \\
        --compare_checkpoint checkpoints/transformer/best.pt \\
        --compare_model_type transformer \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_config import ModelConfig, TrainingConfig
from configs.vocab_config import VocabConfig
from datasets_.wlasl_dataset import WLASLDataset
from models.slr_model_lstm import SLRModelLSTM
from models.slr_model_transformer import SLRModelTransformer
from training.metrics import MetricsResult, compute_metrics, measure_latency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    """Load an SLR model from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model_type: 'lstm' or 'transformer'.
        device: Target device.

    Returns:
        Loaded model in eval mode, moved to device.
    """
    if model_type == "lstm":
        model = SLRModelLSTM.load_checkpoint(checkpoint_path, device=str(device))
    elif model_type == "transformer":
        model = SLRModelTransformer.load_checkpoint(checkpoint_path, device=str(device))
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'lstm' or 'transformer'.")
    model.to(device)
    model.eval()
    return model


def build_test_loader(
    splits_json: str,
    label_json: str,
    model_cfg: ModelConfig,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, VocabConfig]:
    """Build the test DataLoader and vocab.

    Args:
        splits_json: Path to processed splits JSON or test-split JSON.
        label_json: Path to label_to_idx.json.
        model_cfg: ModelConfig for shape parameters.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (test_loader, vocab).
    """
    with open(label_json) as f:
        class_to_idx: Dict[str, int] = json.load(f)

    vocab = VocabConfig.from_class_list(
        phase="asl_wlasl100",
        classes=sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k]),
    )

    # Accept both the combined splits file and a direct test-split JSON
    with open(splits_json) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "test" in raw:
        test_samples = raw["test"]
        test_json = str(Path(splits_json).parent / "_eval_test_tmp.json")
        with open(test_json, "w") as f:
            json.dump(test_samples, f)
    else:
        # Assume it's already the test-split JSON
        test_json = splits_json

    test_ds = WLASLDataset(test_json, model_cfg, vocab, augment=False)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, vocab


@torch.no_grad()
def run_evaluation(
    model,
    loader: DataLoader,
    model_cfg: ModelConfig,
    device: torch.device,
) -> MetricsResult:
    """Run full evaluation on a DataLoader.

    Collects signer_ids from the dataset if available to compute
    cross-signer accuracy.

    Args:
        model: Loaded SLR model in eval mode.
        loader: Test DataLoader.
        model_cfg: ModelConfig.
        device: Evaluation device.

    Returns:
        MetricsResult with all metrics populated.
    """
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(reduction="mean")

    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0

    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        logits = model(sequences)
        loss = criterion(logits, labels_dev)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    mean_loss = total_loss / len(loader)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    # Extract signer_ids if the dataset tracks them
    signer_ids: Optional[List[str]] = None
    ds = loader.dataset
    if hasattr(ds, "samples") and ds.samples and "signer_id" in ds.samples[0]:
        signer_ids = [s["signer_id"] for s in ds.samples]

    return compute_metrics(
        logits=logits_cat,
        labels=labels_cat,
        loss=mean_loss,
        signer_ids=signer_ids,
        num_classes=model_cfg.num_classes,
    )


def print_results_table(
    results: Dict[str, MetricsResult],
    latencies: Dict[str, "LatencyResult"],  # type: ignore[name-defined]
) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: Dict mapping model_type → MetricsResult.
        latencies: Dict mapping model_type → LatencyResult.
    """
    header = (
        f"\n{'Metric':<30}"
        + "".join(f"{name:>20}" for name in results)
    )
    separator = "-" * (30 + 20 * len(results))

    rows = [
        ("Top-1 Accuracy",       lambda r, _: f"{r.top1 * 100:.2f}%"),
        ("Top-5 Accuracy",       lambda r, _: f"{r.top5 * 100:.2f}%"),
        ("Test Loss",            lambda r, _: f"{r.loss:.4f}"),
        ("Cross-signer Top-1",   lambda r, _: f"{r.cross_signer_top1 * 100:.2f}%" if r.cross_signer_top1 else "N/A"),
        ("Num Samples",          lambda r, _: str(r.num_samples)),
        ("Param Count",          lambda _, l: f"{l.param_count:,}"),
        ("CPU Latency (ms)",     lambda _, l: f"{l.cpu_ms:.2f}"),
        ("GPU Latency (ms)",     lambda _, l: f"{l.gpu_ms:.2f}" if l.gpu_ms else "N/A"),
    ]

    print(header)
    print(separator)
    for label, fn in rows:
        row = f"{label:<30}"
        for name in results:
            row += f"{fn(results[name], latencies[name]):>20}"
        print(row)
    print(separator)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or two SLR checkpoints and print a comparison table."
    )
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--model_type",     type=str, required=True, choices=["lstm", "transformer"])
    parser.add_argument("--compare_checkpoint",  type=str, default=None,
                        help="Second checkpoint to compare against (optional).")
    parser.add_argument("--compare_model_type",  type=str, default=None, choices=["lstm", "transformer"])
    parser.add_argument("--splits_json",    type=str, default="data/splits/wlasl_splits_processed.json")
    parser.add_argument("--label_json",     type=str, default="data/splits/label_to_idx.json")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--num_workers",    type=int, default=2)
    parser.add_argument("--log_wandb",      action="store_true",
                        help="Log results to W&B as summary metrics.")
    parser.add_argument("--wandb_project",  type=str, default="slr-phase1-wlasl100")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation device: %s", device)

    model_cfg = ModelConfig()

    test_loader, vocab = build_test_loader(
        splits_json=args.splits_json,
        label_json=args.label_json,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info("Test samples: %d", len(test_loader.dataset))

    # Collect experiments to evaluate
    experiments = [(args.model_type, args.checkpoint)]
    if args.compare_checkpoint is not None:
        if args.compare_model_type is None:
            logger.error("--compare_model_type required when --compare_checkpoint is set.")
            sys.exit(1)
        experiments.append((args.compare_model_type, args.compare_checkpoint))

    all_results: Dict[str, MetricsResult] = {}
    all_latencies: Dict[str, object] = {}

    for model_type, ckpt_path in experiments:
        logger.info("Evaluating %s from %s", model_type.upper(), ckpt_path)

        model = load_model(ckpt_path, model_type, device)
        metrics = run_evaluation(model, test_loader, model_cfg, device)
        latency = measure_latency(model, model_cfg)

        all_results[model_type] = metrics
        all_latencies[model_type] = latency

        logger.info(
            "[%s] top-1=%.4f | top-5=%.4f | loss=%.4f | CPU=%.1fms | GPU=%s | params=%s",
            model_type.upper(),
            metrics.top1, metrics.top5, metrics.loss,
            latency.cpu_ms,
            f"{latency.gpu_ms:.1f}ms" if latency.gpu_ms else "N/A",
            f"{latency.param_count:,}",
        )

    # Print comparison table
    print_results_table(all_results, all_latencies)

    # Per-class accuracy summary (bottom 10 worst classes)
    for model_type, metrics in all_results.items():
        sorted_classes = sorted(metrics.per_class_accuracy.items(), key=lambda x: x[1])
        worst_10 = sorted_classes[:10]
        logger.info(
            "[%s] 10 worst classes: %s",
            model_type.upper(),
            [(c, f"{acc:.2f}") for c, acc in worst_10],
        )

    # Optional W&B logging
    if args.log_wandb:
        import wandb
        for model_type, metrics in all_results.items():
            lat = all_latencies[model_type]
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"eval_{model_type}",
                reinit=True,
            )
            wandb.run.summary.update(
                {
                    **{f"test/{k}": v for k, v in metrics.to_dict().items()},
                    "latency/cpu_ms": lat.cpu_ms,
                    "latency/gpu_ms": lat.gpu_ms or -1.0,
                    "params/total": lat.param_count,
                }
            )
            if metrics.confusion_matrix is not None:
                wandb.log(
                    {"confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=metrics.confusion_matrix.sum(axis=1).tolist(),  # placeholder
                        preds=None,
                        class_names=[str(i) for i in range(model_cfg.num_classes)],
                    )}
                )
            wandb.finish()


if __name__ == "__main__":
    main()
