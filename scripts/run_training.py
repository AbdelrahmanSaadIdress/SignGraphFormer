"""Training entry point for SLR Phase 1 experiments.

Runs either the LSTM (Experiment 1) or Transformer (Experiment 2) model
on WLASL100. Both use identical preprocessing, splits, optimizer, scheduler,
and evaluation — only the temporal encoder differs.

Usage:
    # Experiment 1 — LSTM baseline
    python scripts/run_training.py \\
        --model lstm \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json

    # Experiment 2 — Transformer
    python scripts/run_training.py \\
        --model transformer \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json

    # Override hyperparameters
    python scripts/run_training.py \\
        --model lstm \\
        --splits_json data/splits/wlasl_splits_processed.json \\
        --label_json  data/splits/label_to_idx.json \\
        --batch_size 64 --num_epochs 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Ensure the project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_config import ModelConfig, TrainingConfig
from configs.vocab_config import VocabConfig
from datasets_.wlasl_dataset import WLASLDataset
from models.slr_model_lstm import SLRModelLSTM
from models.slr_model_transformer import SLRModelTransformer
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_training")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seed set to %d", seed)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    splits_json: str,
    label_json: str,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, VocabConfig]:
    """Build train, val, and test DataLoaders with the shared vocab.

    The training loader uses WeightedRandomSampler to address WLASL100's
    class imbalance. Val and test loaders are sequential (no shuffle).

    Args:
        splits_json: Path to wlasl_splits_processed.json (with keypoint_path).
        label_json: Path to label_to_idx.json.
        model_cfg: ModelConfig for shape parameters.
        train_cfg: TrainingConfig for batch size and num_workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab).
    """
    with open(label_json) as f:
        class_to_idx: dict = json.load(f)

    vocab = VocabConfig.from_class_list(
        phase="asl_wlasl100",
        classes=sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k]),
    )

    # Split-specific JSON files expected inside splits_json directory
    # OR a single dict with "train"/"val"/"test" keys
    with open(splits_json) as f:
        raw = json.load(f)

    # Support both formats: {"train": [...], "val": [...], "test": [...]}
    # or separate files
    if isinstance(raw, dict) and "train" in raw:
        split_dir = Path(splits_json).parent
        # Write individual split files if they don't exist
        for split_name in ("train", "val", "test"):
            split_path = split_dir / f"wlasl_{split_name}.json"
            if not split_path.exists():
                with open(split_path, "w") as f:
                    json.dump(raw[split_name], f)
        train_json = str(split_dir / "wlasl_train.json")
        val_json   = str(split_dir / "wlasl_val.json")
        test_json  = str(split_dir / "wlasl_test.json")
    else:
        # Assume caller passed a directory and files exist
        split_dir = Path(splits_json).parent
        train_json = str(split_dir / "wlasl_train.json")
        val_json   = str(split_dir / "wlasl_val.json")
        test_json  = str(split_dir / "wlasl_test.json")

    train_ds = WLASLDataset(train_json, model_cfg, vocab, augment=True)
    val_ds   = WLASLDataset(val_json,   model_cfg, vocab, augment=False)
    test_ds  = WLASLDataset(test_json,  model_cfg, vocab, augment=False)

    # Weighted sampler for training — draws harder classes more often
    class_weights = train_ds.get_class_weights()
    sample_weights = torch.tensor(
        [class_weights[s["label_idx"]].item() for s in train_ds.samples],
        dtype=torch.float32,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
    )

    logger.info(
        "DataLoaders ready | train=%d  val=%d  test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_loader, val_loader, test_loader, vocab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM or Transformer SLR model on WLASL100."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer"],
        help="Which temporal encoder experiment to run.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default="data/splits/wlasl_splits_processed_fake.json",
        help="Path to the processed splits JSON (keypoint_path included).",
    )
    parser.add_argument(
        "--label_json",
        type=str,
        default="data/splits/label_to_idx.json",
        help="Path to label_to_idx.json produced by download_wlasl100.py.",
    )
    # Optional hyperparameter overrides
    parser.add_argument("--batch_size",   type=int,   default=TrainingConfig.batch_size)
    parser.add_argument("--num_epochs",   type=int,   default=TrainingConfig.num_epochs)
    parser.add_argument("--learning_rate",type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--seed",         type=int,   default=TrainingConfig.seed)
    parser.add_argument("--checkpoint_dir", type=str, default=TrainingConfig.checkpoint_dir)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision.")
    parser.add_argument("--wandb_entity", type=str, default=TrainingConfig.wandb_entity,
                        help="W&B entity (team or personal account name).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Configs ---
    model_cfg  = ModelConfig()
    train_cfg  = TrainingConfig()

    # Apply CLI overrides
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.num_epochs is not None:
        train_cfg.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.checkpoint_dir is not None:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.no_amp:
        train_cfg.use_amp = False
    if args.wandb_entity is not None:
        train_cfg.wandb_entity = args.wandb_entity

    seed_everything(train_cfg.seed)

    # --- DataLoaders ---
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        splits_json=args.splits_json,
        label_json=args.label_json,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    # --- Model ---
    if args.model == "lstm":
        model = SLRModelLSTM(model_cfg)
    else:
        model = SLRModelTransformer(model_cfg)

    logger.info("Model type: %s", args.model.upper())
    param_counts = model.count_parameters()
    logger.info("Parameter counts: %s", param_counts)

    # --- Class weights for balanced loss ---
    train_ds = train_loader.dataset
    class_weights = train_ds.get_class_weights()  # type: ignore[attr-defined]

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_type=args.model,
        class_weights=class_weights,
    )
    trainer.train()

    # # --- Final test evaluation ---
    # logger.info("Loading best checkpoint for final test evaluation...")
    # best_ckpt = Path(train_cfg.checkpoint_dir) / args.model / "best.pt"
    # if best_ckpt.exists():
    #     if args.model == "lstm":
    #         best_model = SLRModelLSTM.load_checkpoint(str(best_ckpt))
    #     else:
    #         best_model = SLRModelTransformer.load_checkpoint(str(best_ckpt))

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     best_model.to(device)
    #     best_model.eval()

    #     # Re-use trainer's eval method on test set
    #     trainer.model = best_model
    #     test_metrics = trainer._eval_epoch(test_loader, split="test")

    #     logger.info(
    #         "TEST RESULTS [%s] | top-1: %.4f | top-5: %.4f | loss: %.4f",
    #         args.model.upper(),
    #         test_metrics.top1,
    #         test_metrics.top5,
    #         test_metrics.loss,
    #     )

    #     import wandb
    #     wandb.init(
    #         project=train_cfg.wandb_project,
    #         entity=train_cfg.wandb_entity,
    #         name=f"{args.model}_seed{train_cfg.seed}",
    #         resume="allow",
    #     )
    #     wandb.run.summary.update(
    #         {f"test/{k}": v for k, v in test_metrics.to_dict().items()}
    #     )
    #     wandb.finish()
    # else:
    #     logger.warning("Best checkpoint not found at %s — skipping test eval.", best_ckpt)


if __name__ == "__main__":
    main()
