import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return base_dir / path


def build_wlasl_splits(root_path: str, seq_len: int) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    root = Path(root_path)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    # Collect labels from train split
    train_frames = root / "train" / "frames"
    labels = sorted([d.name for d in train_frames.iterdir() if d.is_dir()])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def process_split(split_name: str):
        split_dir = root / split_name / "frames"
        data = []

        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} does not exist")

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            if label not in label_to_idx:
                continue

            label_idx = label_to_idx[label]

            for seq_dir in label_dir.iterdir():
                if not seq_dir.is_dir():
                    continue

                frames = list(seq_dir.glob("*"))
                if len(frames) != seq_len:
                    continue

                data.append({
                    "video_path": str(seq_dir),
                    "label": label,
                    "label_idx": label_idx,
                })

        random.shuffle(data)
        return data

    splits["train"] = process_split("train")
    splits["val"] = process_split("val")
    splits["test"] = process_split("test")

    return splits, label_to_idx


def main():
    parser = argparse.ArgumentParser(description="Build WLASL dataset splits")

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
        help="Path to dataset root (contains train/val/test)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Where to save JSON outputs"
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=16,
        help="Number of frames per sequence"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Build splits
    splits, label_to_idx = build_wlasl_splits(
        root_path=args.data_root,
        seq_len=args.seq_len
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save outputs
    splits_path = os.path.join(args.output_dir, "wlasl_splits.json")
    labels_path = os.path.join(args.output_dir, "label_to_idx.json")

    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    with open(labels_path, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"Saved splits to: {splits_path}")
    print(f"Saved labels to: {labels_path}")
    print("Train samples:", len(splits["train"]))

    if splits["train"]:
        print("Example sample:", splits["train"][0])


if __name__ == "__main__":
    main()