# import os
# import json
# import argparse
# import random
# from pathlib import Path
# from typing import Dict, List, Tuple


# def resolve_path(path_str: str, base_dir: Path) -> Path:
#     path = Path(path_str)
#     if path.is_absolute():
#         return path
#     return base_dir / path


# def load_class_list(class_list_path: Path) -> Dict[int, str]:
#     class_names = {}

#     with open(class_list_path, "r") as f:
#         for idx, line in enumerate(f):
#             class_names[idx] = line.split()[-1]

#     return class_names


# def build_wlasl_splits(
#     nslt_path: str,
#     class_list_path: str,
#     data_root: str
# ) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:

#     nslt_path = Path(nslt_path)
#     class_list_path = Path(class_list_path)
#     data_root = Path(data_root)

#     # -----------------------------
#     # Load data
#     # -----------------------------
#     with open(nslt_path, "r") as f:
#         data = json.load(f)

#     class_names = load_class_list(class_list_path)

#     label_to_idx = {v: k for k, v in class_names.items()}

#     splits = {
#         "train": [],
#         "val": [],
#         "test": []
#     }

#     # -----------------------------
#     # Process samples
#     # -----------------------------
#     for video_id, sample in data.items():
#         subset = sample["subset"]

#         if subset not in splits:
#             continue

#         class_id = sample["action"][0]

#         if class_id not in class_names:
#             continue

#         label = class_names[class_id]

#         # ⚠️ remove leading zeros if needed
#         clean_video_id = str(int(video_id))

#         video_path = data_root / subset / "frames" / label / clean_video_id

#         splits[subset].append({
#             "video_path": str(video_path),
#             "label": label,
#             "label_idx": class_id,
#         })

#     # Shuffle each split
#     for split in splits:
#         random.shuffle(splits[split])

#     return splits, label_to_idx


# def main():
#     parser = argparse.ArgumentParser(description="Build WLASL dataset splits (from JSON)")

#     parser.add_argument(
#         "--nslt_path",
#         type=str,
#         required=True,
#         help="Path to nslt_2000.json"
#     )

#     parser.add_argument(
#         "--class_list_path",
#         type=str,
#         required=True,
#         help="Path to wlasl_class_list.txt"
#     )

#     parser.add_argument(
#         "--data_root",
#         type=str,
#         default="data/raw",
#         help="Root path containing train/val/test folders"
#     )

#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="data/splits",
#         help="Where to save JSON outputs"
#     )

#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed"
#     )

#     args = parser.parse_args()

#     # Set seed
#     random.seed(args.seed)

#     # Build splits
#     splits, label_to_idx = build_wlasl_splits(
#         nslt_path=args.nslt_path,
#         class_list_path=args.class_list_path,
#         data_root=args.data_root
#     )

#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Save outputs
#     splits_path = os.path.join(args.output_dir, "wlasl_splits.json")
#     labels_path = os.path.join(args.output_dir, "label_to_idx.json")

#     with open(splits_path, "w") as f:
#         json.dump(splits, f, indent=2)

#     with open(labels_path, "w") as f:
#         json.dump(label_to_idx, f, indent=2)

#     print(f"Saved splits to: {splits_path}")
#     print(f"Saved labels to: {labels_path}")
#     print("Train samples:", len(splits["train"]))

#     if splits["train"]:
#         print("Example sample:", splits["train"][0])


# if __name__ == "__main__":
#     main()