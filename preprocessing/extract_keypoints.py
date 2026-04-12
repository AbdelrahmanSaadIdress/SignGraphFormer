"""MediaPipe Holistic keypoint extraction pipeline for WLASL100.

Reads each sequence directory (16 JPEG frames), extracts per-frame landmarks
using MediaPipe Holistic, normalizes the skeleton, pads to seq_len=16 frames,
and saves a float32 .npy file of shape (16, 225).

This script is run once offline. Training never touches raw frames again.

Usage:
    python preprocessing/extract_keypoints.py \
        --splits_json data/splits/wlasl_splits.json \
        --output_dir data/processed \
        --num_workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp_lib
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("extract_keypoints")

# ---------------------------------------------------------------------------
# Constants (mirrors configs/model_config.py — do NOT duplicate in prod,
# import from there; kept here for script self-containment)
# ---------------------------------------------------------------------------
SEQ_LEN: int = 16          # target temporal length after pad/crop
NUM_JOINTS: int = 75       # 21 left hand + 21 right hand + 33 pose
COORDS_PER_JOINT: int = 3  # x, y, z
FEATURE_DIM: int = NUM_JOINTS * COORDS_PER_JOINT  # 225

# MediaPipe landmark counts
_N_HAND: int = 21
_N_POSE: int = 33

# Pose landmark indices for normalization anchors
_NOSE_IDX: int = 0          # in pose landmarks (index within the 33 pose joints)
_L_SHOULDER_IDX: int = 11   # in pose landmarks
_R_SHOULDER_IDX: int = 12   # in pose landmarks


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python random and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_frame(joints: np.ndarray) -> np.ndarray:
    """Center and scale one frame's joint coordinates.

    Origin is shifted to the nose tip (pose landmark 0).
    Scale is set so that the Euclidean distance between the left and right
    shoulders equals 1.0. If shoulder width is zero (both shoulders missing),
    the frame is returned un-scaled but centered.

    Args:
        joints: Float array of shape (75, 3) — raw MediaPipe coords.

    Returns:
        Normalized float32 array of shape (75, 3).
    """
    joints = joints.copy()

    # Pose joints start at index 42 in our concatenation order:
    # [left_hand(0:21), right_hand(21:42), pose(42:75)]
    pose_offset = _N_HAND * 2  # 42

    nose = joints[pose_offset + _NOSE_IDX].copy()          # (3,)
    l_shoulder = joints[pose_offset + _L_SHOULDER_IDX]     # (3,)
    r_shoulder = joints[pose_offset + _R_SHOULDER_IDX]     # (3,)

    # Translate
    joints -= nose

    # Scale
    shoulder_width = float(np.linalg.norm(l_shoulder - r_shoulder))
    joints /= shoulder_width + 1e-6

    return joints.astype(np.float32)


# ---------------------------------------------------------------------------
# Single-frame extraction
# ---------------------------------------------------------------------------

def extract_frame_landmarks(
    frame_bgr: np.ndarray,
    holistic: mp_lib.solutions.holistic.Holistic,
) -> np.ndarray:
    """Run MediaPipe Holistic on one BGR frame and return joint array.

    Missing hands or pose are filled with zeros so the output is always
    a fixed-size array regardless of occlusion.

    Args:
        frame_bgr: BGR uint8 image from cv2.imread.
        holistic: A pre-initialised MediaPipe Holistic instance.

    Returns:
        Float32 array of shape (75, 3): [left_hand, right_hand, pose].
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    left_hand = np.zeros((_N_HAND, 3), dtype=np.float32)
    right_hand = np.zeros((_N_HAND, 3), dtype=np.float32)
    pose = np.zeros((_N_POSE, 3), dtype=np.float32)

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i] = [lm.x, lm.y, lm.z]

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i] = [lm.x, lm.y, lm.z]

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y, lm.z]

    joints = np.concatenate([left_hand, right_hand, pose], axis=0)  # (75, 3)
    return joints


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------

def extract_frame_index(path: Path) -> int:
    # filename: secretary_15.jpg → 15
    return int(path.stem.split("_")[-1])

def extract_sequence(
    seq_dir: Path,
    holistic: mp_lib.solutions.holistic.Holistic,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Extract, normalize, and pad/crop a full video sequence.

    Frames are read in alphabetical order (matches the naming convention
    <label>_<seq_id>_<frame_idx>.jpg used in the WLASL100 preprocessed
    dataset).

    Padding strategy: zero-pad at the end. Zero frames represent absent
    motion, which is semantically neutral for the Transformer.

    Args:
        seq_dir: Path to the directory containing JPEG frames.
        holistic: Pre-initialised MediaPipe Holistic instance.
        seq_len: Target sequence length (default 16).

    Returns:
        Float32 array of shape (seq_len, 225).
    """
    frame_paths = sorted(seq_dir.glob("*.jpg"), key=extract_frame_index)
    if not frame_paths:
        # Try PNG as fallback
        frame_paths = sorted(seq_dir.glob("*.png"), key=extract_frame_index)

    if not frame_paths:
        logger.warning("No frames found in %s — returning zeros.", seq_dir)
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    raw_frames: List[np.ndarray] = []
    for fp in frame_paths:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            logger.warning("Could not read frame %s — skipping.", fp)
            continue
        joints = extract_frame_landmarks(bgr, holistic)   # (75, 3)
        joints = normalize_frame(joints)                  # (75, 3), normalised
        raw_frames.append(joints.reshape(FEATURE_DIM))   # (225,)

    T = len(raw_frames)
    if T == 0:
        return np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)

    frames_arr = np.stack(raw_frames, axis=0)  # (T, 225)

    # Pad or crop to seq_len
    if T >= seq_len:
        # Centre-crop: discard equally from both ends
        start = (T - seq_len) // 2
        result = frames_arr[start : start + seq_len]
    else:
        # Zero-pad at the end
        pad = np.zeros((seq_len - T, FEATURE_DIM), dtype=np.float32)
        result = np.concatenate([frames_arr, pad], axis=0)

    print(result.shape)
    return result.astype(np.float32)  # (T, 225)


# ---------------------------------------------------------------------------
# Worker function (used for multiprocessing)
# ---------------------------------------------------------------------------

def _process_sample(args: Tuple[dict, str]) -> Optional[str]:
    """Process a single dataset sample (top-level for pickling).

    Args:
        args: Tuple of (sample_dict, output_dir_str).
            sample_dict has keys: video_path, label, label_idx.

    Returns:
        The output .npy path on success, None on failure.
    """
    sample, output_dir_str = args
    output_dir = Path(output_dir_str)
    seq_dir = Path(sample["video_path"])

    # Mirror the relative path structure: <label>/<seq_name>
    label = sample["label"]
    seq_name = seq_dir.name
    out_path = output_dir / label / f"{seq_name}.npy"

    if out_path.exists():
        return str(out_path)  # already processed, skip

    out_path.parent.mkdir(parents=True, exist_ok=True)

    mp_holistic = mp_lib.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,   # each call is independent; no tracking state
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:
        try:
            arr = extract_sequence(seq_dir, holistic)
            np.save(str(out_path), arr)
            return str(out_path)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("extract_keypoints").error(
                "Failed on %s: %s", seq_dir, exc
            )
            return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_splits(
    splits: Dict[str, List[dict]],
    output_dir: Path,
    num_workers: int = 1,
) -> Dict[str, List[dict]]:
    """Run extraction for all splits and return updated split dicts.

    Each sample dict in the returned structure gains a `keypoint_path` key
    pointing to the saved .npy file. Samples that failed extraction are
    dropped with a warning.

    Args:
        splits: Dict with keys 'train', 'val', 'test', each a list of sample
                dicts (video_path, label, label_idx).
        output_dir: Root directory for .npy output files.
        num_workers: Number of parallel worker processes.

    Returns:
        Updated splits dict with keypoint_path added and failures removed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_splits: Dict[str, List[dict]] = {}

    for split_name, samples in splits.items():
        logger.info("Processing split '%s' — %d sequences.", split_name, len(samples))

        worker_args = [(s, str(output_dir)) for s in samples]

        if num_workers > 1:
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_process_sample, worker_args)
        else:
            results = [_process_sample(a) for a in worker_args]

        updated: List[dict] = []
        n_failed = 0
        for sample, npy_path in zip(samples, results):
            if npy_path is None:
                n_failed += 1
                continue
            updated.append({**sample, "keypoint_path": npy_path})

        logger.info(
            "Split '%s': %d succeeded, %d failed.",
            split_name, len(updated), n_failed,
        )
        updated_splits[split_name] = updated

    return updated_splits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe keypoints from WLASL100 frame sequences."
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        required=True,
        help="Path to wlasl_splits.json produced by build_wlasl_splits().",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Root directory where .npy files will be written.",
    )
    parser.add_argument(
        "--updated_splits_json",
        type=str,
        default="data/splits/wlasl_splits_processed.json",
        help="Output path for the updated splits JSON (includes keypoint_path).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for Kaggle notebooks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    splits_path = Path(args.splits_json)
    if not splits_path.exists():
        logger.error("Splits JSON not found: %s", splits_path)
        sys.exit(1)

    with open(splits_path) as f:
        splits: Dict[str, List[dict]] = json.load(f)

    logger.info(
        "Loaded splits — train: %d, val: %d, test: %d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    updated = process_splits(
        splits=splits,
        output_dir=Path(args.output_dir),
        num_workers=args.num_workers,
    )

    out_path = Path(args.updated_splits_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2)

    logger.info("Updated splits written to %s", out_path)

    # Sanity check: load one .npy and verify shape
    for split_name, samples in updated.items():
        if samples:

            probe = np.load(samples[0]["keypoint_path"])
            print(probe.shape)
            logger.info(
                "Shape check [%s][0]: %s dtype=%s",
                split_name, probe.shape, probe.dtype,
            )
            assert probe.shape == (SEQ_LEN, FEATURE_DIM), (
                f"Expected ({SEQ_LEN}, {FEATURE_DIM}), got {probe.shape}"
            )
            break

    logger.info("Extraction complete.")


if __name__ == "__main__":
    main()