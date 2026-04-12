"""
Temporal and spatial augmentations for sign language keypoint sequences.

All functions operate on numpy arrays of shape (T, 225) or torch tensors
of shape (T, 225). Applied during Dataset.__getitem__ at training time only.
Augmentations must preserve the overall motion pattern — they can change
speed/position/mirror but must not corrupt meaningful joint relationships.
"""

import random
from typing import Optional

import numpy as np


def temporal_crop_resize(
    seq: np.ndarray,
    target_len: int,
    min_crop_ratio: float = 0.7,
) -> np.ndarray:
    """Randomly crop a contiguous temporal window, then resize to target_len.

    Simulates signers performing the sign at different speeds.

    Args:
        seq: Keypoint sequence of shape (T, D).
        target_len: Output sequence length.
        min_crop_ratio: Minimum fraction of T to keep (e.g., 0.7 = keep 70%+).

    Returns:
        Resized sequence of shape (target_len, D).
    """
    T = seq.shape[0]
    crop_len = random.randint(int(T * min_crop_ratio), T)
    start = random.randint(0, T - crop_len)
    cropped = seq[start : start + crop_len]  # (crop_len, D)

    # Linear interpolation resize along time axis
    old_indices = np.linspace(0, crop_len - 1, crop_len)
    new_indices = np.linspace(0, crop_len - 1, target_len)
    resized = np.stack(
        [np.interp(new_indices, old_indices, cropped[:, d]) for d in range(seq.shape[1])],
        axis=1,
    ).astype(np.float32)
    return resized


def spatial_jitter(
    seq: np.ndarray,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Add Gaussian noise to all joint coordinates.

    Simulates natural hand tremor and MediaPipe jitter. Applied only
    to non-zero joints (zero joints are missing detections, leave them zero).

    Args:
        seq: Keypoint sequence of shape (T, 225).
        noise_std: Standard deviation of Gaussian noise (in normalized units).

    Returns:
        Jittered sequence of same shape.
    """
    noise = np.random.normal(0, noise_std, seq.shape).astype(np.float32)
    # Only add noise where landmarks are detected (non-zero)
    mask = (seq != 0).astype(np.float32)
    return seq + noise * mask


def horizontal_flip(
    seq: np.ndarray,
    num_joints: int = 75,
    coords_per_joint: int = 3,
) -> np.ndarray:
    """Mirror the sign horizontally by flipping x-coordinates and swapping hands.

    Critical: left and right hands are swapped in the joint array, because
    a mirrored signer\'s left hand becomes the viewer\'s right hand.

    Joint layout: [left_hand (0-20), right_hand (21-41), pose (42-74)]
    After flip:   [right_hand (0-20), left_hand (21-41), pose_flipped (42-74)]

    Args:
        seq: Keypoint sequence of shape (T, 225).
        num_joints: Total number of joints.
        coords_per_joint: Coordinates per joint.

    Returns:
        Horizontally flipped sequence of same shape.
    """
    T = seq.shape[0]
    pts = seq.reshape(T, num_joints, coords_per_joint).copy()

    # Negate x-coordinate (index 0) to mirror
    pts[:, :, 0] = -pts[:, :, 0]

    # Swap left hand (0-20) ↔ right hand (21-41)
    left = pts[:, :21, :].copy()
    right = pts[:, 21:42, :].copy()
    pts[:, :21, :] = right
    pts[:, 21:42, :] = left

    return pts.reshape(T, -1).astype(np.float32)


def apply_train_augmentations(
    seq: np.ndarray,
    target_len: int,
    flip_prob: float = 0.5,
    jitter_prob: float = 0.5,
    crop_prob: float = 0.5,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Apply all training augmentations with configurable probabilities.

    This is the single entry point called by WLASLDataset.__getitem__.

    Args:
        seq: Keypoint sequence of shape (T, 225).
        target_len: Final sequence length after augmentation.
        flip_prob: Probability of applying horizontal flip.
        jitter_prob: Probability of applying spatial jitter.
        crop_prob: Probability of applying temporal crop+resize.
        noise_std: Std dev for spatial jitter.

    Returns:
        Augmented sequence of shape (target_len, 225).
    """
    if random.random() < flip_prob:
        seq = horizontal_flip(seq)

    if random.random() < jitter_prob:
        seq = spatial_jitter(seq, noise_std=noise_std)

    if random.random() < crop_prob:
        seq = temporal_crop_resize(seq, target_len)

    # Always ensure correct length
    T = seq.shape[0]
    if T != target_len:
        if T > target_len:
            start = (T - target_len) // 2
            seq = seq[start : start + target_len]
        else:
            pad = np.tile(seq[-1:], (target_len - T, 1))
            seq = np.concatenate([seq, pad], axis=0)

    return seq.astype(np.float32)

