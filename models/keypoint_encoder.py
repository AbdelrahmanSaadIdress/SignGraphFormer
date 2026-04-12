"""Spatial encoder: graph convolution over skeleton keypoints per frame.

Takes a single frame\'s joint coordinates (75 joints × 3 coords) and
produces a per-joint embedding (75 × hidden_dim) that is then mean-pooled
to a single frame vector (hidden_dim,). Applied independently to all 16
frames via a batched reshape in SLRModel.
"""

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def build_adjacency_matrix(num_joints: int = 75) -> torch.Tensor:
    """Build a fixed anatomical adjacency matrix for the MediaPipe Holistic skeleton.

    Joint index layout (matches MediaPipe Holistic output order):
        0-20:  Left hand (21 joints)
        21-41: Right hand (21 joints)
        42-74: Pose / body (33 joints)

    Returns:
        Normalized symmetric adjacency matrix of shape (num_joints, num_joints).
        Self-loops are included. Values are D^{-1/2} A D^{-1/2} (symmetric norm).
    """
    edges = []

    # --- Left hand (joints 0-20, MediaPipe hand topology) ---
    hand_bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # index
        (0, 9), (9, 10), (10, 11), (11, 12),      # middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # ring
        (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
        (5, 9), (9, 13), (13, 17),                # palm knuckle arch
    ]
    for (i, j) in hand_bones:
        edges.extend([(i, j), (j, i)])

    # --- Right hand (joints 21-41, same topology offset by 21) ---
    for (i, j) in hand_bones:
        edges.extend([(i + 21, j + 21), (j + 21, i + 21)])

    # --- Pose body (joints 42-74, MediaPipe Pose topology) ---
    # Key bones: spine, shoulders, elbows, wrists, hips
    # MediaPipe pose landmark indices (offset by 42 here):
    # 0=nose,1=left_eye_inner,...,11=left_shoulder,12=right_shoulder,
    # 13=left_elbow,14=right_elbow,15=left_wrist,16=right_wrist,
    # 23=left_hip,24=right_hip, etc.
    pose_bones = [
        (0, 1), (1, 2), (2, 3), (3, 7),           # left eye/ear
        (0, 4), (4, 5), (5, 6), (6, 8),           # right eye/ear
        (9, 10),                                   # mouth
        (11, 12),                                  # shoulders
        (11, 13), (13, 15),                        # left arm
        (12, 14), (14, 16),                        # right arm
        (11, 23), (12, 24), (23, 24),             # torso
        (23, 25), (25, 27), (27, 29), (29, 31),   # left leg
        (24, 26), (26, 28), (28, 30), (30, 32),   # right leg
        (15, 17), (15, 19), (15, 21),             # left hand connection
        (16, 18), (16, 20), (16, 22),             # right hand connection
    ]
    offset = 42
    for (i, j) in pose_bones:
        if i < 33 and j < 33:  # guard against out-of-range
            edges.extend([(i + offset, j + offset), (j + offset, i + offset)])

    # --- Cross-skeleton connections: wrists to hand roots ---
    # Left wrist (pose idx 15 → offset 42+15=57) to left hand root (joint 0)
    # Right wrist (pose idx 16 → offset 42+16=58) to right hand root (joint 21)
    edges.extend([(57, 0), (0, 57)])
    edges.extend([(58, 21), (21, 58)])

    # Build dense adjacency with self-loops
    A = torch.zeros(num_joints, num_joints)
    for i in range(num_joints):
        A[i, i] = 1.0  # self-loop
    for (i, j) in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1.0

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


class GraphConvLayer(nn.Module):
    """Single graph convolution layer: Z = A_norm * X * W.

    Uses a fixed (non-learned) adjacency matrix — topology is anatomical,
    not data-driven. Followed by LayerNorm and ReLU.

    Args:
        in_features: Input feature dimension per joint.
        out_features: Output feature dimension per joint.
        adjacency: Pre-computed normalized adjacency matrix (num_joints, num_joints).
    """

    def __init__(self, in_features: int, out_features: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("adjacency", adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution.

        Args:
            x: Joint features of shape (batch, num_joints, in_features).

        Returns:
            Updated joint features of shape (batch, num_joints, out_features).
        """
        out = torch.matmul(self.adjacency, x)   # aggregate neighbors first
        out = self.weight(out)                  # then project
        out = self.norm(out)
        return F.relu(out)


class KeypointEncoder(nn.Module):
    """Spatial encoder: two stacked graph convolution layers over skeleton joints.

    Processes one frame at a time (or many frames stacked into the batch
    dimension). Outputs a fixed-length embedding per frame by mean-pooling
    across the joint dimension.

    Input shape:  (batch, num_joints=75, coords=3)
    Output shape: (batch, hidden_dim=256)

    Args:
        cfg: ModelConfig instance carrying all hyperparameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        adjacency = build_adjacency_matrix(cfg.num_joints)
        # Two GCN layers: coords→hidden_dim/2→hidden_dim
        mid_dim = cfg.hidden_dim // 2
        self.gcn1 = GraphConvLayer(cfg.coords_per_joint, mid_dim, adjacency)
        self.gcn2 = GraphConvLayer(mid_dim, cfg.hidden_dim, adjacency)

        self.input_proj = nn.Linear(cfg.coords_per_joint, cfg.hidden_dim)


        logger.info(
            "KeypointEncoder: %d joints, 2 GCN layers (%d→%d→%d)",
            cfg.num_joints, cfg.coords_per_joint, mid_dim, cfg.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode per-frame joint coordinates to a frame embedding.

        Args:
            x: Joint coordinates of shape (batch, num_joints, coords_per_joint).

        Returns:
            Frame embedding of shape (batch, hidden_dim).
        """
        skip = self.input_proj(x).mean(dim=1)   # (B, hidden_dim) — direct projection

        h = self.gcn1(x)          # (B, J, hidden_dim/2)
        h = self.gcn2(h)          # (B, J, hidden_dim)

        out = h.mean(dim=1)                      # (B, hidden_dim)

        return out + skip                        # residual
