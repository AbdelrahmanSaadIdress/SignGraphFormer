"""Abstract base class for all sign language datasets."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class BaseSignDataset(Dataset, ABC):
    """Abstract base for sign language keypoint datasets.

    Every concrete dataset must implement __len__ and __getitem__,
    and return tensors of the agreed shapes so DataLoaders can be
    swapped between Phase 1 and Phase 2 without modifying training code.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in the split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return (keypoint_sequence, class_index) for sample idx.

        Returns:
            keypoint_sequence: Float32 tensor of shape (seq_len, input_dim).
            class_index: Integer class label.
        """
        ...

    @abstractmethod
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted sampling or loss.

        Returns:
            Float32 tensor of shape (num_classes,).
        """
        ...