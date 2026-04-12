
"""Vocabulary configurations for Phase 1 (ASL) and Phase 2 (ESL).

Class-to-index mappings are centralized here so dataset loaders and
evaluation scripts always agree on label encoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VocabConfig:
    """Class label vocabulary for a sign language phase.

    Attributes:
        phase: Human-readable phase identifier (e.g., 'asl_wlasl100').
        class_to_idx: Mapping from sign gloss string to integer class index.
    """
    phase: str
    class_to_idx: Dict[str, int] = field(default_factory=dict)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        """Inverted mapping from index to gloss string."""
        return {v: k for k, v in self.class_to_idx.items()}

    @property
    def num_classes(self) -> int:
        """Number of classes in this vocabulary."""
        return len(self.class_to_idx)

    @classmethod
    def from_class_list(cls, phase: str, classes: List[str]) -> "VocabConfig":
        """Build a VocabConfig from an ordered list of gloss strings.

        Args:
            phase: Phase identifier string.
            classes: Ordered list of gloss strings; index = position in list.

        Returns:
            VocabConfig with class_to_idx populated.
        """
        return cls(phase=phase, class_to_idx={c: i for i, c in enumerate(classes)})
