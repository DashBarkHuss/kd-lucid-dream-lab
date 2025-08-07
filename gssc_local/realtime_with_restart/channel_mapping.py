from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ChannelIndexMapping:
    """Primary key for tracking channels through all transformations"""
    board_position: int  # 0-32 (position in raw BrainFlow board data array) - PRIMARY KEY

    def validate(self):
        """Validate board position is in valid range"""
        if not (0 <= self.board_position <= 32):
            raise ValueError(f"Invalid board_position: {self.board_position}. Must be 0-32.")


@dataclass
class DataWithBrainFlowDataKey:
    """General wrapper for any data that tracks BrainFlow board data indices as primary keys"""
    data: np.ndarray
    channel_mapping: List[ChannelIndexMapping]  # Maps array positions to BrainFlow data indices

    def _get_indices_by_brainflow_keys(self, brainflow_keys: List[int]) -> List[int]:
        """Find array indices for given BrainFlow keys

        Note: Private for now. Consider making public if complex use cases emerge
        that need direct access to array indices for custom operations.
        """
        result = []
        for brainflow_key in brainflow_keys:
            for array_idx, mapping in enumerate(self.channel_mapping):
                if mapping.board_position == brainflow_key:
                    result.append(array_idx)
                    break
            else:
                raise ValueError(f"BrainFlow key {brainflow_key} not found in channel mapping")
        return result

    def get_by_key(self, brainflow_key: int) -> np.ndarray:
        """Get single data row by BrainFlow data key"""
        array_idx = self._get_indices_by_brainflow_keys([brainflow_key])[0]
        return self.data[array_idx]

    def get_by_keys(self, brainflow_keys: List[int]) -> np.ndarray:
        """Get multiple data rows by BrainFlow data keys"""
        array_indices = self._get_indices_by_brainflow_keys(brainflow_keys)
        return self.data[array_indices]

    def __getitem__(self, key):
        """Allow array-like access to underlying data"""
        return self.data[key]

    @property
    def shape(self):
        """Expose shape of underlying data"""
        return self.data.shape