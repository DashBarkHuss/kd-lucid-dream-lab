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
class NumPyDataWithBrainFlowDataKey:
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

    def set_by_key(self, brainflow_key: int, new_data: np.ndarray):
        """Set data for a specific BrainFlow data key"""
        array_idx = self._get_indices_by_brainflow_keys([brainflow_key])[0]
        self.data[array_idx] = new_data

    def __getitem__(self, key):
        """Allow array-like access to underlying data"""
        return self.data[key]

    @property
    def shape(self):
        """Expose shape of underlying data"""
        return self.data.shape


@dataclass
class ListDataWithBrainFlowDataKey:
    """Wrapper for list-of-lists data with BrainFlow key tracking"""
    data: List[List]  # [channel][time] format
    channel_mapping: List[ChannelIndexMapping]

    def get_by_key(self, brainflow_key: int) -> List:
        """Get single channel list by BrainFlow data key"""
        for array_idx, mapping in enumerate(self.channel_mapping):
            if mapping.board_position == brainflow_key:
                return self.data[array_idx]
        raise ValueError(f"BrainFlow key {brainflow_key} not found in channel mapping")

    def get_by_keys(self, brainflow_keys: List[int]) -> List[List]:
        """Get multiple channel lists by BrainFlow data keys"""
        return [self.get_by_key(key) for key in brainflow_keys]

    def set_by_key(self, brainflow_key: int, new_data: List):
        """Set data for a specific BrainFlow data key"""
        for array_idx, mapping in enumerate(self.channel_mapping):
            if mapping.board_position == brainflow_key:
                self.data[array_idx] = new_data
                return
        raise ValueError(f"BrainFlow key {brainflow_key} not found in channel mapping")

    def extend_by_key(self, brainflow_key: int, new_data: List):
        """Extend data for a specific BrainFlow data key"""
        for array_idx, mapping in enumerate(self.channel_mapping):
            if mapping.board_position == brainflow_key:
                self.data[array_idx].extend(new_data)
                return
        raise ValueError(f"BrainFlow key {brainflow_key} not found in channel mapping")

    def __getitem__(self, key):
        """Allow array-like access to underlying data"""
        return self.data[key]

    def __len__(self):
        """Return number of channels"""
        return len(self.data)
    
    def clear_all(self):
        """Clear all channel data"""
        for channel_list in self.data:
            channel_list.clear()


@dataclass
class RawBoardDataWithKeys:
    """Wrapper for raw BrainFlow board data where array index = board position
    
    This class makes explicit that we're working with raw board data where
    the array index directly corresponds to the board position (0-32).
    This eliminates ambiguity about whether data uses board positions or array indices.
    """
    data: np.ndarray
    
    def get_by_key(self, board_position: int) -> np.ndarray:
        """Get data by board position key (board position = array index for raw data)
        
        Args:
            board_position: Board position (0-32) corresponding to BrainFlow channel
            
        Returns:
            np.ndarray: Data for the specified board position
        """
        if not (0 <= board_position < self.data.shape[0]):
            raise IndexError(f"Board position {board_position} out of range [0, {self.data.shape[0]-1}]")
        return self.data[board_position]
    
    def __getitem__(self, key):
        """Allow direct array access for compatibility, but prefer get_by_key()"""
        return self.data[key]
    
    @property
    def shape(self):
        """Expose shape of underlying data"""
        return self.data.shape
    
    @property
    def size(self):
        """Expose size of underlying data for numpy compatibility"""
        return self.data.size


def create_numpy_data_with_brainflow_keys(data: np.ndarray, board_channels: List[int]) -> NumPyDataWithBrainFlowDataKey:
    """Create NumPyDataWithBrainFlowDataKey with proper channel mapping.
    
    Args:
        data: Raw data array (channels, samples) 
        board_channels: List of BrainFlow board positions for each channel
        
    Returns:
        NumPyDataWithBrainFlowDataKey: Structured data with channel mapping
    """
    channel_mapping = [
        ChannelIndexMapping(board_position=board_channel)
        for board_channel in board_channels
    ]
    
    return NumPyDataWithBrainFlowDataKey(
        data=data,
        channel_mapping=channel_mapping
    )