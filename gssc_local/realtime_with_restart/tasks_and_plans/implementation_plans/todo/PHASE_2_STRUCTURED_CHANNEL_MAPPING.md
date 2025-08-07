# Phase 2 Implementation Plan: Structured Channel Mapping

## Overview

Phase 1 (generic variable renaming) is complete. Phase 2 focuses on replacing hardcoded channel indices with structured data classes and centralized channel mapping logic.

## Current State Analysis ✅

Based on analysis of the codebase:

1. **Phase 1 Complete** - All generic variable names have been improved with descriptive names
2. **Montage.py Well-Structured** - Already has `ChannelConfig` dataclass and good interfaces
3. **Current Pain Points Identified:**
   - **Hardcoded electrode indices** in `data_manager.py:577-583`:
     ```python
     eeg_combo_indices_electrode_channel_mapping = [0, 1, 2]  # Hardcoded
     eog_combo_indices_electrode_channel_mapping = [10]       # Hardcoded
     ```
   - **Manual index mapping logic** scattered across files
   - **No centralized validation** of channel combinations between montages and processing

## Primary Goal

**Create consistent tracking of index transformations through structured data classes**

The main issue is that we have 5 different index systems with manual/implicit conversions between them:

1. **Board data indices** (0-32): Raw BrainFlow data array positions
2. **Electrode channel indices** (0-15): Array positions containing electrode data
3. **Physical channel numbers** (1-16): OpenBCI hardware channel labels
4. **Montage indices** (0-N): Position within a specific montage configuration
5. **Processing combinations**: EEG/EOG channel selections for predictions

Currently these transformations happen manually throughout the code with hardcoded arrays and implicit conversions. Phase 2 will create structured data classes that explicitly track these transformations.

## Implementation Plan

### 1. Create Simple Channel Identifier Class

**NEW FILE**: `/gssc_local/realtime_with_restart/channel_mapping.py`

```python
from dataclasses import dataclass

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
```

This simple data class replaces raw integers with a structured identifier that makes it explicit we're referring to board data positions.

### Example of how to use the DataWithBrainFlowDataKey class:

**UPDATE**: `/gssc_local/realtime_with_restart/data_manager.py`

**Current problematic code (lines 570-590)**:

```python
# CURRENT: Hardcoded based on montage type detection
if all(ch_type == "EOG" for ch_type in channel_types):
    eeg_combo_indices_electrode_channel_mapping = []        # Hardcoded
    eog_combo_indices_electrode_channel_mapping = [10, 11]  # Hardcoded
else:
    eeg_combo_indices_electrode_channel_mapping = [0, 1, 2]  # Hardcoded
    eog_combo_indices_electrode_channel_mapping = [10]       # Hardcoded
```

**Proposed solution**:

```python
from gssc_local.realtime_with_restart.channel_mapping import ChannelIndexMapping

# Create mapping for all electrode channels (board positions 1-16)
epoch_data_all_electrode_channels_on_board_mapping = [
    ChannelIndexMapping(board_position=1),   # Array index 0 -> Board position 1
    ChannelIndexMapping(board_position=2),   # Array index 1 -> Board position 2
    ChannelIndexMapping(board_position=3),   # Array index 2 -> Board position 3
    ChannelIndexMapping(board_position=4),   # Array index 3 -> Board position 4
    ChannelIndexMapping(board_position=5),   # Array index 4 -> Board position 5
    ChannelIndexMapping(board_position=6),   # Array index 5 -> Board position 6
    ChannelIndexMapping(board_position=7),   # Array index 6 -> Board position 7
    ChannelIndexMapping(board_position=8),   # Array index 7 -> Board position 8
    ChannelIndexMapping(board_position=9),   # Array index 8 -> Board position 9
    ChannelIndexMapping(board_position=10),  # Array index 9 -> Board position 10
    ChannelIndexMapping(board_position=11),  # Array index 10 -> Board position 11
    ChannelIndexMapping(board_position=12),  # Array index 11 -> Board position 12
    ChannelIndexMapping(board_position=13),  # Array index 12 -> Board position 13
    ChannelIndexMapping(board_position=14),  # Array index 13 -> Board position 14
    ChannelIndexMapping(board_position=15),  # Array index 14 -> Board position 15
    ChannelIndexMapping(board_position=16),  # Array index 15 -> Board position 16
]

# Create epoch data
epoch_data = np.array([
    self.etd_buffer_manager.electrode_and_timestamp_data[i][start_idx_rel:end_idx_rel]
    for i in range(num_electrode_channels)
])

# Wrap data with BrainFlow index mapping
epoch_data_key_wrapper = DataWithBrainFlowDataIndex(
    data=epoch_data,
    channel_mapping=epoch_data_all_electrode_channels_on_board_mapping
)

# Select data by BrainFlow data key (clean and simple!)
# Single channel
f3_data = epoch_data_key_wrapper.get_by_key(1)  # Just F3 channel

# Multiple channels
desired_brainflow_keys = [2, 3, 4]  # BrainFlow keys we want (F4, C3, C4)
selected_epoch_data = epoch_data_key_wrapper.get_by_keys(desired_brainflow_keys)



```

**Search and replace** other locations in the codebase that use raw integer arrays for channel identification and replace them with `ChannelIndexMapping` lists.

## Success Criteria

1. **All channel arrays consistently use board_positions using the DataWithBrainFlowDataKey** (0-32 range) instead of mixed indexing systems
2. **Simpler variable naming** - We'll be able to remove the verbose "mapping" suffixes since all arrays are now using the DataWithBrainFlowDataKey for tracking primary keys
3. **Signal processing results identical** to pre-refactor behavior
4. **Foundation for Phase 3** - Consistent indexing system ready for centralized management

## Comprehensive List of Arrays and Index Variables to Update

Based on codebase analysis, here are all arrays and index variables that need to be converted to structured channel mapping:

### Primary Hardcoded Channel Arrays (Critical Priority)

**File**: `data_manager.py` (lines 577-583)

```python
# CURRENT - These are the main targets for Phase 2:
eeg_combo_indices_electrode_channel_mapping = []        # Line 577 (EOG-only montage)
eog_combo_indices_electrode_channel_mapping = [10, 11]  # Line 578 (EOG-only montage)
eeg_combo_indices_electrode_channel_mapping = [0, 1, 2] # Line 582 (normal montage)
eog_combo_indices_electrode_channel_mapping = [10]      # Line 583 (normal montage)
```

**File**: `processor.py` (line 95)

```python
# CURRENT - Default parameter values:
def predict_sleep_stage(self, epoch_data, hidden_states, eeg_indices = [0, 1, 2], eog_indices = [3]):
```

### Buffer Management Arrays

**File**: `data_manager.py`

```python
# Line 76 - Buffer count array (fixed size, may need ChannelIndexMapping for validation)
self.epochs_processed_count_per_buffer = [0] * 6   # Count of epochs processed per buffer

# Line 743 - Reset logic for buffer counts
self.epochs_processed_count_per_buffer = [0] * 6
```

### Data Access Index Variables

**Multiple files** - Variables that access specific array positions/channels:

```python
# Shape and dimension access that may need ChannelIndexMapping context
board_data_chunk.shape[0]  # Number of channels dimension
board_data_chunk.shape[1]  # Number of samples dimension
epoch_data.shape[0]        # Number of channels in epoch
epoch_data.shape[1]        # Number of samples in epoch

# Timestamp channel access (these use board position indices)
new_data[board_timestamp_channel][0]     # First timestamp
timestamps[0]                            # First timestamp
timestamps[-1]                          # Last timestamp

# Channel data access
channel_data[0]                         # First sample in channel
data.data[array_indices]                # Multi-channel access by indices
```

### Channel Selection and Processing Arrays

**Files**: Various processing files

```python
# Range-based channel selections that may need ChannelIndexMapping
range(num_electrode_channels)           # Iterating over all electrode channels
range(epoch_data.shape[0])             # Iterating over channels in epoch data

# Specific channel index access
epoch_data_all_electrode_channels_on_board[0]   # Physical channel 1 (F3)
epoch_data_all_electrode_channels_on_board[10]  # Physical channel 11 (R-LEOG)
epoch_data_all_electrode_channels_on_board[11]  # Physical channel 12 (L-LEOG)
```

### Array Construction Patterns

**Pattern**: Array index to channel mapping

```python
# CURRENT - Manual array construction with implicit channel mapping
epoch_data = np.array([
    self.etd_buffer_manager.electrode_and_timestamp_data[i][start_idx_rel:end_idx_rel]
    for i in range(num_electrode_channels)  # This 'i' becomes array index, maps to board channel i+1
])
```

### ETD Buffer Manager Channel References

**File**: `etd_buffer_manager.py`

```python
# Channel counting and validation
len(self.electrode_and_timestamp_data[0])           # Buffer size check
board_data_chunk.shape[0]                          # Input channel count
len(self.electrode_and_timestamp_channels)         # Expected channel count
```

## Variables That Do NOT Need ChannelIndexMapping

Not all arrays need ChannelIndexMapping. Only arrays that are used to access the streamed data from the board need ChannelIndexMapping.

**These arrays work with already-processed data and don't need channel mapping:**

- CSV formatting arrays (`fmt`, `data_array.shape[1]`)
- Signal processing results (`results[i][0].numpy()`)
- Validation arrays (`duplicate_indices`, `unique_indices`)
- Model state arrays (`new_hidden_states`)
- Time-based calculations (`total_duration`, `expected_samples`)

## Future Cleanup: Variable Name Simplification

**Note for Phase 2 completion**: Once all arrays are updated to use `ChannelIndexMapping`, we can simplify variable names by removing "mapping" suffix since the structured approach will be implicit:

- `epoch_data_electrode_channel_mapping` → `epoch_data`
- `eeg_combo_indices_electrode_channel_mapping` → `eeg_combo_indices`
- `eog_combo_indices_electrode_channel_mapping` → `eog_combo_indices`
- `epoch_data_all_electrodes_on_board_mapping` → `epoch_data`
