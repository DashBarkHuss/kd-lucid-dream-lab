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

✅ completed data class creation

- ChannelIndexMapping
- DataWithBrainFlowDataKey

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

✅ completed epoch_data_all_electrode_channels_on_board_mapping update

**File**: `processor_improved.py` (lines 71-73, 75-77)

```python
# CURRENT - Raw array indexing in make_combo_dictionary method:
if eeg_index is not None:
    eeg_data = epoch_tensor[eeg_index].reshape(1, 1, -1)
    input_dict['eeg'] = eeg_data

if eog_index is not None:
    eog_data = epoch_tensor[eog_index].reshape(1, 1, -1)
    input_dict['eog'] = eog_data
```

✅ completed

**File**: `processor_improved.py` (line 269)

```python
# CURRENT - Extracts raw data from wrapper instead of using key-based access:
input_dict_list = self.prepare_input_data(epoch_data_wrapper.data, index_combinations)
```

✅ completed

**File**: `processor.py` (line 95)

```python
# CURRENT - Default parameter values:
def predict_sleep_stage(self, epoch_data, hidden_states, eeg_indices = [0, 1, 2], eog_indices = [3]):
```

✅ completed

### replace get_electrode_channel_indices with get_board_keys

on montage.py we need to add a function that gets the board keys for all channels in the montage
✅ completed

### Buffer Management Arrays

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

✅ completed - board_data_chunk we kept the same since it is the raw board positions. channel_data is just one channel. new data is the raw board data.

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

Nothing to to hear anymore.

## montage

    electrode_indices = self.montage.get_electrode_channel_indices()
        montage_electrode_data = epoch_data[electrode_indices]

        we need to add a function to the montage that gets the montage indices by board position

✅ completed

### Array Construction Patterns

**Pattern**: Array index to channel mapping

```python
# CURRENT - Manual array construction with implicit channel mapping
epoch_data = np.array([
    self.etd_buffer_manager.electrode_and_timestamp_data[i][start_idx_rel:end_idx_rel]
    for i in range(num_electrode_channels)  # This 'i' becomes array index, maps to board channel i+1
])
```

✅ completed

we need to change edt to have keys then we can reference them using the mapping we made.

✅ completed

### ETD Buffer Manager Channel References

**File**: `etd_buffer_manager.py`

#### **Core Data Structure Updates (Critical Priority)**

**1. Main Buffer Structure - `electrode_and_timestamp_data`**
**Lines 74, 82, 90, 135, 204, 281**

Currently: `self.electrode_and_timestamp_data = [[] for _ in range(channel_count)]`

**Needs**: Wrapper with `DataWithBrainFlowDataKey` to track board position mapping:

✅ completed

```python
# Current access pattern
self.electrode_and_timestamp_data[i]  # Line 281, 204, etc.

# Should become
self.electrode_and_timestamp_data.get_by_key(board_position)
```

✅ completed

**2. Channel Selection Logic - `electrode_and_timestamp_channels`**
**Lines 60, 67, 73, 272, 276, 280**

Currently: `List[int]` of channel indices
**Needs**: `List[ChannelIndexMapping]` with board positions

✅ completed original code was correct.

```python
# Current (Line 280)
for i, channel in enumerate(self.electrode_and_timestamp_channels):
    self.electrode_and_timestamp_data[i].extend(board_data_chunk[channel].tolist())

# Should use board position keys
for i, channel_mapping in enumerate(self.electrode_and_timestamp_channels):
    board_position = channel_mapping.board_position
    self.electrode_and_timestamp_data[i].extend(board_data_chunk[board_position].tolist())
```

✅ completed with the extend_by_key method.

**3. Board Data Access Pattern**
**Line 281**

```python
# Current
board_data_chunk[channel]

# Should become (if board_data_chunk gets wrapped)
board_data_chunk.get_by_key(channel)
```

⭐️ skipped for nowsince board_data_chunk is the raw board data.

**4. Timestamp Channel Index**
**Lines 72, 90**

Currently: `self.timestamp_channel_index: int`
**Needs**: `ChannelIndexMapping` with board position

```python
# Current (Line 90)
return self.electrode_and_timestamp_data[self.timestamp_channel_index]

# Should become
return self.electrode_and_timestamp_data.get_by_key(self.timestamp_channel_mapping.board_position)

```

✅ completed

**5. Channel Count and Validation**
**Lines 74, 135, 272, 276**

These remain as integers but validation logic needs updating:

```python
# Current (Line 272)
if board_data_chunk.shape[0] < len(self.electrode_and_timestamp_channels):

# Should become
if board_data_chunk.shape[0] < len(self.electrode_and_timestamp_channels):
    # But with better error message referencing board positions
```

✅ completed

#### **Method Updates Required**

**6. Constructor Updates**
**Lines 60, 67, 72-74**

```python
def __init__(self, max_buffer_size: int, timestamp_channel_mapping: ChannelIndexMapping,
             channel_count: int, electrode_and_timestamp_channel_mappings: List[ChannelIndexMapping]):
```

✅ completed but different from the original code.

**7. Data Addition Method**
**Lines 260-285**

The `select_channel_data_and_add()` method needs complete refactor to use key-based access patterns.

✅ completed
s
**8. Buffer Access Methods**
**Lines 82, 90**

All methods that access `electrode_and_timestamp_data` need to use `get_by_key()` instead of array indexing.

✅ completed

#### **Variables That Do NOT Need Updates in ETD Buffer Manager**

These work with processed data/metadata and should remain unchanged:

- `max_buffer_size`, `offset`, `total_streamed_samples` (Lines 69-71)
- `current_size`, `points_removed`, `safe_remove_points` (Lines 183, 160, 195)
- Shape access: `board_data_chunk.shape[0]`, `board_data_chunk.shape[1]` (Lines 272, 276)
- List operations: `len()`, `range()`, slicing operations
- Validation variables: `expected_size`, `actual_size`, `channel_lengths`

```python
# Other channel counting and validation that remain unchanged
len(self.electrode_and_timestamp_data[0])           # Buffer size check (if not wrapped)
board_data_chunk.shape[0]                          # Input channel count
len(self.electrode_and_timestamp_channels)         # Expected channel count
```

**File: main_speed_controlled_stream.py**

- Line 116: `new_data[board_timestamp_channel][0]` → `new_data.get_by_key(board_timestamp_channel)[0]`
- Line 124: `new_data[board_timestamp_channel][-1]` → `new_data.get_by_key(board_timestamp_channel)[-1]`

skip - new data is the raw board data.

**File: main_realtime_stream.py**

- Line 176: `raw_board_data_chunk[board_timestamp_channel][0]` → `raw_board_data_chunk.get_by_key(board_timestamp_channel)[0]`

skip - raw board data chunk is the raw board data.

**File: data_manager.py**

- Line 198: `board_data_chunk[adjusted_channel_idx]` → `board_data_chunk.get_by_key(adjusted_channel_idx)`

skip - board data chunk is the raw board data.

**File: etd_buffer_manager.py**

- Line 281: `board_data_chunk[channel]` → `board_data_chunk.get_by_key(channel)`

skip - board data chunk is the raw board data.

**File: utils/data_filtering_utils.py**

- Line 61: `sanitized_board_data[board_timestamp_channel, :]` → `sanitized_board_data.get_by_key(board_timestamp_channel)`

skip - sanitized board data is the raw board data.

**File: core/brainflow_child_process_manager.py**

- Line 92: `board_data[timestamp_channel]` → `board_data.get_by_key(timestamp_channel)`

## Variables That Do NOT Need ChannelIndexMapping

Not all arrays need ChannelIndexMapping. Only arrays that are used to access the streamed data from the board need ChannelIndexMapping.

**These arrays work with already-processed data and don't need channel mapping:**

- CSV formatting arrays (`fmt`, `data_array.shape[1]`)
- Signal processing results (`results[i][0].numpy()`)
- Validation arrays (`duplicate_indices`, `unique_indices`)
- Model state arrays (`new_hidden_states`)
- Time-based calculations (`total_duration`, `expected_samples`)

## Specific Array Access Patterns to Convert to `.get_by_key()`

Based on codebase analysis, here are all the specific array access patterns that need to be converted to use `DataWithBrainFlowDataKey.get_by_key()`:

### Arrays That Do NOT Need Conversion:

These access shape properties or work with already-processed data and should remain as regular array access:

- `board_data_chunk.shape[0]`, `board_data_chunk.shape[1]` - Shape properties
- `new_data[0]` when checking sample count - Length checking
- Array construction loops like `range(num_electrode_channels)`
- CSV formatting arrays (`fmt`, `data_array.shape[1]`)
- Signal processing results (`results[i][0].numpy()`)
- Model state arrays (`new_hidden_states`)

## Data Manager Specific Updates (Comprehensive Analysis)

**Required Changes:**

- Replace hardcoded arrays with board position keys
- Use `epoch_data_key_wrapper.get_by_keys([board_positions])` instead of array indices

✅ completed - was correct but i was using the wrong variable name

### **Epoch Data Construction (Lines 550-553)**

```python
# CURRENT (Lines 550-553):
epoch_data = np.array([
    self.etd_buffer_manager.electrode_and_timestamp_data[i][start_idx_rel:end_idx_rel] # Line 551 - NEEDS KEY ACCESS
    for i in range(num_electrode_channels)  # Line 552 - NEEDS BOARD KEY MAPPING
])
```

⏳ this is mostly an etd buffer manager issue. need to update the etd buffer manager to use the new mapping.

✅ completed

**Required Changes:**

- Add structured channel mapping to `etd_buffer_manager.electrode_and_timestamp_data`
- Use board position keys instead of sequential array indices

✅ completed

### **ETD Buffer Manager Integration**

The `etd_buffer_manager.electrode_and_timestamp_data` needs to be wrapped with `DataWithBrainFlowDataKey` so that:

```python
# CURRENT: Raw array access
self.etd_buffer_manager.electrode_and_timestamp_data[i][start_idx_rel:end_idx_rel]

# AFTER: Key-based access
self.etd_buffer_manager.electrode_and_timestamp_data.get_by_key(board_position)[start_idx_rel:end_idx_rel]
```

⏳ need to update the etd buffer manager to use the new mapping.

✅ completed

### **Validation Method Updates (Line 197)**

```python
# CURRENT (Line 197):
channel_data = board_data_chunk[adjusted_channel_idx]  # NEEDS KEY ACCESS

# AFTER:
channel_data = board_data_chunk.get_by_key(adjusted_channel_idx)
```

✅ completed
actually not necessary since board ata isn't transofrmed and the pk is the index

### **Signal Processor Integration (Lines 603-607)**

```python
# CURRENT (Lines 600, 604):
index_combinations = self.signal_processor.get_index_combinations(eeg_combo_indices_electrode_channel_mapping, eog_combo_indices_electrode_channel_mapping)
predicted_class, class_probs, new_hidden_states = self.signal_processor.predict_sleep_stage(
    epoch_data_key_wrapper,  # Already wrapped correctly
    index_combinations,
    self.buffer_hidden_states[buffer_id]
)
```

**Required Changes:**

- Update `get_index_combinations()` to work with board position keys
- Ensure `SignalProcessor.predict_sleep_stage()` uses key-based access internally - i wonder how thoguh since we pass in the wrong mapping- eeg_combo_indices_electrode_channel_mapping

✅ completed

### **Variables That Do NOT Need Updates**

These work with processed data or metadata and should remain unchanged:

- `buffer_id`, `epoch_start_idx_abs`, `epoch_end_idx_abs` (indices for buffer management)
- `start_idx_rel`, `end_idx_rel` (relative buffer indices)
- `timestamp_data` (already processed timestamps)
- `self.points_per_epoch`, `self.sampling_rate` (configuration constants)
- `predicted_class`, `class_probs`, `new_hidden_states` (model outputs)
- Shape access: `epoch_data.shape[1]` (remains as property access) ⭐️shouldn't this be epoch_data_wrapper? no maybe not because this might be before we add it to the wrapper

### **Montage Integration Required**

The `montage` object needs a new method:

```python
# NEW METHOD NEEDED in montage.py:
def get_board_keys(self) -> List[int]:
    """Return board position keys for all channels in this montage"""
```

This will replace the hardcoded channel arrays with montage-driven board position selections.

### **Summary of Primary Updates Needed:**

1. **Lines 589-590, 594-595**: Replace hardcoded `eeg_combo_indices` and `eog_combo_indices` arrays with board position keys
2. **Lines 550-553**: Convert epoch data construction to use key-based access
3. **Line 197**: Update validation method to use `get_by_key()`
4. **ETD Buffer Manager**: Wrap `electrode_and_timestamp_data` with `DataWithBrainFlowDataKey`
5. **Montage Integration**: Add `get_board_keys()` method to montage class

The core pattern is replacing `array[index]` with `data_wrapper.get_by_key(board_position)` for all arrays that access raw board data.

## Future Cleanup: Variable Name Simplification

**Note for Phase 2 completion**: Once all arrays are updated to use `ChannelIndexMapping`, we can simplify variable names by removing "mapping" suffix since the structured approach will be implicit:

- `epoch_data_electrode_channel_mapping` → `epoch_data`
- `eeg_combo_indices_electrode_channel_mapping` → `eeg_combo_indices`
- `eog_combo_indices_electrode_channel_mapping` → `eog_combo_indices`
- `epoch_data_all_electrodes_on_board_mapping` → `epoch_data`
