# Array/Tensor Mapping Analysis and Refactor Plan

## Problem Statement

The codebase has confusing array/tensor mappings with poor variable naming that creates a "code smell". Multiple different index systems are used throughout:

1. **Board data indices** (0-32): Raw BrainFlow data array positions
2. **Electrode channel indices array** (0-15 positions, values 1-16): Array with 16 positions (0-15) containing board data indices (1-16) that point to electrode channels
3. **Physical channel numbers** (1-16): OpenBCI hardware channel labels
4. **Montage indices**: Position within a specific montage configuration
5. **Combination indices**: EOG/EEG channel selections for predictions

## Current Issues Identified - REVISED

### Actually Good Variables (Previously Misidentified)

Located in `data_manager.py:575-581`:

```python
eeg_combo_indices_electrode_channel_mapping = []        # GOOD: Explicit about mapping
eog_combo_indices_electrode_channel_mapping = [10, 11]  # GOOD: Clear what it contains
```

**Note**: These verbose names are actually solving the problem by being explicit about which index system they use.

### REAL Most Problematic Variables - Vague Generic Names

#### Type 1: Generic `data` variables

```python
data[channel]                    # Which channel mapping? Which data array?
channel_data = data[adjusted_channel_idx]  # data_manager.py:197
selected_epoch_data = epoch_data[electrode_indices]  # What type of data now?
```

#### Type 2: Ambiguous index variables

```python
for eeg_idx, eog_idx in index_combinations:  # processor_improved.py:245,274
electrode_indices = self.montage.get_electrode_channel_indices()  # Which type of indices?
```

#### Type 3: Generic array access

```python
data[index]                     # Which index system (0-32, 0-15, 1-16, montage)?
timestamps = data[timestamp_channel]  # Board channel or electrode channel?
for channel in channels:        # Physical numbers or array indices?
```

### Problems with Vague Naming

1. **Ambiguous index systems** - Can't tell if variable refers to board (0-32), electrode (0-15), physical (1-16), montage, or combination indices
2. **Generic `data` arrays** - No indication of which transformation/filtering has been applied
3. **Context-dependent meaning** - Same variable name means different things in different functions
4. **Channel mapping bugs** - Easy to use wrong index system
5. **Maintenance difficulty** - Hard to understand data flow through pipeline

## Analysis of Different Index Systems

### 1. Board Data Indices (0-32)

- **Source**: `board_shim.get_data()` from BrainFlow
- **Range**: 0-32 (includes EEG, timestamps, accelerometer, etc.)
- **Usage**: Raw data collection
- **Files**: `data_manager.py`, board managers

### 2. Electrode Channel Indices Array (16 positions, values 1-16)

- **Source**: `board_shim.get_exg_channels()` returns `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`
- **Structure**: Array with 16 positions (indexed 0-15) containing board data indices (values 1-16)
- **Usage**: To extract electrode data from board data: `electrode_data = board_data[electrode_channel_indices]`
- **Example**: `electrode_channel_indices[0] = 1` (first electrode is at board position 1)

### 3. Physical Channel Numbers (1-16)

- **Source**: OpenBCI hardware labels
- **Range**: 1-16 (physical channel numbers)
- **Usage**: Montage definitions, user documentation
- **Examples**: Channel 11 = R-LEOG, Channel 12 = L-LEOG

### 4. Montage Indices (Variable)

- **Source**: `montage.get_electrode_channel_indices()`
- **Range**: 0-based, depends on montage size
- **Usage**: Extracting montage-specific channels

## Recommended Solutions

### Option 1: Fix Vague Generic Variables (High Impact, Low Risk)

```python
# Current problematic vague variables:
data[channel]                    # PROBLEM: Which data? Which channel mapping?
for eeg_idx, eog_idx in index_combinations:  # PROBLEM: Which index system?
selected_epoch_data = epoch_data[electrode_indices]  # PROBLEM: What data now?

# Improved with specific naming:
electrode_data[electrode_channel_idx]        # CLEAR: electrode data, electrode channel index
for eeg_electrode_idx, eog_electrode_idx in electrode_index_combinations:  # CLEAR: electrode indices
montage_epoch_data = epoch_data[montage_electrode_indices]  # CLEAR: filtered to montage channels
```

### Option 2: Structured Data Classes (Better)

```python
@dataclass
class ChannelMapping:
    physical_channel: int      # 1-16 (OpenBCI hardware number)
    array_index: int          # 0-15 (position in electrode array)
    channel_name: str         # "F3", "R-LEOG", etc.
    channel_type: str         # "EEG", "EOG"

@dataclass
class MontageIndices:
    eeg_channels: List[ChannelMapping]
    eog_channels: List[ChannelMapping]

    @property
    def eeg_data_indices(self) -> List[int]:
        return [ch.array_index for ch in self.eeg_channels]

    @property
    def eog_data_indices(self) -> List[int]:
        return [ch.array_index for ch in self.eog_channels]
```

### Option 3: Centralized Channel Manager (Best)

```python
class ChannelIndexManager:
    def __init__(self, montage):
        self.montage = montage
        self._build_mappings()

    def get_eeg_array_indices(self) -> List[int]:
        """Get 0-based array indices for EEG channels"""

    def get_eog_array_indices(self) -> List[int]:
        """Get 0-based array indices for EOG channels"""

    def physical_to_array_index(self, physical_channel: int) -> int:
        """Convert physical channel (1-16) to array index (0-15)"""

    def get_channel_info(self, array_index: int) -> ChannelInfo:
        """Get full channel information from array index"""
```

## Implementation Plan - REVISED

### Phase 1: Fix Vague Generic Variables (High Impact, Low Risk)

## Pipeline Variables to Change - Detailed List

_(Only variables in the main processing pipeline starting from main_realtime_stream.py)_

### **1. main_realtime_stream.py**

1. ✅ **COMPLETED** - **`data` → `new_raw_board_chunk`**
   - **Line:** 98 
   - **Context:** `new_raw_board_chunk = self.board_shim.get_board_data()`
   - **Issue:** Generic name doesn't specify it's raw board data with all channels
   - **Status:** ✅ Updated to `new_raw_board_chunk`

2. ✅ **COMPLETED** - **`new_data` → `raw_board_data_chunk`**  
   - **Lines:** 164, 176, 180
   - **Context:** `raw_board_data_chunk = streaming_board_manager.get_new_raw_data_chunk()`
   - **Issue:** Doesn't specify it's raw board data from stream
   - **Status:** ✅ Updated to `raw_board_data_chunk` with method `get_new_raw_data_chunk()`

### **2. received_stream_data_handler.py**

3. ✅ **COMPLETED** - **`board_data` → `raw_board_data_chunk`**
   - **Lines:** 24, 36
   - **Context:** `def process_board_data_chunk(self, raw_board_data_chunk):`
   - **Issue:** Generic parameter name doesn't specify raw vs processed
   - **Status:** ✅ Updated parameter and method name to `process_board_data_chunk()`

4. ✅ **COMPLETED** - **`filtered_data` → `sanitized_board_data_chunk`**
   - **Lines:** 35, 44, 47, 49
   - **Context:** `sanitized_board_data_chunk = sanitize_data(raw_board_data_chunk, ...)`
   - **Issue:** "Filtered" is vague - it's sanitized/deduplicated but still all channels
   - **Status:** ✅ Updated to `sanitized_board_data_chunk`

5. ✅ **COMPLETED** - **`epoch_start_idx` → `epoch_start_idx_abs`**
   - **Lines:** 56, 68, 74
   - **Context:** `can_process, reason, epoch_start_idx_abs, epoch_end_idx_abs = ...`
   - **Issue:** Doesn't specify absolute vs relative indexing
   - **Status:** ✅ Updated to `epoch_start_idx_abs`

6. ✅ **COMPLETED** - **`epoch_end_idx` → `epoch_end_idx_abs`**
   - **Lines:** 56, 70
   - **Context:** Same as above
   - **Issue:** Doesn't specify absolute vs relative indexing
   - **Status:** ✅ Updated to `epoch_end_idx_abs`

### **3. data_manager.py**

7. ✅ **COMPLETED** - **`new_data` → (REMOVED - function eliminated)**
   - **Lines:** N/A - `add_to_data_processing_buffer()` function was removed
   - **Context:** Function was eliminated in refactoring
   - **Issue:** Generic parameter name doesn't specify sanitized board data
   - **Status:** ✅ Function removed - data now goes directly to ETD buffer

8. ✅ **COMPLETED** - **`new_data` → (PARAMETER NAME UNCHANGED)**
   - **Lines:** 247
   - **Context:** `def queue_data_for_csv_write(self, new_data, is_initial=False):`
   - **Issue:** Generic parameter name doesn't specify sanitized board data  
   - **Status:** ✅ Parameter name kept generic as it's a pass-through function

9. ✅ **COMPLETED** - **`data` → `board_data_chunk`**
   - **Lines:** 170, 197
   - **Context:** `def validate_consecutive_data(self, board_data_chunk, channel_idx=None):`
   - **Issue:** Generic parameter doesn't specify board data format
   - **Status:** ✅ Updated to `board_data_chunk`

10. ✅ **COMPLETED** - **`epoch_data` → `epoch_data_all_electrode_channels_on_board`**
    - **Lines:** 550-553
    - **Context:** `epoch_data_all_electrode_channels_on_board = np.array([self.etd_buffer_manager.electrode_and_timestamp_data[i]...])`
    - **Issue:** Doesn't specify it's electrode data (no timestamps) from ETD buffer
    - **Status:** ✅ Updated to `epoch_data_all_electrode_channels_on_board` - very descriptive!

### **4. pyqt_visualizer.py**

11. ✅ **COMPLETED** - **`data` → `channel_data`**
    - **Lines:** 293, 308, 311
    - **Context:** `def apply_bandpass_filter(self, channel_data, low_freq, high_freq, sampling_rate):`
    - **Issue:** Generic parameter doesn't specify single channel electrode data
    - **Status:** ✅ Updated to `channel_data`

12. ✅ **COMPLETED** - **`epoch_data` → `epoch_data_all_electrode_channels_on_board`**
    - **Lines:** 359, 362, 390, 394
    - **Context:** `def plot_polysomnograph(self, epoch_data_all_electrode_channels_on_board, sampling_rate, ...):`
    - **Issue:** Doesn't specify it contains electrode channels from buffer
    - **Status:** ✅ Updated to `epoch_data_all_electrode_channels_on_board` - extremely descriptive!

13. ✅ **COMPLETED** - **`selected_epoch_data` → `montage_electrode_data`**
    - **Lines:** 394, 400, 402
    - **Context:** `montage_electrode_data = epoch_data_all_electrode_channels_on_board[electrode_indices]`
    - **Issue:** Doesn't specify it's montage-selected electrode channels
    - **Status:** ✅ Updated to `montage_electrode_data`

14. ✅ **COMPLETED** - **`display_data` → `filtered_display_montage_electrode_data`**
    - **Lines:** 400, 402, 405
    - **Context:** `filtered_display_montage_electrode_data = self.apply_complete_filtering(montage_electrode_data, ...)`
    - **Issue:** Doesn't specify it's filtered montage data ready for visualization
    - **Status:** ✅ Updated to `filtered_display_montage_electrode_data` - very descriptive!

### **5. data_filtering_utils.py**

15. ✅ **COMPLETED** - **`board_data` → `raw_board_data`**
    - **Lines:** 19, 21
    - **Context:** `def sanitize_data(raw_board_data, board_timestamp_channel, ...):`
    - **Issue:** Generic parameter doesn't specify raw board data
    - **Status:** ✅ Updated to `raw_board_data`

16. ✅ **COMPLETED** - **`filtered_data` → `sanitized_board_data`**
    - **Lines:** 21, 35, 37, 44, 54, 61, 78
    - **Context:** Processing stages in sanitize_data function
    - **Issue:** Doesn't specify it's progressively sanitized board data
    - **Status:** ✅ Updated to `sanitized_board_data`

### **Variables to KEEP (Good verbose names)**

- ✅ `eeg_combo_indices_electrode_channel_mapping` - KEEP as is
- ✅ `eog_combo_indices_electrode_channel_mapping` - KEEP as is

### Phase 2: Structural Improvements (Medium Risk)

1. **Create ChannelMapping data classes**
2. **Refactor montage.py** to return structured data
3. **Update signal processor** to use clearer interfaces

### Phase 3: Centralized Management (Higher Risk)

1. **Implement ChannelIndexManager**
2. **Refactor all channel index usage** through manager
3. **Add comprehensive validation**

## Files Requiring Changes - REVISED

### Primary Files (Generic Variable Problems)

- `gssc_local/realtime_with_restart/data_manager.py` - Generic `data`, `channel_data` variables
- `gssc_local/realtime_with_restart/pyqt_visualizer.py` - `selected_epoch_data`, generic `data` arrays
- Signal processor files - `eeg_idx`, `eog_idx`, `index_combinations`
- Loop variables throughout codebase - Generic `data`, `index`, `channel`

### Files That Are Actually Good (Keep As-Is)

- `gssc_local/realtime_with_restart/data_manager.py:575-581` - The verbose variable names are GOOD
- Any files using explicit naming like `electrode_channel_indices`

### Documentation Updates

- Update comments to match new variable names
- Add index mapping documentation
- Update any README sections about channel mapping

## Best Practices Moving Forward - REVISED

1. **Specific data array naming**:

   - `raw_board_data` - Direct from BrainFlow (shape: channels x samples)
   - `electrode_data` - Extracted electrode channels only
   - `montage_data` - Data filtered to montage channels
   - `filtered_epoch_data` - Processed epoch data

2. **Specific index naming**:

   - `board_channel_indices` - Indices into raw board data (0-32)
   - `electrode_channel_indices` - Physical electrode indices (0-15)
   - `montage_channel_indices` - Positions in montage (0-N)
   - `eeg_electrode_indices` - Electrode indices for EEG channels
   - `eog_electrode_indices` - Electrode indices for EOG channels

3. **When verbose is good**: For complex mappings, verbose names like `eeg_combo_indices_electrode_channel_mapping` are actually helpful
4. **Always include type hints** showing index ranges
5. **Document transformations** when data arrays change meaning
6. **Use validation** to catch index mismatches

## Questions for Discussion - REVISED

1. Should we focus on Phase 1 (fixing vague generic variables) as the highest impact change?
2. Are there other vague/generic variable names causing confusion that I missed?
3. Do you agree the verbose names like `eeg_combo_indices_electrode_channel_mapping` should be kept as-is?
4. Should we prioritize data_manager.py first since it's the core data pipeline?
5. Do you want to add type hints to clarify index ranges during the refactor?

## Risk Assessment - REVISED

- **Phase 1 (Fix vague variables)**: Low risk, high impact - Makes code immediately more understandable
- **Phase 2**: Medium risk, requires careful testing of data flow
- **Phase 3**: Higher risk but provides long-term maintainability

The core issue is **vague generic variable names** that don't specify which of the 5 index systems they use. The verbose variables are actually good because they're explicit. The underlying logic seems sound but is obscured by ambiguous naming of generic variables like `data[index]` and `eeg_idx`.
