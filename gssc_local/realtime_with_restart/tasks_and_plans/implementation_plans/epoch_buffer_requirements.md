# EEG Epoch Processing Buffer Requirements

## Overview

The EEG epoch processing buffer (`electrode_and_timestamp_data`) needs memory management to prevent unbounded growth. Currently, the buffer is never cleared during regular operation.

## Core Requirements

### Data Structure

- Store correct channels in right order (EEG + timestamp)
- Maintain data point order and channel synchronization
- Handle floating point values for EEG and timestamps
- Validate NaN and infinite values

### Access Patterns

- Support 30-second epoch extraction
- Quick access to latest timestamp
- Track buffer status and processed epochs
- Maintain alignment during slicing

### Memory Management

- Minimum buffer size: (30s + 25s + padding) × sampling_rate points
- Can discard processed data
- Must maintain data continuity for gap detection
- Must track absolute positions in total streamed data

### Buffer Overlap

The round-robin system processes overlapping epochs:

- Each buffer processes a 30-second epoch
- Buffers are offset by 5 seconds (seconds_per_step)
- This creates significant overlap between buffers:
  - Buffer 0: 0-30s
  - Buffer 1: 5-35s
  - Buffer 2: 10-40s
  - Buffer 3: 15-45s
  - Buffer 4: 20-50s
  - Buffer 5: 25-55s

This overlap means we must keep data until the last buffer that could need it has processed its epoch. For example, when Buffer 5 is processing its epoch (25-55s), it needs data that was used by Buffer 0 (0-30s) for the overlap period 25-30s.

Therefore, we need to maintain a buffer size of 35 seconds to ensure all overlapping epochs have access to their required data. This means:

- Minimum buffer size = 35 seconds × sampling_rate points
- Can only clear data older than 35 seconds
- Must maintain data continuity for the full 35-second window

The 35-second requirement comes from:

- Latest unprocessed data: 25-55s (30 seconds)
- Plus the 5-second step size
- Total: 35 seconds

### Buffer Trimming Implementation

After each complete processing cycle (including visualization), we should call `self._trim_etd_buffer()` which will:

1. Check if buffer length > 35 seconds × sampling_rate
2. If yes, remove oldest data to bring buffer back to 35 seconds
3. Update any offset tracking to maintain absolute position references

This ensures we:

- Never keep more data than necessary
- Always have enough data for the next processing cycle
- Maintain memory efficiency
- Don't interfere with active processing

### Method Compatibility Requirements

To safely implement buffer trimming, we need to ensure compatibility with:

1. Methods using absolute indices:

   - `_process_epoch`: Needs to handle offset when accessing data
   - `validate_epoch_gaps`: Must account for trimmed data in gap detection
   - `add_sleep_stage_to_csv_buffer`: Needs correct timestamp indices
   - `_get_next_epoch_indices`: Must adjust indices based on trimmed data

2. State tracking:

   - Update `matrix_of_round_robin_processed_epoch_start_indices_abs` to account for trimmed data
   - Maintain `total_streamed_samples_since_start` for absolute position tracking
   - Add `etd_offset` to track how much data has been trimmed

3. Index translation:
   - Implement `_adjust_index_with_etd_offset` to convert between absolute and relative indices
   - Update all index-based operations to use the offset

### Minimal Test Cases

1. Buffer Trimming Tests:

   ```python
   def test_trim_etd_buffer():
       # Test trimming when buffer > 35s
       # Test no trimming when buffer < 35s
   ```

2. Index Translation Tests:

   ```python
   def test_index_adjustment():
       # Test absolute to relative conversion
       # Test relative to absolute conversion
   ```

3. Processing Continuity Tests:

   ```python
   def test_processing_after_trim():
       # Test epoch processing after trim
       # Test gap detection after trim
   ```

4. Round-Robin Integration Test:
   ```python
   def test_round_robin_with_trimming():
       # Test full round-robin cycle (buffers 0-5)
       # Verify each buffer can process its epoch after trimming
       # Check that overlap data is preserved correctly
       # Verify indices remain valid throughout the cycle
   ```

### Affected Components

1. **Core Methods**

   - `_get_etd_timestamps`
   - `_get_total_data_points_etd`
   - `_process_epoch`
   - `validate_epoch_gaps`
   - `add_sleep_stage_to_csv_buffer`
   - `_has_enough_delay_since_last_epoch`
   - `manage_epoch`

## Recent Changes

- Added index variable naming to distinguish absolute vs relative positions
- Created placeholder for memory management offset tracking
- Updated gap handler to work with relative indices

## Next Steps

1. Implement buffer clearing logic that accounts for overlap
2. Add offset tracking
3. Update index translation methods
4. Add buffer clearing triggers
5. Update affected methods
6. Add validation and testing

# Implementation Plan

## Write Tests

Implement the tests

- [x] Buffer Trimming Tests
- [x] Index Translation Tests
- [x] Processing Continuity Tests
- [x] Round-Robin Integration Test/

Triming Functionality

- [x] Implement `_trim_etd_buffer`
- [x] Test `_trim_etd_buffer`

Refactor the code to use an ETDBufferManager class

- [x] Create ETDBufferManager class and move the code from DataManager to it

Index Translation

- [x] Implement the index translation methods
- [x] Test the index translation methods
