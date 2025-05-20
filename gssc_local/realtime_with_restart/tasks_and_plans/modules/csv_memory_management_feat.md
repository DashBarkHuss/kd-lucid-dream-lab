# CSV Memory Management Feature Implementation Plan

## Overview

Currently, the CSVManager stores all data in memory until the end of the stream, which could cause memory issues during long recordings. This plan outlines the implementation of incremental CSV saving to manage memory usage.

## BrainFlow Internal Buffer Behavior

We will control our own buffer; `main_csv_buffer`. But, so we understand the upper limits of the buffer we could build, we should understand how BrainFlow's internal buffer works.

BrainFlow uses a ring buffer for data collection with the following characteristics:

- Default buffer size: 450,000 samples
- Behavior: Circular buffer that overwrites oldest data when full
- No data loss: Instead of dropping samples, oldest data is overwritten
- Buffer size is configurable via `start_stream(num_samples)`

Example buffer behavior at 125 Hz sampling rate:

- 5 minutes of data = 37,500 samples (well within buffer)
- 1 hour of data = 450,000 samples (exactly fills buffer)
- 2 hours of data = 900,000 samples (oldest 450,000 samples overwritten)

This means:

1. We can safely sleep/process for up to 1 hour without data loss
2. After 1 hour, we'll only have the most recent hour of data
3. The buffer size can be adjusted if we need more or less history

## Current Implementation

### CSVManager

- Stores all data in `self.saved_data` list
- Only writes to disk at end of stream via `save_to_csv()`
- No memory management or incremental saving
- Sleep stage data is stored in the main buffer
- Note: `saved_data` is NOT used for visualization or processing - it's purely for CSV export

## Proposed Changes

### New Attributes

- `buffer_size`: Maximum rows to keep in memory (default: 10,000)
- `current_buffer_size`: Current number of rows in memory
- `is_first_write`: Flag for first write vs append operations
- `sleep_stage_csv_path`: Path for separate sleep stage CSV file
- `sleep_stage_buffer`: List holding current sleep stage data
- `sleep_stage_buffer_size`: Maximum sleep stage entries to keep in memory (default: 100)

### New Methods

- `save_incremental_to_csv()`: Write current buffer to disk and clear it

  - Handles both first write (create new file) and subsequent writes (append)
  - Preserves exact CSV format
  - Clears buffer after successful write

- `merge_files()`: Combine main CSV and sleep stage CSV at end of recording
  - Matches timestamps between files
  - Creates final merged output
  - Validates merged data integrity

### Modified Methods

- `save_new_data_to_csv_buffer()`:

  - Add buffer size check after adding new data
  - If buffer exceeds size limit, call `save_incremental_to_csv()`
  - Maintain exact format of original data
  - No sleep stage columns in main buffer

- `add_sleep_stage_to_csv_buffer()`:

  - Add sleep stage data to sleep stage buffer
  - If buffer exceeds size limit, call `save_sleep_stages_to_csv()`
  - Format: timestamp_start, timestamp_end, sleep_stage, buffer_id

- `save_sleep_stages_to_csv()`:

  - Write current sleep stage buffer to CSV file
  - Clear buffer after successful write
  - Handle both first write and append operations

- `save_to_csv()`:

  - Save any remaining data in buffer
  - Save any remaining sleep stage data
  - Ensure proper cleanup

- `cleanup()`:
  - Save any remaining data before cleanup
  - Save any remaining sleep stage data
  - Reset buffer state
  - Close sleep stage CSV file

## Implementation Phases

### Phase 1: Basic Incremental Saving

1. Add buffer size configuration to CSVManager
2. Implement `save_incremental_to_csv()`
3. Modify `save_new_data_to_csv_buffer()` to check buffer size
4. Update `save_to_csv()` and `cleanup()` to handle remaining data

### Phase 2: Sleep Stage File Management

1. Add sleep stage CSV file handling
2. Implement direct sleep stage writing
3. Add file merging functionality
4. Add validation for sleep stage data

### Phase 3: Error Handling

1. Add basic error handling for disk writes
2. Add file permission checks
3. Add basic validation for data format

### Phase 4: Testing & Validation

1. Test with large datasets
2. Verify memory usage
3. Test basic error handling
4. Validate data integrity
5. Test CSV file merging
6. Test sleep stage file handling

## Success Criteria

1. Memory Usage

   - Peak memory usage stays below configured limit
   - No memory leaks

2. Data Integrity

   - No data loss during incremental saves
   - Proper handling of sleep stage data
   - Accurate CSV format preservation
   - Correct timestamp matching in merged files

3. Performance

   - Minimal impact on processing speed
   - Efficient disk I/O
   - Proper cleanup of resources

4. Error Handling
   - Basic error handling for disk operations
   - Clear error messages

## Dependencies

- numpy
- pandas
- logging

## Notes

- Buffer size should be configurable based on available memory
- Need to ensure proper file permissions for writing
- Sleep stage data is handled separately from main data
- Files are merged at end of recording for final output
- Buffer size limits should be handled gracefully and maintain data integrity

## Implementation Approach

### Overview

- Never lose sleep stage updates
- Fixed, predictable memory usage
- Simple to implement
- No need for complex file updates
- Separate sleep stage handling for better memory management

### Implementation Details

- Buffer Size:

  - Fixed size based on memory constraints (default: 10,000 samples)
  - Not tied to processing cycles
  - Configurable based on system requirements

- Sleep Stage Buffer Size:

  - Fixed size based on memory constraints (default: 100 entries)
  - Each entry contains: timestamp_start, timestamp_end, sleep_stage, buffer_id
  - Write to disk when buffer is full
  - Much smaller than main buffer since sleep stages are less frequent

- Buffer Retention Strategy:

  - Keep buffer until it reaches the configured size limit
  - Save to CSV and clear buffer when size limit is reached
  - No need to wait for sleep stage scoring

- File Management:

  - Main CSV: Contains raw EEG data with timestamps
  - Sleep Stage CSV: Contains sleep stage annotations
  - Both files use consistent timestamp format
  - Files are merged at end of recording

- Save Strategy:

  - Save EEG data to main CSV when buffer size limit is reached
  - Save sleep stages to CSV when sleep stage buffer is full
  - No need to modify existing CSV data
  - Simpler error recovery

- Memory Management:
  - Keep buffer until size limit is reached
  - Implement buffer overflow protection
  - Clear buffer after successful save

### Data Flow

1. New data arrives via `save_new_data_to_csv_buffer()`

   - Add data to buffer
   - Check if len(saved_data) (changing to main_csv_buffer) exceeds buffer_size
   - If exceeds, trigger save to main CSV
   - Clear buffer after successful save

2. Save to main CSV

   - Write current buffer to main CSV file
   - Clear buffer after successful save

3. Sleep stage updates via `add_sleep_stage_to_csv_buffer()`

   - Add to sleep stage buffer
   - If buffer exceeds size limit, trigger save to sleep stage CSV
   - Clear buffer after successful save

4. File Management

   - Main CSV: Contains raw EEG data with timestamps
   - Sleep Stage CSV: Contains sleep stage annotations
   - Both files use consistent timestamp format
   - Files are merged at end of recording

5. Cleanup process
   - Check for remaining data in buffer
   - Save any remaining data to main CSV
   - Save any remaining sleep stages
   - Merge files if needed
   - Reset all state variables

### Key Methods

#### save_new_data_to_csv_buffer()

```python
def save_new_data_to_csv_buffer(self, new_data: np.ndarray, is_initial: bool = False) -> bool:
    # Add new data to buffer
    # If buffer size exceeds limit:
    #   - Call save_to_main_csv()
    #   - Clear buffer after successful save
```

#### save_to_main_csv()

```python
def save_to_main_csv(self, output_path: Union[str, Path]) -> bool:
    # If first write:
    #   - Create new file
    #   - Set is_first_write = False
    # Else:
    #   - Append to existing file
    # Clear buffer after successful save
```

#### add_sleep_stage_to_csv_buffer()

```python
def add_sleep_stage_to_csv_buffer(self, sleep_stage: float, buffer_id: float,
                                timestamp_start: float, timestamp_end: float) -> None:
    # Add to sleep stage buffer
    # If buffer size exceeds limit:
    #   - Call save_sleep_stages_to_csv()
    #   - Clear buffer after successful save
```

#### save_sleep_stages_to_csv()

```python
def save_sleep_stages_to_csv(self) -> bool:
    # If first write:
    #   - Create new file
    #   - Set is_first_write = False
    # Else:
    #   - Append to existing file
    # Clear buffer after successful save
```

#### merge_files()

```python
def merge_files(self, main_csv_path: Union[str, Path],
               sleep_stage_csv_path: Union[str, Path],
               output_path: Union[str, Path]) -> bool:
    # Read both CSV files
    # Match timestamps
    # Create merged file with all data
    # Validate merged data
    # Clean up temporary files
```

#### cleanup()

```python
def cleanup(self) -> None:
    # If buffer has data:
    #   - Call save_to_main_csv()
    # Save any pending sleep stages
    # Merge files if needed
    # Reset all state variables
    # Clear buffers
```

### State Management

- `is_first_write`: Tracks if we need to create or append to file
- `main_csv_buffer`: List holding current buffer of data
- `main_csv_path`: Path to main EEG data file
- `sleep_stage_csv_path`: Path to sleep stage annotations file

### Error Handling

- Handle disk write failures
- Handle file permission issues
- Ensure data integrity during saves
- Handle sleep stage update failures
- Handle file merging errors
- Log warnings for late sleep stage updates

### Testing Strategy

#### Unit Tests

1. Buffer Management Tests

   - Test buffer initialization
   - Verify buffer overflow handling
   - Test buffer clearing after incremental saves

2. Incremental Save Tests

   - Test first write vs append operations
   - Verify CSV format preservation
   - Test handling of partial writes

3. Sleep Stage Update Tests

   - Test writing to sleep stage CSV file
   - Test timestamp matching
   - Test file merging
   - Verify sleep stage data integrity

4. Error Handling Tests
   - Test disk write failures
   - Test file permission issues
   - Test buffer overflow handling

#### Integration Tests

1. End-to-End Flow Tests

   - Test complete data flow from collection to final CSV
   - Verify memory usage over long recordings
   - Verify data integrity across saves

2. Performance Tests
   - Measure memory usage over time
   - Test processing speed impact
   - Verify disk I/O efficiency

#### Validation Tests

1. Data Integrity

   - Verify all data points are saved
   - Check timestamp continuity
   - Verify sleep stage accuracy
   - Validate CSV format compliance

2. Memory Management
   - Monitor memory usage patterns
   - Verify buffer size limits
   - Check for memory leaks
