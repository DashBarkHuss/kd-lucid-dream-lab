# CSV Manager Memory Management Update

## Overview

This document outlines the tasks required to update the existing CSVManager class with memory management features. These changes will enhance the CSVManager's ability to handle large datasets by implementing buffer management and preventing memory overflow.

## Implementation Tasks

### 1. Update Module Structure

- [ ] Update class interface:
  - [ ] Add method stubs with docstrings for new buffer management methods:
    - [ ] `save_incremental_to_csv()`
    - [ ] `save_sleep_stages_to_csv()`
    - [ ] `merge_files()`
  - [ ] Add buffer-related custom exceptions
  - [ ] Add input validation methods for buffer operations
  - [ ] Define error handling behavior for buffer operations
  - [ ] Add data validation methods for buffer operations

### 2. Implement Buffer Management

- [ ] Add new attributes to CSVManager:
  - [ ] `buffer_size` (default: 10,000 samples)
  - [ ] `current_buffer_size`
  - [ ] `is_first_write`
  - [ ] `sleep_stage_csv_path`
  - [ ] `main_csv_buffer` (renamed from `saved_data`)
  - [ ] `sleep_stage_buffer`
  - [ ] `sleep_stage_buffer_size` (default: 100 entries)
- [ ] Implement buffer state tracking
- [ ] Add buffer overflow protection
- [ ] Add buffer validation methods

### 3. Update Existing Methods and Add New Ones

- [ ] Update existing methods:
  - [ ] `save_new_data_to_csv_buffer()` → `add_data_to_buffer()`
    - [ ] Add buffer size check
    - [ ] Trigger save when buffer is full
  - [ ] `save_to_csv()`
    - [ ] Handle remaining data
    - [ ] Ensure proper cleanup
  - [ ] `add_sleep_stage_to_csv_buffer()` → `add_sleep_stage_to_sleep_stage_csv()`
    - [ ] Add to sleep stage buffer
    - [ ] Handle buffer size limits
    - [ ] Trigger save when buffer is full
- [ ] Add new methods:
  - [ ] `save_incremental_to_csv()`
    - [ ] Handle first write vs append operations
    - [ ] Preserve exact CSV format
    - [ ] Clear buffer after successful write
  - [ ] `save_sleep_stages_to_csv()`
  - [ ] `merge_files()`

### 4. Update Dependencies

- [ ] Update DataManager to use new buffer management
- [ ] Ensure proper initialization and cleanup
- [ ] Handle buffer configuration in main application

### 5. Add Unit Tests

- [ ] Buffer Management Tests
  - [ ] Buffer initialization
  - [ ] Overflow handling
  - [ ] Buffer clearing
- [ ] Incremental Save Tests
  - [ ] First write vs append
  - [ ] CSV format preservation
  - [ ] Partial writes
- [ ] Sleep Stage Update Tests
  - [ ] Sleep stage buffer handling
  - [ ] Timestamp matching
  - [ ] Data integrity
- [ ] Error Handling Tests
  - [ ] Disk write failures
  - [ ] File permission issues
  - [ ] Buffer overflow handling

### 6. Documentation

- [ ] Update class and method documentation
- [ ] Add usage examples
- [ ] Document validation rules
- [ ] Update main README if needed

### 7. Validation & Review

- [ ] Run existing tests
- [ ] Check for regressions
- [ ] Review error handling
- [ ] Verify file format consistency

## Success Criteria

1. Memory Usage

   - Peak memory usage below configured limit
   - No memory leaks

2. Data Integrity

   - No data loss during incremental saves
   - Proper handling of sleep stage data
   - Accurate CSV format preservation

3. Performance

   - Minimal impact on processing speed
   - Efficient disk I/O

4. Error Handling
   - Basic error handling for disk operations
   - Clear error messages

## Dependencies

- numpy
- pandas
- logging

## Notes

- Keep existing format exactly for compatibility
- Maintain current validation rules
- Consider adding optional data format validation
- Plan for future format versioning
