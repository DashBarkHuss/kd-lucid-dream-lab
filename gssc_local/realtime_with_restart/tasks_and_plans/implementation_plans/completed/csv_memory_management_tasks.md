# CSV Manager Memory Management Update

## Overview

This document outlines the tasks required to update the existing CSVManager class with memory management features. These changes will enhance the CSVManager's ability to handle large datasets by implementing buffer management and preventing memory overflow.

## Implementation Tasks

### 1. Update Module Structure

- [x] Update class interface:
  - [x] Add method stubs with docstrings for new buffer management methods:
    - [x] `save_incremental_to_csv()`
    - [x] `save_sleep_stages_to_csv()`
    - [x] `merge_files()`
  - [x] Add buffer-related custom exceptions
  - [x] Add input validation methods for buffer operations
  - [x] Define error handling behavior for buffer operations
  - [x] Add data validation methods for buffer operations

### 2. Implement Buffer Management

- [x] Add new attributes to CSVManager:
  - [x] `main_buffer_size` (default: 10,000 samples)
  - [x] `is_first_write`
  - [x] `sleep_stage_csv_path`
  - [x] `main_csv_buffer` (renamed from `saved_data`)
  - [x] `sleep_stage_buffer`
  - [x] `sleep_stage_buffer_size` (default: 100 entries)
- [x] Implement buffer validation methods:
  - [x] Size validation in data addition methods
    - [x] Add overflow check to save_new_data_to_csv_buffer
    - [x] Add overflow check to add_sleep_stage_to_csv_buffer
  - [x] Data integrity checks
    - [x] Data structure validation
    - [x] Row length consistency
    - [x] Numeric value validation
    - [x] NaN/infinite value checks
  - [x] State validation for operations
    - [x] Buffer initialization checks
    - [x] Buffer size configuration checks

### 3. Update Existing Methods and Add New Ones

- [x] Update existing methods:
  - [x] `save_new_data_to_csv_buffer()` → `add_data_to_buffer()`
    - [x] Add buffer size check
    - [x] Trigger save when buffer is full
  - [x] `save_to_csv()` → `save_all_and_cleanup()`
    - [x] Handle remaining data
    - [x] Ensure proper cleanup
  - [x] `add_sleep_stage_to_csv_buffer()` → `add_sleep_stage_to_sleep_stage_csv()`
    - [x] Add to sleep stage buffer
    - [x] Handle buffer size limits
    - [x] Trigger save when buffer is full
- [x] Add new methods:
  - [x] `save_incremental_to_csv()`
    - [x] Handle first write vs append operations
    - [x] Preserve exact CSV format
    - [x] Clear buffer after successful write
  - [x] `save_sleep_stages_to_csv()`
  - [x] `merge_files()`

### 3A.[x] Update docs if any changes were made during implementation

### 4. Add Unit Tests

- [x] Add unit tests for the new methods (Tests added, pending verification)

  - [x] `save_incremental_to_csv()` (Tests added, pending verification)
    - [x] Test first write vs append operations
    - [x] Test error cases (no output path, invalid data)
    - [x] Test edge cases (empty data, maximum buffer sizes)
  - [x] `save_sleep_stages_to_csv()` (Tests added, pending verification)
    - [x] Test first write vs append operations
    - [x] Test error cases (no output path, invalid data)
    - [x] Test edge cases (empty buffer, maximum buffer sizes)
  - [x] `merge_files()` (Tests added, pending verification)
    - [x] Test successful merge
    - [x] Test error cases (missing files, invalid formats)
    - [x] Test edge cases (empty files, large files)

- [x] Update existing tests for the updated methods

  - [x] `save_new_data_to_csv_buffer()` -> `add_data_to_buffer()`
    - [x] Test buffer size limits
    - [x] Test automatic save when buffer is full
    - [x] Test backward compatibility and deprecation warnings
  - [x] `add_sleep_stage_to_csv_buffer()` -> `add_sleep_stage_to_sleep_stage_csv()`
    - [x] Test sleep stage buffer management
    - [x] Test automatic save when buffer is full
    - [x] Test backward compatibility and deprecation warnings
  - [x] `save_to_csv()` -> `save_all_and_cleanup()`
    - [x] Test remaining data handling
    - [x] Test proper cleanup
    - [x] Test backward compatibility and deprecation warnings

- [x] Run all tests and fix any issues

  - Run all tests and ask the llm to pick one test to fix
    Prompt:

    To start create a numbered list of all the tests that failed.

        ```
          Run all tests in the test csv manager. Create a numbered list of all the tests that failed and put them in a test_failures.md file. It should list the test name, the issue, the status, and the fix if any. Wait for my instructions.
        ```

        ````
         @gssc_local/realtime_with_restart/export/csv_manager.py @TEST_FAILURES.md @gssc_local/tests/test_csv_manager.py @gssc_local/realtime_with_restart/tasks_and_plans/modules/csv_memory_management_tasks.md    Get the first text that needs to be fixed in the @TEST_FAILURES.md and fix it . To fix it, first make sure we understand the issue: Is the issue the test, or the code that the test is testing? Add logs to figure this out. use the -s command to run the tests and see the logs. Once sure, fix the issue and then run the test again. I fit works, run all the tests again to make sure there are no regressions ```

After fixing the test and running all the test to check for regressions:

      ```Update the TEST_FAILURES.md to record the changes of status of the tests and the fix for the fixed test/s.```

- Open a new chat and repeat the process until all tests are fixed

- [x] Run all tests for updated methods
- [x] Run all tests for deprecated methods

- [x] document any changed made during the unit test fixes

  - No deviations from original plan - all changes were part of the planned implementation

### 5. Update Dependencies

- [x] Update DataManager to use new buffer management
- [x] Ensure proper initialization and cleanup
- [x] Handle buffer configuration in main application- this is done in data_manager.py not main.py

### 6. Documentation

- [ ] Update class and method documentation
- [ ] Add usage examples
- [ ] Document validation rules
- [ ] Update main README if needed
- [x] Move old implementation (csv_manager.md) plan to outdated directory and rename to csv_manager_outdated.md

#### Files Requiring Documentation Updates

The following files need to be updated to reflect the breaking changes in csv_manager.py:

1. Documentation Files:

   - [x] Does not exist. `/gssc_local/realtime_with_restart/tasks_and_plans/modules/csv_manager.md` - Main documentation file
   - [x] `/gssc_local/realtime_with_restart/tasks_and_plans/implementation_plans/outdated/csv_manager_outdated.md` - Outdated implementation plan- since it's marked as outdated, we don't need to update it.
   - [x] `/gssc_local/realtime_with_restart/tasks_and_plans/REFACTORING.md` - Contains reference to csv_manager.md
   - [x] `/gssc_local/README.md` - Contains usage examples that need updating

2. Code Files with Documentation References:
   - [x] `/gssc_local/realtime_with_restart/data_manager.py` - Contains integration documentation
   - [x] `/gssc_local/realtime_with_restart/main.py` - Contains usage examples
   - [x] `/gssc_local/realtime_with_restart/main_speed_controlled_stream.py` - Contains usage examples
   - [x] `/gssc_local/realtime_with_restart/tasks_and_plans/implementation_plans/completed/csv_memory_management_feat.md`
   - [x] `/gssc_local/realtime_with_restart/received_stream_data_handler.py` - Uses csv_manager through data_manager
   - [x] `/gssc_local/realtime_with_restart/export/csv_manager.py` - Contains implementation details

Breaking Changes to Document:

[x] Double check that this below is correct, the source fo truth is the actually implementation in main.py, main_controlled_stream.py, and data_manager.pt @test_csv_manager.py and @csv_manager.py. double check that and documentation in thes fiels are correct, but the code is the ultimate source of truth.

1. `save_new_data_to_csv_buffer()` → `add_data_to_buffer()`
2. `add_sleep_stage_to_csv_buffer()` → `add_sleep_stage_to_sleep_stage_csv()`
3. `save_to_csv()` → `save_all_data()` followed by `cleanup()`

### 7. Validation & Review

- [x] Run existing tests
- [x] Check for regressions
- [x] Review error handling
- [x] Verify file format consistency

## Success Criteria

1. Memory Usage

   - Peak memory usage below configured limit
     ✅ Tested in test_peak_memory_usage
     We verify memory stays below 100MB during 10 minutes of data processing
     However, we're not actually testing against a "configured limit" - we're using a hardcoded 100MB threshold

   - No memory leaks
     ✅ Tested in test_memory_leak_detection
     We verify memory doesn't grow by more than 10MB across 5 iterations
     However, this is a relatively simple test that might miss subtle leaks

2. Data Integrity

   - [x] No data loss during incremental saves
   - [x] Proper handling of sleep stage data
   - [x] Accurate CSV format preservation

3. Performance

   - [x] Minimal impact on processing speed
   - [x] Efficient disk I/O

4. Error Handling
   - [x] Basic error handling for disk operations
   - [x] Clear error messages

## Dependencies

- numpy
- pandas
- logging

## Notes

- Keep existing format exactly for compatibility
- Maintain current validation rules
- Consider adding optional data format validation
- Plan for future format versioning

@gssc_local/realtime_with_restart/tasks_and_plans/modules/csv_memory_management_tasks.md We need to implement our new memory management that we just made in csv manager in our @gssc_local/realtime_with_restart/main.py flow.
we need to update the DataManager to use the new CSV buffer management methods. @gssc_local/realtime_with_restart/data_manager.py

The key areas that need updating in DataManager are:

⏺ Update Todos
⎿ ☑ Update DataManager to use new CSV buffer management methods
☑ Replace save_new_data() with new add_data_to_buffer() method
☑ Update add_sleep_stage_to_csv_buffer() to use new method signature with timestamps
☑ Replace save_to_csv() with save_all_and_cleanup() and configure CSV paths
☑ Set main_csv_path and sleep_stage_csv_path in CSVManager initialization
☑ Test the updated DataManager with main.py flow @gssc_local/realtime_with_restart/main.py
