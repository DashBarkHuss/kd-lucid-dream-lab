# CSV Manager Implementation Plan

## Overview

Extracts CSV management functionality from DataManager into a dedicated CSVManager class.

## Phase 1: Basic Implementation

### 1. Create Module Structure

- [x] Create `export/__init__.py`
- [x] Create `export/csv_manager.py` with basic structure

### 2. Implement CSVManager Class

- [x] Define class with clear interface:
  - [x] Create method stubs with docstrings
  - [x] Define custom exceptions
  - [x] Add input validation methods
  - [x] Define error handling behavior
  - [x] Add data validation methods
- [x] Move CSV-related functionality from DataManager:

  - [x] `save_to_csv()`
  - [x] `validate_saved_csv()`
  - [x] `save_new_data()`
  - [x] `add_sleep_stage_to_csv()`
        clarifying questions after the step above:

        1. I'm unclear what validate shape came from. I don't see that method in data_manager.
        - answer: the llm made it up

        2. why do we have different functions in the save to csv in data_manager vs csv_manager? the old one in data_manager worked, so we need to make sure this new onw works. if there isn't a clear reason why you changed the function, then maybe it was a mistake and you should use the old version.
        answer: mistake. the llm modified the save_to_csv function in csv_manager.py to match the working version from data_manager.py.

        3. l216- why don't we use th e custom class for errors here? was this on purpose?
        answer: it was there, didn't see it.
        4. l259 same as above
        answer: it was there, didn't see it.

        5. the new validate_saved_csv is a fine idea, but it isn't the same function as the old validate_saved_csv in data_manager. the one it data managger mayber should be called validate_saved_csv_matches_original_source and the ine you made in csv_manager should be called validate_saved_csv_format, as they have different purposes. we need to add validate_saved_csv_matches_original_source.
        - answer: the llm renamed the function validate_saved_csv to validate_saved_csv_format and added a new function validate_saved_csv_matches_original_source that matches the old one in data_manager.

        6. where in data_manger did we have this or something like it _validate_timestamp_continuity? I'm not sure this is something we need to be true of our data

        7. make sure ll our referenced self variables are defined somewhere. It's possible we copied them from data_manager without making sure they were defined in csv_manager.
        answer: the llm did not find any self variables that were not defined in csv_manager.

- [x] Add proper error handling and validation:
  - [x] Set up logging
  - [x] Implement input validation
  - [x] Add error handling for file operations
  - [x] Add data integrity checks
- [x] Add type hints and docstrings

### 3. Update Dependencies

In progress:

- [... ] Update DataManager to use new CSVManager
- [... ] Ensure proper initialization and cleanup
- [... ] Handle timestamp channel configuration

### 4. Add Unit Tests

- [x] Test CSV writing functionality
- [x] Test validation logic
- [x] Test error handling
- [x] Test data integrity checks

### 5. Documentation

- [ ] Add class and method documentation
- [ ] Add usage examples
- [ ] Document validation rules
- [ ] Update main README if needed

### 6. Validation & Review

- [ ] Run existing tests
- [ ] Check for regressions
- [ ] Review error handling
- [ ] Verify file format consistency

## Success Criteria

1. All CSV operations work through new CSVManager
2. No changes to output file format
3. All tests passing
4. Clear error messages for validation failures
5. No data loss during export
6. Proper handling of sleep stage data

## Dependencies

- numpy
- pandas (if needed)
- logging

## Notes

- Keep existing format exactly for compatibility
- Maintain current validation rules
- Consider adding optional data format validation
- Plan for future format versioning

## Future Enhancements (Phase 2)

- [ ] Add support for different CSV formats
- [ ] Add data format versioning
- [ ] Improve validation configurability
- [ ] Add data compression options
- [ ] Add batch processing capabilities
