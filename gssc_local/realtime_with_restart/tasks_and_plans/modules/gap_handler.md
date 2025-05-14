# Gap Handler Implementation Plan

## Overview

Extracts gap detection functionality from DataManager into a dedicated GapHandler class. The GapHandler is responsible for detecting gaps in the data stream and returning them. The GapHandler focuses purely on gap detection and reporting, without knowledge of buffer management or other system components.

## Phase 1: Basic Implementation

### 1. Create Module Structure

- [x] Create `core/gap_handler.py` with basic structure
- [x] Create `core/__init__.py` if it doesn't exist

### 2. Implement GapHandler Class

- [x] Define class with clear interface:
  - [x] Create method stubs with docstrings
  - [x] Define custom exceptions
  - [x] Add input validation methods
  - [x] Define error handling behavior
  - [x] Add data validation methods
- [x] Move gap-related functionality from DataManager:
  - [x] `detect_gap()`
  - [x] `validate_epoch_gaps()`
- [x] Add proper error handling and validation:
  - [x] Set up logging
  - [x] Implement input validation
  - [x] Add error handling for gap detection
  - [x] Add data integrity checks
- [x] Add type hints and docstrings

### 3. Update Dependencies

- [x] Update DataManager to use new GapHandler
- [x] Ensure proper initialization and cleanup

### 4. Add Unit Tests

- [x] Test gap detection functionality
- [x] Test gap validation logic
- [x] Test error handling
- [x] Test data integrity checks

### 5. Documentation

- [ ] Add class and method documentation
- [ ] Add usage examples
- [ ] Document validation rules
- [ ] Update main README if needed
- [ ] Update tasks_and_plans/REFACTORING.md with instructions to move buffer timing calculations to BufferManager

### 6. Validation & Review

- [x] Run existing tests
- [x] Check for regressions
- [x] Review error handling
- [x] Verify gap detection accuracy

## Buffer Timing Migration

The following functionality from `_get_affected_buffer()` needs to be moved to BufferManager:

- [ ] Create new method in BufferManager for buffer timing calculations:
  - [ ] Calculate buffer ID from timestamp
  - [ ] Handle buffer timing windows
  - [ ] Manage buffer epoch boundaries
- [ ] Update DataManager to use BufferManager for timing calculations
- [ ] Add tests for buffer timing functionality
- [ ] Document buffer timing rules and calculations

## Success Criteria

1. All gap detection works through new GapHandler
2. No changes to gap detection behavior
3. All tests passing
4. Clear error messages for validation failures
5. Clean interface for gap reporting
6. Accurate gap detection
7. No knowledge of buffer management or other system components

## Dependencies

- numpy
- logging

## Notes

- Keep existing gap detection logic exactly for compatibility
- Maintain current validation rules
- Consider adding configurable gap thresholds
- Plan for future interpolation capabilities
- Gap response coordination handled by orchestrator
- GapHandler should be buffer-agnostic
- Buffer timing calculations belong in BufferManager

## Future Enhancements (Phase 2)

- [ ] Add interpolation for small gaps
- [ ] Add gap visualization:
  - [ ] Show red vertical lines at gap locations
  - [ ] Display gap size labels
  - [ ] Adjust time axis for discontinuities
- [ ] Add configurable gap thresholds
- [ ] Define clear gap reporting interface
