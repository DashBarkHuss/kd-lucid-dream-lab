# Main Files Refactoring Notes

## Current Issues with main.py and main_speed_controlled_stream.py

These files are **application entry points/scripts** rather than traditional testable modules. They have several design issues that make them difficult to test and maintain.

### Problems with Current Design

1. **Monolithic main() functions** - Both have large, complex main functions that do too many things
2. **Hard-coded file paths** - They reference specific data files that may not exist in test environments
3. **External dependencies** - Rely on GUI components (PyQt), multiprocessing, and hardware simulation
4. **Side effects** - Create files, start processes, modify global state
5. **No return values** - Main functions don't return testable results
6. **Mixed responsibilities** - Orchestration logic mixed with business logic

### Current Architecture Problem

```python
# ORCHESTRATION (coordination) mixed with IMPLEMENTATION (business logic)
def main():
    # Setup (orchestration)
    board_manager = BoardManager(playback_file, board_id)
    board_manager.setup_board()
    
    # Business logic embedded in orchestration
    while True:
        new_data = board_manager.get_new_data()
        if new_data.size > 0:
            # Complex timestamp logic (implementation)
            if start_first_data_ts is None and board_timestamp_channel is not None:
                start_first_data_ts = float(new_data[board_timestamp_channel][0])
            
            # Gap detection logic (implementation)  
            timestamps = original_playback_data.iloc[:, board_timestamp_channel]
            next_rows = timestamps[timestamps > last_good_ts]
            
            # File creation logic (implementation)
            create_trimmed_csv(playback_file, trimmed_file, offset)
```

## Proposed Refactoring: Separate Orchestration from Implementation

### Orchestration Layer (main.py)
Should only coordinate components - thin composition root:

#### Conceptual Example:
```python
def main():
    # Just coordinate components
    pipeline = DataProcessingPipeline(config)
    gap_handler = GapHandler(config)
    
    while pipeline.has_data():
        data = pipeline.get_next_batch()
        result = pipeline.process(data)
        
        if gap_handler.gap_detected(result):
            pipeline.restart_from_gap(gap_handler.get_restart_point())
```

### Implementation Layer (separate modules)
Contains the actual business logic:

```python
class DataProcessingPipeline:
    def process(self, data):
        # All the timestamp logic, data processing, etc.
        
class GapHandler:
    def gap_detected(self, result):
        # All the gap detection logic
        
    def get_restart_point(self):
        # All the offset calculation logic
```

## Benefits of Separation

1. **Testable**: You can test `DataProcessingPipeline.process()` without running the full app
2. **Focused**: Each class has one clear responsibility  
3. **Reusable**: The implementation classes can be used in different contexts
4. **Maintainable**: Business logic changes don't affect orchestration and vice versa
5. **Configurable**: Dependencies can be injected for testing
6. **Modular**: Components can be developed and tested independently

## Refactoring Plan

1. **Extract business logic** into separate, focused classes
2. **Create smaller, testable functions** with clear inputs/outputs
3. **Use dependency injection** for file paths and external dependencies
4. **Separate concerns**: gap handling, data processing, file management, etc.
5. **Make main files thin orchestrators** that just wire components together

## What Should Be Extracted

From the current main files, these concerns should be separate modules:

- **Gap Detection & Handling** - Logic for detecting data gaps and calculating restart points
- **File Management** - Creating trimmed files, managing offset files
- **Timestamp Processing** - All timestamp-related calculations and conversions
- **Stream Coordination** - Managing stream lifecycle and restart logic
- **Configuration Management** - Handling file paths, board settings, speed multipliers

The main files should become simple composition roots that instantiate and coordinate these components.

## Detailed Module Breakdown

### Core Modules Needed (3 main components):

#### 1. **DataProcessingPipeline**
- **Location**: `gssc_local/realtime_with_restart/data_processing_pipeline.py`
- **Purpose**: Orchestrates the main data processing flow
- **Responsibilities**: 
  - Coordinates board_manager → data_handler → visualizer flow
  - Manages processing state and data chunks
  - Handles the main processing loop logic
  - Integrates with existing CSVManager for data export
- **Key Methods**: `process()`, `has_data()`, `get_next_batch()`, `restart_from_gap()`

#### 2. **GapHandler** 
- **Location**: `gssc_local/realtime_with_restart/gap_handler.py` *(already exists!)*
- **Purpose**: Detects gaps and calculates restart points
- **Responsibilities**:
  - Gap detection logic from main loop
  - Offset calculation for restart points
  - Timestamp comparison and validation
- **Key Methods**: `detect_gap()`, `get_restart_point()`, `calculate_offset()`

#### 3. **StreamCoordinator**
- **Location**: `gssc_local/realtime_with_restart/stream_coordinator.py`
- **Purpose**: Manages the stream lifecycle and restart logic
- **Responsibilities**:
  - Start/stop streams and stream managers
  - Handle the outer restart loop from main()
  - Coordinate between pipeline and gap handler
  - Manage stream state transitions
  - Handle file operations like creating trimmed CSV files
- **Key Methods**: `process_until_gap_or_end()`, `restart_from_gap()`, `has_more_data()`, `create_trimmed_csv()`

### Utility Modules to Extract:

#### 4. **TimestampUtils**
- **Location**: `gssc_local/realtime_with_restart/utils/timestamp_utils.py`
- **Purpose**: Timestamp processing utilities
- **Responsibilities**:
  - Format timestamps for display (`format_timestamp()`)
  - Format elapsed time (`format_elapsed_time()`)
  - Timestamp calculations and conversions
- **Key Methods**: `format_timestamp()`, `format_elapsed_time()`, `calculate_elapsed()`

### Existing Components to Leverage:

#### **CSVManager** *(already exists)*
- **Location**: `gssc_local/realtime_with_restart/export/csv/manager.py`
- **Purpose**: Handles all CSV data export and validation
- **Why not create FileManager**: CSVManager already handles all file operations needed for BrainFlow data export
- **Integration**: DataProcessingPipeline will use existing CSVManager instance from data_handler

## Refactored Architecture

### Concrete Implementation Plan:
```python
def main():
    # Configuration and setup
    config = ProcessingConfig(playback_file, board_id, speed_multiplier)
    
    # Core components (leveraging existing CSVManager)
    gap_handler = GapHandler(config)  # Already exists
    pipeline = DataProcessingPipeline(config)  # Uses existing CSVManager internally
    stream_coordinator = StreamCoordinator(pipeline, gap_handler)
    
    # Thin orchestration loop
    while stream_coordinator.has_more_data():
        result = stream_coordinator.process_until_gap_or_end()
        
        if result.gap_detected:
            stream_coordinator.restart_from_gap(result.restart_point)
        elif result.no_more_data:
            break
    
    # Cleanup
    stream_coordinator.cleanup()
```

### Benefits of This Architecture:
- **Single Responsibility**: Each module has one clear purpose
- **Testable**: Each component can be unit tested independently
- **Dependency Injection**: Components receive their dependencies, making them configurable
- **Reusable**: Components can be used in different contexts (real vs simulated streams)
- **Maintainable**: Changes to business logic don't affect orchestration