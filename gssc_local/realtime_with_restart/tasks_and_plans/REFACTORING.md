# DataManager Refactoring Plan

## Architectural Principles

1. **Scoped Validation**: Validation should be scoped to the thing it protects

   - Each component validates its own data
   - Validation logic lives with the code it protects
   - Components can't be used incorrectly without validation
   - Changes to validation rules are scoped to the affected component

2. **Single Responsibility**: Each module should have one primary responsibility
   - Clear boundaries between components
   - Focused, testable functionality
   - Minimal dependencies between modules

## Current Responsibilities

### 1. Buffer Management

- **Purpose**: Manages data buffers and their states for EEG data processing
- **Key Functions**:
  - `add_data()`: Adds new data to buffers
  - `_init_channels()`: Initializes channel information
  - `_get_next_epoch_indices()`: Calculates next epoch indices
  - `_has_enough_data_for_buffer()`: Checks buffer data sufficiency
- **Validation Scope**:
  - Buffer data integrity
  - Consecutive data values
  - Buffer state consistency
- **Dependencies**:
  - Board configuration
  - Sampling rate
  - Channel configuration

### 2. Epoch Processing

- **Purpose**: Processes EEG data in epochs for sleep stage prediction
- **Key Functions**:
  - `_process_epoch()`: Processes individual epochs
  - `manage_epoch()`: Manages epoch processing workflow
  - `next_available_epoch_on_buffer()`: Determines next processable epoch
- **Validation Scope**:
  - Epoch length
  - Epoch data integrity
  - Epoch timing
- **Dependencies**:
  - Signal processor
  - Buffer states
  - ML model

### 3. Gap Detection/Handling

- **Purpose**: Detects and handles data gaps
- **Key Functions**:
  - `detect_gap()`: Identifies gaps in data
  - `handle_gap()`: Manages gap handling
  - `reset_buffer_states()`: Resets states after gaps
- **Validation Scope**:
  - Timestamp continuity
  - Gap thresholds
  - Interpolation boundaries
- **Dependencies**:
  - Gap thresholds
  - Buffer states

### 4. CSV Export/Management

- **Purpose**: Manages data export and validation
- **Key Functions**:
  - `save_to_csv()`: Exports data to CSV
  - `save_new_data()`: Saves new data chunks
  - `add_sleep_stage_to_csv()`: Adds sleep stage data
- **Validation Scope**:
  - CSV format integrity
  - Data consistency
  - Output format
- **Dependencies**:
  - File system
  - Data format

### 5. State Management

- **Purpose**: Manages processing state
- **Key Functions**:
  - `_calculate_next_buffer_id_to_process()`: Determines next buffer
  - Tracks processed epochs
  - Manages buffer states
- **Validation Scope**:
  - State transitions
  - Processing order
  - Buffer synchronization
- **Dependencies**:
  - Buffer configuration
  - Processing state

### 6. Visualization Coordination

- **Purpose**: Manages data visualization
- **Key Functions**:
  - Coordinates with visualizer
  - Updates plots
  - Manages visualization timing
- **Validation Scope**:
  - Plot data integrity
  - Timing synchronization
  - Display format
- **Dependencies**:
  - PyQtVisualizer
  - Plot configuration

### 7. Channel Management

- **Purpose**: Manages EEG channels
- **Key Functions**:
  - Manages electrode channels
  - Handles timestamp channels
  - Coordinates channel indices
- **Validation Scope**:
  - Channel configuration
  - Channel mapping
  - Data alignment
- **Dependencies**:
  - Board configuration
  - Montage settings

## Refactoring Goals

1. **Modularity**: Split into focused, single-responsibility modules
2. **Testability**: Improve unit test coverage
3. **Maintainability**: Reduce code complexity
4. **Documentation**: Improve code documentation
5. **Performance**: Optimize critical paths

## Proposed New Structure

```
gssc_local/realtime_with_restart/
├── data/
│   ├── __init__.py
│   ├── buffer_manager.py      # Buffer management with buffer-specific validation
│   ├── epoch_processor.py     # Epoch processing with epoch-specific validation
│   └── gap_handler.py        # Gap handling with gap-specific validation
├── export/
│   ├── __init__.py
│   └── csv_manager.py        # CSV management with format-specific validation
├── state/
│   ├── __init__.py
│   └── state_manager.py      # State management with state-specific validation
└── data_manager.py           # Main orchestrator (simplified)
```

## Refactoring Steps

1. Create new module structure
2. Move each responsibility to its own module, including its specific validation logic
3. Update imports and dependencies
4. Add unit tests for each module's functionality and validation
5. Update documentation
6. Verify functionality
7. Clean up old code

## Testing Strategy

1. Unit tests for each module's core functionality
2. Unit tests for each module's validation logic
3. Integration tests for module interactions
4. End-to-end tests for full functionality
5. Performance benchmarks
6. Regression testing

## Migration Plan

1. Create new modules alongside existing code
2. Move functionality and its corresponding validation
3. Update dependencies
4. Test each change
5. Remove old code
6. Update documentation

## Notes

- Each module owns its validation logic
- Validation failures should be clear and actionable
- Keep existing functionality working during migration
- Add tests before moving code
- Document validation rules in each module
- Review performance impact

## Implementation Plans

See detailed implementation plans for each module in the `tasks_and_plans/modules/` directory:

- [`modules/csv_manager.md`](modules/csv_manager.md) - CSV Export/Management implementation
- `modules/buffer_manager.md` (Coming soon)
- `modules/epoch_processor.md` (Coming soon)
- `modules/gap_handler.md` (Coming soon)
- `modules/state_manager.md` (Coming soon)
- `modules/channel_manager.md` (Coming soon)

Each module plan includes:

- Detailed implementation steps
- Success criteria
- Dependencies
- Testing strategy
- Notes and considerations
