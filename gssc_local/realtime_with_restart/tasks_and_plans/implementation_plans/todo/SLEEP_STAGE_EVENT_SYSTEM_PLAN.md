# Sleep Stage Event System Implementation Plan

## Overview
Create an event-driven system to trigger different experimental flows based on real-time sleep stage detection. Use main script registration for simplicity and flexibility.

## Core Components to Create

### 1. Event System (`gssc_local/realtime_with_restart/event_system.py`)
- **SleepStageEvent**: Data class containing sleep stage, timestamp, confidence, epoch data
- **EventDispatcher**: Central hub that manages event delivery to registered handler functions
- **No handler classes needed** - use simple Python functions as handlers

### 2. Handler Functions (`gssc_local/experiments/handlers.py`)
- **Simple Python functions** that take a SleepStageEvent and perform actions
- Examples:
  - `play_n1_sound(event)`: Play sound when N1 detected
  - `log_all_stages(event)`: Log all sleep stage transitions
  - `custom_rem_action(event)`: Custom REM sleep logic
- **Much simpler than class-based approach** - just write the function you need

### 3. Integration Points

#### Modify DataManager (`gssc_local/realtime_with_restart/data_manager.py`)
- Add EventDispatcher to `__init__()` method
- Add event emission after sleep stage prediction in `_process_epoch()` (around line 611)
- Maintain backward compatibility with existing visualization

#### Update Main Scripts (Later Phase)
- For now: hardcode handler function registration in main scripts
- Later: can add argument parsing if needed
- Keep single main script pattern

## Implementation Steps

### Phase 1: Core Event System
1. ✅ Create `SleepStageEvent` data class
2. ✅ Implement `EventDispatcher` with handler function registration/emission  
3. ~~Create `ExperimentHandler` abstract base class~~ (Not needed - using functions)
4. Create example handler functions for testing event flow

### Phase 2: Integration
1. Modify `DataManager.__init__()` to accept optional EventDispatcher
2. Add event emission line in `_process_epoch()`
3. Create test main script to verify event flow

### Phase 3: Handler Functions  
1. Create example handler functions:
   - `play_sound_on_n1(event)`: Play sound when N1 detected using pygame/playsound
   - `log_all_events(event)`: Log all sleep stage transitions
   - `print_rem_detection(event)`: Simple REM detection printer
2. Add utility functions for common actions (play_sound, log_message, etc.)

### Phase 4: Testing and Refinement
1. Test event flow with existing data files
2. Add more sophisticated handler functions as needed
3. Add safety features (rate limiting, confidence thresholds) if needed
4. Document usage patterns

## Safety Features
- Rate limiting to prevent spam (max 1 action per minute)
- Confidence thresholds (only trigger on high-confidence predictions)
- Emergency stop functionality
- Experiment state management (start/stop/pause)

## Architecture Benefits
- **Separation of concerns**: Sleep detection logic stays unchanged
- **Flexibility**: Easy to add new experiment types
- **Explicit configuration**: Everything configured in main scripts (no hidden config files)
- **Backward compatibility**: Existing functionality unaffected
- **Testability**: Each component can be tested independently

## Example Usage

### Simple Audio Experiment (Hardcoded in Main Script)
```python
# In main_speed_controlled_stream.py
from gssc_local.experiments.handlers import play_sound_on_n1, log_all_events

# Create event dispatcher  
dispatcher = EventDispatcher()
dispatcher.register_handler("N1", play_sound_on_n1)
dispatcher.register_handler("ALL", log_all_events)

# Pass to DataManager
data_manager = DataManager(board_shim, sampling_rate, montage, event_dispatcher=dispatcher)
```

### Custom Handler Functions
```python
# In gssc_local/experiments/handlers.py
def play_sound_on_n1(event):
    if event.stage_text == "N1":
        play_sound("n1_alert.wav")
        print(f"N1 detected at {event.timestamp}")

def log_all_events(event):
    print(f"Sleep stage: {event.stage_text}, Confidence: {event.confidence:.2f}")

def custom_rem_action(event):
    if event.stage_text == "REM" and event.confidence > 0.8:
        # Your custom logic here
        print("High-confidence REM detected!")
```

### Benefits of Function-Based Approach
- **Simple**: Just write the function you need
- **Flexible**: Full Python power for custom logic
- **Testable**: Easy to test individual functions
- **No over-engineering**: No unnecessary classes or configuration

This approach keeps the core sleep detection unchanged while adding flexible experiment capabilities through explicit main script configuration.