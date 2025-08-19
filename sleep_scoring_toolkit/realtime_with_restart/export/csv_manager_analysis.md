# CSV Manager Refactoring Plan

## Current State Analysis

The CSVManager class currently handles multiple responsibilities:

1. Buffer management for EEG data
2. Buffer management for sleep stage data
3. File I/O operations
4. Data validation
5. Data merging
6. Error handling

Key issues:

- Class is too large (964 lines)
- Multiple responsibilities violate Single Responsibility Principle
- Complex state management
- Tight coupling between components
- Some validation methods are only used in tests

## Proposed Architecture

### 1. Core Components

```python
# Core Classes
class CSVManager:  # Orchestrator
    def __init__(self, config: CSVConfig):
        self.eeg_manager = EEGDataManager(config)
        self.sleep_manager = SleepStageManager(config)
        self.merger = DataMerger(config)

class EEGDataManager:  # Handles EEG data
    def __init__(self, config: CSVConfig):
        self.buffer = CircularBuffer(config.main_buffer_size)
        self.file_handler = CSVFileHandler()
        self.validator = EEGDataValidator()

class SleepStageManager:  # Handles sleep stage data
    def __init__(self, config: CSVConfig):
        self.buffer = CircularBuffer(config.sleep_stage_buffer_size)
        self.file_handler = CSVFileHandler()
        self.validator = SleepStageValidator()

class DataMerger:  # Handles merging of data
    def __init__(self, config: CSVConfig):
        self.file_handler = CSVFileHandler()
        self.validator = MergeValidator()
```

### 2. Supporting Classes

```python
# Data Models
@dataclass
class CSVConfig:
    main_buffer_size: int = 10_000
    sleep_stage_buffer_size: int = 100
    timestamp_precision: int = 6
    delimiter: str = '\t'
    file_format: str = '%.6f'

@dataclass
class EEGData:
    timestamp: float
    channels: List[float]

@dataclass
class SleepStageData:
    timestamp_start: float
    timestamp_end: float
    sleep_stage: float
    buffer_id: float

# Buffer Management
class CircularBuffer(Generic[T]):
    def __init__(self, max_size: int):
        """Initialize buffer with maximum size"""
        self.max_size = max_size
        self.data: Deque[T] = deque(maxlen=max_size)

    def add(self, item: T) -> bool:
        """
        Add single item to buffer.
        Returns True if buffer is at max_size after addition.
        """
        self.data.append(item)
        return self.is_full()

    def add_many(self, items: Iterable[T]) -> bool:
        """
        Add multiple items to buffer.
        Returns True if buffer is at max_size after addition.
        """
        self.data.extend(items)
        return self.is_full()

    def clear(self) -> None:
        """Remove all items from buffer"""
        self.data.clear()

    def is_full(self) -> bool:
        """Check if buffer is at max_size"""
        return len(self.data) >= self.max_size

    def get_contents(self) -> List[T]:
        """Return current contents as list"""
        return list(self.data)

# File Operations
class CSVFileHandler:
    def write(self, path: Path, data: Any, mode: str = 'a')
    def read(self, path: Path) -> pd.DataFrame
    def validate_path(self, path: Union[str, Path]) -> Path
```

### 3. Validation Framework

We originally considered changing the validation framework to be more OOP but decided that would be over engineering and will keep validations as functions in validations.py unless validations.py becomes too long.

## Code Organization

```
csv/
├── __init__.py
├── config.py           # Configuration
├── models.py           # Data models
├── manager.py          # Main CSVManager
├── components/
│   ├── __init__.py
│   ├── eeg.py         # EEG data management
│   ├── sleep.py       # Sleep stage management
│   └── merger.py      # Data merging
├── core/
│   ├── __init__.py
│   ├── buffer.py      # Buffer implementation
│   ├── validation.py  # Validation framework
│   └── file_io.py     # File operations
└── tests/
    ├── __init__.py
    ├── test_buffer.py
    ├── test_eeg.py
    ├── test_sleep.py
    └── test_merger.py
```

## Implementation Plan

### Phase 1: Core Utilities

We'll start with implementing the two core utility classes that other components will depend on. This gives us a solid foundation and lets us validate our approach early.

#### Step 1: CircularBuffer (1 day)

Required functionality:

```python
class CircularBuffer[T]:
    """
    Circular buffer implementation using collections.deque for efficient rolling buffer operations.
    When buffer reaches max_size, adding new items automatically removes oldest items.
    deque with maxlen handles this rolling behavior efficiently.
    """
    def __init__(self, max_size: int):
        """Initialize buffer with maximum size"""
        self.max_size = max_size
        self.data: Deque[T] = deque(maxlen=max_size)

    def add(self, item: T) -> bool:
        """
        Add single item to buffer.
        Returns True if buffer is at max_size after addition.
        """
        self.data.append(item)
        return self.is_full()

    def add_many(self, items: Iterable[T]) -> bool:
        """
        Add multiple items to buffer.
        Returns True if buffer is at max_size after addition.
        """
        self.data.extend(items)
        return self.is_full()

    def clear(self) -> None:
        """Remove all items from buffer"""
        self.data.clear()

    def is_full(self) -> bool:
        """Check if buffer is at max_size"""
        return len(self.data) >= self.max_size

    def get_contents(self) -> List[T]:
        """Return current contents as list"""
        return list(self.data)

Test cases to implement:
1. Basic operations:
   - Create buffer with size N
   - Add single item, verify contents
   - Add multiple items, verify contents
   - Clear buffer, verify empty

2. Size management:
   - Fill buffer to max_size with single items
   - Fill buffer to max_size with multiple items
   - Verify is_full() returns True
   - Add items when full, verify oldest items removed
   - Add multiple items when full, verify correct items remain

3. Edge cases:
   - Create buffer with size 0 or negative
   - Add None or invalid items
   - Clear empty buffer

4. Type checking:
   - Use with different types (int, float, list, etc)
   - Verify type hints work correctly
```

Tasks:

1. Create buffer.py
2. Implement CircularBuffer using collections.deque for automatic rolling buffer behavior
3. Write unit tests
4. Add documentation

#### Step 2: CSVFileHandler (1 day)

Required functionality:

```python
class CSVFileHandler:
    def write(self, path: Path, data: Any, mode: str = 'a') -> None
    def read(self, path: Path) -> pd.DataFrame
```

- Make sure you aren't adding validations, errors, or utilities that already exist in utils.py, validation.py, or errors.py or whatever the path actually is.

Tasks:

1. Create file_handler.py
2. Implement CSVFileHandler
3. Write unit tests
4. Add documentation

Success Criteria:

- All tests pass
- Code is well documented
- Type hints are complete
- Error handling is robust
- Can be used as foundation for manager classes

After completing these core utilities, we'll evaluate our approach before proceeding with the manager classes.
