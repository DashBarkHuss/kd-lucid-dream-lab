We added the folder `gssc/` because `gssc-version-6990abd04c0fc72418500e73fd1bc16242dfc8d0/` was our original folder of scripts. But that folder must be used with gssc commit 6990abd04c0fc72418500e73fd1bc16242dfc8d0. Breaking changes were made to the gssc api after that commit. That is why we have a separate folder for the new api. We use conda environemnt `gssc` for this folder, so make sure you `conda activate gssc` before running the scripts. We've updated the conda environment with the latest gssc commit in conda environment `gssc`. Conda environment `base` still uses the old gssc version.

## Checking the Commit Version of GSSC

When working with GSSC, you may need to verify which commit version is installed, especially since the version number (0.0.9) isn't regularly updated. Here's a practical approach:

### Step 1: Identify Recent Changes in the Repository

First, check what has changed in recent commits:

```bash
# Get the latest 3 commits with messages and dates
curl -s https://api.github.com/repos/jshanna100/gssc/commits | jq '.[0:3] | .[] | {sha: .sha, date: .commit.author.date, message: .commit.message}'

# Example output:
# {
#   "sha": "b0420c509a3238867150515a47730841ab28b282",
#   "date": "2024-12-16T23:04:19Z",
#   "message": "basic documentation for ArrayInfer"
# }
```

Then, examine what files were changed in a specific commit:

```bash
# Replace COMMIT_HASH with the actual hash from above
curl -s https://api.github.com/repos/jshanna100/gssc/commits/COMMIT_HASH | jq '.files[].filename'

# Example output:
# "gssc/infer.py"
# "README.md"
```

Finally, look at the actual code changes:

```bash
curl -s https://api.github.com/repos/jshanna100/gssc/commits/COMMIT_HASH | jq '.files[0].patch'
```

Take note of specific code patterns that were added or modified. For example, in December 2024 updates, the `ArrayInfer` class constructor was changed to use `use_cuda` parameter, and a `@torch.no_grad()` decorator was added.

### Step 2: Check Your Installed Version

Now that you know what to look for, check if these changes are present in your installed version:

```bash
# Find the location of your installed package
python -c "import gssc; print(gssc.__file__)"

# Check for specific code patterns identified in Step 1
grep -A 3 "def __init__" /path/to/your/environment/lib/python3.x/site-packages/gssc/infer.py

# Check for other identified patterns
grep -A 2 "@torch.no_grad()" /path/to/your/environment/lib/python3.x/site-packages/gssc/infer.py
```

### Examples of Version-Specific Code

Here are key code signatures for different versions:

**Latest Version (December 2024)**:

- ArrayInfer constructor: `def __init__(self, net=None, con_net=None, use_cuda=False, gpu_idx=None):`
- Uses `@torch.no_grad()` decorator for the `infer` method

**Older Version**:

- ArrayInfer constructor: `def __init__(self, net, con_net, sig_combs, perm_matrix, all_chans, sig_len=2560):`
- No `@torch.no_grad()` decorator

### Updating to the Latest Version

If your version is outdated:

```bash
pip cache purge
pip uninstall -y gssc
pip install git+https://github.com/jshanna100/gssc.git --force-reinstall
```

Then verify the installation by checking for the latest code patterns identified in Step 1.

## CSV Export and Validation

The system provides robust CSV export capabilities through the `CSVManager` class in the `realtime_with_restart` package which helps manage brainflow data.

### Features

- Exact format preservation for compatibility
- Comprehensive data validation
- Sleep stage data integration
- Detailed error reporting

### Usage

```python
from gssc_local.realtime_with_restart.export import CSVManager

# Initialize
csv_manager = CSVManager(board_shim)

# Save data
csv_manager.save_new_data(new_data, is_initial=True)
csv_manager.save_to_csv("output.csv")

# Add sleep stage data
csv_manager.add_sleep_stage_to_csv(sleep_stage=1.0,
                                 next_buffer_id=2.0,
                                 epoch_end_idx=100)

# Validate
csv_manager.validate_saved_csv_matches_original_source("original.csv")
```

## Gap Detection

The system provides robust gap detection capabilities through the `GapHandler` class in the `realtime_with_restart` package which helps identify discontinuities in EEG data streams. Gap detection is passed to the orchestrator class to determine how to handle gaps.

### Features

- Detects gaps between data chunks (`detect_gap()` with prev_timestamp parameter)
- Identifies gaps within data chunks (`detect_gap()` without prev_timestamp parameter)
- Validates epoch boundaries (`validate_epoch_gaps()`) TODO: Should this be in the GapHandler class or the epoch_manager class or orchestator class?
- Comprehensive error handling (via custom exceptions: `GapError`, `InvalidTimestampError`, `EmptyTimestampError`, `InvalidGapThresholdError`, `InvalidSamplingRateError`, `InvalidEpochIndicesError`)
- Configurable gap thresholds (set in `__init__()`)

### Usage

```python
from gssc_local.realtime_with_restart.core import GapHandler
import numpy as np

# Initialize with sampling rate and optional gap threshold
gap_handler = GapHandler(sampling_rate=100.0, gap_threshold=2.0)

# Detect gaps between chunks
# BrainFlow timestamps are Unix timestamps with microsecond precision (6 decimal places)
timestamps = np.array([
    1746193963.801430,  # Base timestamp
    1746193963.811430,  # 10ms later
    1746193963.821430,  # 20ms later
    1746193963.831430   # 30ms later
])
prev_timestamp = 1746193963.791430  # Previous chunk's last timestamp
has_gap, gap_size, start_idx, end_idx = gap_handler.detect_gap(timestamps, prev_timestamp)

# Detect gaps within a chunk
timestamps = np.array([
    1746193963.801430,  # Base timestamp
    1746193963.811430,  # 10ms later
    1746193963.851430,  # 50ms later (gap of 40ms)
    1746193963.861430   # 60ms later
])
has_gap, gap_size, start_idx, end_idx = gap_handler.detect_gap(timestamps)

# Validate epoch boundaries
# Create a realistic 30-second epoch (3000 samples at 100Hz)
epoch_duration = 30.0  # 30 seconds
samples_per_epoch = int(epoch_duration * 100)  # 3000 samples at 100Hz
epoch_timestamps = np.array([
    1746193963.801430 + i * 0.01  # Base timestamp + i * (1/100) seconds
    for i in range(samples_per_epoch)
])
has_gap, gap_size = gap_handler.validate_epoch_gaps(epoch_timestamps, epoch_start_idx=0, epoch_end_idx=samples_per_epoch)

# Clean up resources when done
gap_handler.cleanup()
```

## Speed-Controlled Board Implementation

The system provides a speed-controlled board implementation through the `SpeedControlledBoardManager` class in the `realtime_with_restart` package which simulates data streaming from a CSV file at configurable speeds. This is particularly useful for testing and development, allowing you to run your entire pipeline (including visualizations) at accelerated speeds - hours of recorded data can be played back in minutes.

### Features

- Configurable playback speed (e.g., 100x faster than real-time)

### Usage

```python
from gssc_local.realtime_with_restart.speed_controlled_board_manager import SpeedControlledBoardManager

# Initialize speed-controlled board with 100x speed for fast testing
board_manager = SpeedControlledBoardManager("path/to/data.csv", speed_multiplier=100.0)
board_manager.setup_board()

# Start streaming
board_manager.start_stream()

# Process data in chunks
while True:
    new_data = board_manager.get_new_data()
    if new_data.size == 0:
        break
    # Process your data here
```

## Complete Implementation Example

For a complete working example of using the speed-controlled board, see `gssc_local/realtime_with_restart/main_speed_controlled_stream.py`. This script demonstrates:

1. A simpler, single-process implementation (unlike the real board which requires StreamManager)
2. How to integrate the speed-controlled board with the full data processing pipeline
3. How to handle visualization updates during playback
4. How to use the speed-controlled board for testing and development

Key differences from the real board implementation:

- No need for multiprocessing or StreamManager
- Direct data processing in the main process
- Simpler gap handling (if needed)
- Easier debugging and development

The script can be run directly:

```bash
python gssc_local/realtime_with_restart/main_speed_controlled_stream.py
```

This makes it ideal for:

- Testing the full pipeline without hardware
- Development and debugging
- Running long recordings at accelerated speeds
- Testing visualization updates
