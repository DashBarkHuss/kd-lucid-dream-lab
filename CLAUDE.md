# CLAUDE.md

⚠️ **IMPORTANT: When asked "does this follow CLAUDE.md?", systematically check ALL sections below, especially Project-Specific Implementation Notes**

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python toolkit for working with OpenBCI hardware and Brainflow for EEG data processing and analysis. The project focuses on sleep stage classification using the Greifswald Sleep Stage Classifier (GSSC). It includes functionality for real-time data acquisition, processing, and visualization, as well as tools for handling gaps in data streams and managing CSV data export.

## Environment Setup

The project uses Conda for environment management. The main environment for the project is called `gssc`.

```bash
# Activate the gssc environment
conda activate gssc
```

### ⚠️ CRITICAL: Always Verify Environment Before Testing

**ALWAYS check you're in the correct environment before running tests or commands!** The GSSC dependencies and APIs are only available in the `gssc` environment.

```bash
# Check current environment (gssc should have * next to it)
conda info --envs

# If not in gssc environment, use persistent activation for commands:
source ~/anaconda3/etc/profile.d/conda.sh && conda activate gssc && [your-command]
# Or if using miniconda:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate gssc && [your-command]
```

**Common mistake**: Running tests in `base` environment will cause `ArrayInfer API` errors and other dependency issues.

## Common Commands

### Running Tests

```bash
# Run all tests
python -m pytest sleep_scoring_toolkit/tests/

# Run a specific test file
python -m pytest sleep_scoring_toolkit/tests/test_csv_manager.py

# Run tests with verbose output and -s to see print statements
CI=true python -m pytest sleep_scoring_toolkit/tests/ -v -s

```

### Running the Application

```bash
# Run the main speed-controlled board stream simulation
python sleep_scoring_toolkit/realtime_with_restart/main_speed_controlled_stream.py

# Run demo files (examples of how to use the library)
python demo_files/stream-from-openbci-GUI.py
python demo_files/python_plot_example.py
python demo_files/filtered_plot.py
python demo_files/playback_from_file.py
```

## Project Structure

### Core Components

1. **Board Managers**

   - `BoardManager`: Manages BrainFlow BoardShim API, board setup, session management, stream control, and data collection
   - `SpeedControlledBoardManager`: Inherits from `BoardManager`, simulates data streaming from a CSV file at configurable speeds for testing

2. **Data Management**

   - `DataManager`: Manages data buffers and orchestrates processing through visualizations, epoch management, and sleep classification
   - `CSVManager`: Handles CSV data export and validation for BrainFlow data

3. **Signal Processing**

   - `SignalProcessor`: Processes EEG signals for sleep stage classification
   - `StatefulInferenceManager`: Manages stateful sleep stage inference with persistent hidden states across epochs
   - `GapHandler`: Detects and reports gaps in BrainFlow EEG data streams

4. **Visualization**
   - `PyQtVisualizer`: Visualizes EEG data and sleep stage classification using PyQt

### Key Directories

- `/sleep_scoring_toolkit/`: Contains the main project code
  - `/realtime_with_restart/`: Core functionality for real-time stream processing
  - `/tests/`: Test files for the project
- `/demo_files/`: Example scripts demonstrating BrainFlow usage with OpenBCI data streams
- `/proof_of_concepts/`: Isolated implementations for testing specific features before integration
- `/data/`: Directory for storing data files
  - `/data/realtime_inference_test/8-16-25/`: Cyton-Daisy OpenBCI sleep data (good for testing) including Karen Konkoly's scoring file

## Architecture

The system follows a component-based architecture where:

1. A `BoardManager` manages the BrainFlow BoardShim API and obtains data from OpenBCI hardware or simulation
2. The `DataManager` processes the data in buffers and orchestrations the processing of the data through epoch, processing, sleep scoring, and visualization
3. `GapHandler` detects and handles discontinuities in the data
4. `SignalProcessor` applies the GSSC model for sleep stage classification
5. `PyQtVisualizer` displays the processed data and classification results
6. `CSVManager` exports data to CSV files for storage

Data flows through these components in a pipeline architecture, with each component responsible for a specific aspect of the processing.

## Terminology

- **ETD**: Electrode and Timestamp Data - the combined dataset containing both EEG channel data and timestamp information
- **EXG**: Electrophysiological channels (EEG, EMG, ECG, etc.) - the raw signal channels from OpenBCI hardware

## Important Notes

1. The GSSC model requires 30-second epochs of EEG data for sleep stage classification
2. Real-time inference approaches:

   - **Multiple Hidden States (Round-Robin Approach)** - ✅ **Currently implemented** - Uses 6 buffers processing epochs at 5-second offsets
   - Basic 30-Second Window Processing - 📋 Planned but not implemented
   - Overlapping Windows with Single Hidden State - 📋 Planned but not implemented

3. CSV data management has been updated with new buffer management logic:
   - `save_new_data_to_csv_buffer()` -> `add_data_to_buffer()`
   - `add_sleep_stage_to_csv_buffer()` -> `add_sleep_stage_to_sleep_stage_csv()`
   - `save_to_csv()` -> `save_all_data()` followed by `cleanup()`

### Project-Specific Implementation Notes

- Use the existing `sleep_scoring_toolkit` module structure and follow the established patterns for EEG data processing components
- When working with BrainFlow data, always consider the 30-second epoch requirements for GSSC model compatibility
- **Always get board configuration using BoardShim API methods, never hardcode board-specific configuration values** - Use `BoardShim.get_eeg_channels()`, `BoardShim.get_eog_channels()`, `BoardShim.get_sampling_rate()`, `BoardShim.get_timestamp_channel()`, etc. instead of hardcoding channel numbers, sampling rates, or other board-specific configuration values. Using BrainFlow API enums like `BoardIds.CYTON_DAISY_BOARD` is correct because it uses the BrainFlow API, especially when testing a specific board like in tests. However, using a board ID as a default value in live code violates the fail-fast principle and should be avoided. The rule applies to configuration values like `125` for sampling rate or `[1, 2, 3, 4]` for channel numbers, not to proper API usage.

## Testing Principles

- **Infrastructure tests**: Use simple, valid data formats. Sine waves, random data, or basic patterns are sufficient when testing system plumbing, buffer management, API contracts, etc.
- **Integration tests**: Use real data when testing domain-specific behavior, model accuracy, or end-to-end workflows with actual user data.
- **Format realistic vs Content realistic**:
  - Format realistic = correct shapes, types, structure (usually sufficient)
  - Content realistic = actual domain patterns/values (only needed for domain logic validation)
- **Always ask**: "What type of failure should this test catch?" before choosing test data approach
- **Default to simplicity**: Can existing test utilities handle this? What's the minimal change needed?
- 1 is refering to the hardcodings. \
  \
  let's do #2 and #3
