# kd-lucid-dream-lab

A Python-based toolkit for working with OpenBCI and Brainflow for EEG data processing and analysis, with a focus on sleep stage classification using the Greifswald Sleep Stage Classifier (GSSC).

## Table of Contents

- [Usage](#usage)
  - [Streaming from OpenBCI GUI to Brainflow](#how-to-stream-from-openbci-gui-to-brainflow)
  - [Plotting Data with Python](#how-to-plot-data-from-brainflow-with-python)
  - [Filtering and Plotting Data](#how-to-filter-and-plot-data-from-brainflow)
  - [Data Playback](#how-to-playback-data-from-a-file)
- [Real-Time Inference with GSSC](#real-time-inference-with-gssc-greifswald-sleep-stage-classifier)
- [Development](#development)

## Usage

## TBD: How to auto sleep score in realtime OpenBCI stream

## How to stream from OpenBCI GUI to Brainflow

1. Open OpenBCI GUI
2. System Control Panel -> Ganglion Live
3. Select the following settings:
   1. Pick Transfer Protocol: BLED112 Dongle
   2. BLE Device: select your ganglion board
   3. Session Data: BDF+
   4. Brainflow Streamer: Choose "Network", Set IP address to 225.1.1.1, Set port to 6677
4. Start Session
5. Start Data Stream
6. If you want the value in the OpenBCI GUI to match the value in the script, you need to remove the filters.
   1. Click on the filters icon in the top left corner of the OpenBCI GUI
   2. Click "All" to turn the channel icons black which means no filters are applied
7. In a new terminal, run the script `demo_files/stream-from-openbci-GUI.py`

In the terminal you'll see the samples ploted out in a vertical stream.

<div>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <p>Stream to brainflow demo - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/7c4b133287134a08a924a850928adf90-866d7ca113c60d57-full-play.gif">
    </a>
  </div>

## How to plot data from Brainflow with python

1. Follow steps 1-5 in the "How to stream from OpenBCI GUI to Brainflow" section
2. Run the script `demo_files/plot_python_example.py`

<div>
    <a href="https://www.loom.com/share/7783bd62fe1b4170b1a8eb21419f26f6">
      <p>python - BrainFlow Plot  - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/7783bd62fe1b4170b1a8eb21419f26f6">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/7783bd62fe1b4170b1a8eb21419f26f6-54ac665bd899918e-full-play.gif">
    </a>
  </div>

## How to filter and plot data from Brainflow

1. Follow steps 1-5 in the "How to stream from OpenBCI GUI to Brainflow" section
2. Run the script `demo_files/filtered_plot.py`

<div>
    <a href="https://www.loom.com/share/ee904b05f8484db69ce555ad0e6a11c5">
      <p>filtered python plot brainflow - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/ee904b05f8484db69ce555ad0e6a11c5">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/ee904b05f8484db69ce555ad0e6a11c5-fd57ab856caa8c2d-full-play.gif">
    </a>
  </div>

## How to playback data from a file

### Recording Data

You need to use the BrainFlow-RAW.csv file to play from the file

1. Open OpenBCI GUI
2. System Control Panel -> Ganglion Live
3. Setting to record data:
4. Pick Transfer Protocol: BLED112 Dongle
5. BLE Device: select your ganglion board
6. Session Data: OpenBCI (I don't know if this actually matter, but it will cause an error the current GUI if you choose BDF+)
7. Brainflow Streamer: File
8. Start Session
9. Start Data Stream
10. Record your EEG/ECG/etc data
11. When done recording your session: Stop Data Stream
12. Stop Session
13. OpenBCI should have saved the session recording as `BrainFlow-RAW_<whatever you put in the session data name>.csv`

### Playback Data

1. Move the BrainFlow CSV file to the same folder as this script.
2. Write your .csv's path into `demo_files/playback_from_file.py`
3. Run the script, and you should see the data you previosly recorded show up in the plot.

# Real-Time Inference with GSSC (Greifswald Sleep Stage Classifier)

This document outlines different approaches for implementing real-time inference with the Greifswald Sleep Stage Classifier (GSSC).

## Background

GSSC is designed to process sleep EEG/EOG data in 30-second epochs, which presents unique challenges for real-time applications. Below are several implementation strategies, each with their own advantages and trade-offs.

## Implementation Approaches

### 1. Basic 30-Second Window Processing

The simplest approach that processes data in non-overlapping 30-second windows.

**Pros:**

- Simplest implementation
- Follows the model's expected input pattern
- Maintains temporal consistency
- Most accurate predictions

**Cons:**

- Only provides updates every 30 seconds
- High latency for initial prediction

### 2. Overlapping Windows with Single Hidden State

This approach processes overlapping windows to provide more frequent updates.

**Pros:**

- More frequent updates (configurable shift interval)
- Reuses existing data for faster feedback
- Lower latency than basic approach

**Cons:**

- May reduce accuracy due to hidden state inconsistency
- Doesn't match the model's training pattern
- Introduces redundancy in processing

### 3. Multiple Hidden States (Round-Robin Approach)

This approach maintains separate hidden states for different time-shifted streams of the same data.

**Pros:**

- More frequent updates (configurable shift interval)
- Maintains temporal consistency for each stream
- Preserves the model's expected input pattern
- Better accuracy than single hidden state with overlapping windows

**Cons:**

- More complex implementation
- Higher memory usage (multiple hidden states)
- Initial delay still required for first prediction

## Data Preprocessing Requirements

All approaches require the following preprocessing steps:

1. **Filtering:**

   - Bandpass filter: 0.3-30Hz (GSSC was trained on this range)

2. **Channel Selection:**

   - EEG: Use C3, C4, F3, or F4
   - EOG: HEOG

3. **Resampling:**

   - Each 30-second window must be resampled to 2560 samples

4. **Normalization:**
   - Z-scoring recommended for optimal performance

## Performance Considerations

- **GPU Acceleration:**

  - Using CUDA significantly improves processing speed
  - Recommended for real-time applications

- **CPU-Only Systems:**
  - Limit the number of channels to improve performance
  - Model inference typically takes milliseconds
  - Main bottleneck is the 30-second data collection requirement

## Recommendations

### Clinical Applications

- Use Basic 30-Second Window Processing
- Prioritize accuracy over update frequency
- Ensure proper preprocessing and filtering

### Monitoring Applications

- Use Multiple Hidden States approach
- Configure shift interval based on needs (e.g., 5-10 seconds)
- Balance update frequency with computational resources

### Research/Development

- Experiment with different approaches
- Compare accuracy vs. update frequency trade-offs
- Consider implementing multiple approaches for comparison

## Implementation Example

For implementation details and code examples, refer to the following files in this repository:

- `basic_processor.py` - Basic 30-second window implementation
- `overlapping_processor.py` - Overlapping windows implementation
- `multi_state_processor.py` - Multiple hidden states implementation
