AI generated

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

## Implementation Steps

We will need to implement:

- `basic_processor_implementation_example.py` - Basic 30-second window implementation
- `multi_state_processor_implementation_example.py` - Multiple hidden states implementation

Basic Requirements:

- basic_processor_implementation_example.py

  - Since are creating a sample implementation, the basic processor should take in a prerecorded chunk of sleep data that is atleast 2 minutes long. It should also take mat file that has the scores for the prerecorded data. This way we can compare the results of the streaming inference to the ground truth.
  - The file should be organized such that the main funcationality is seperate from the test data preparation and comparison.
  - This basic implementation should use basic 30 second windows with one hidden state

- multi_state_processor_implementation_example.py
  - Since are creating a sample implementation, the basic processor should take in a prerecorded chunk of sleep data that is atleast 2 minutes long. It should also take mat file that has the scores for the prerecorded data. This way we can compare the results of the streaming inference to the ground truth.
  - The file should be organized such that the main funcationality is seperate from the test data preparation and comparison.
  - This multi state implementation should use the multiple hidden states approach
  - The multi state implementation should have a shift interval of 5 seconds and the interval should be 30 seconds.
  - The multi state implementation should have 6 hidden states since we are using 5 second shifts and 30 second intervals. 5 \* 6 = 30
