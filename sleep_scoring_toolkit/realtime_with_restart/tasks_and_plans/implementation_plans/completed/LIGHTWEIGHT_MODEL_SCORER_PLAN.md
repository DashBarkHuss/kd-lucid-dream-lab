# Lightweight Model Scorer Implementation Plan (Corrected)

## Overview

This document outlines the implementation strategy for a fast, headless batch processor that generates sleep stage predictions from EEG CSV files without the GUI and real-time overhead of the current system. This addresses Part 1 of the `EPOCH_SCORE_COMPARISON_PLAN.md`.

**Key Correction**: This plan now uses **sequential processing of all epochs** (not targeted extraction) to maintain hidden state continuity and generate complete model prediction datasets.

## Problem Statement

The current main scripts (`main_speed_controlled_stream.py`, `main_multiprocess.py`) take hours to run due to:

- Real-time simulation with timing delays
- PyQt GUI visualization updates
- Complex round-robin buffer management (6 buffers with 5-second offsets)
- Event dispatching and visualization overhead
- Memory management for streaming scenarios

**Goal**: Create a tool that processes hours of EEG data in minutes for complete model scoring.

## Architecture Comparison

### Current System (Complex Real-Time)

```
CSV → BrainFlow Simulation → Round-Robin Buffers → Real-Time Delays → GUI Updates → Model Inference → Visualization
```

**Components:**
- `SpeedControlledBoardManager`: Simulates real-time streaming
- `DataManager`: Complex buffer management with 6 buffers
- `ETDBufferManager`: Memory-efficient streaming buffers
- `PyQtVisualizer`: Real-time GUI updates
- Gap detection and buffer state management
- Event dispatching system

### Proposed System (Sequential Epoch Processing)

```
CSV Data → Gap Detection & Fail-Fast → Sequential Epoch Processing → Continuous Hidden State → Model Inference → Complete CSV Output
```

**Components:**
- **Direct CSV loading** - No simulation overhead
- **Gap detection with fail-fast** - Reject files with gaps >2s for comparison validity
- **Sequential epoch processing** - Process all epochs continuously (no gaps = no resets)
- **Continuous hidden state** - No resets needed since gaps cause processing failure
- **Complete model predictions** - Every epoch processed and saved

## Current Code State Analysis

✅ **Already Completed by Refactoring:**
- `core/epoch_inference.py` - Shared epoch inference logic extracted from DataManager
- `stateful_inference_manager.py` - Clean stateful inference manager 
- `batch_processing.py` - CSV extraction utilities for epoch processing
- Core infrastructure is in place and working

## Implementation Strategy

### Phase 1: Audit Existing Validation Functions

**Before writing new validation code, identify reusable validation logic:**

**Locations to audit:**
- `realtime_with_restart/export/csv/validation.py` - CSV data validation functions
- `realtime_with_restart/core/gap_handler.py` - Gap detection and validation
- `realtime_with_restart/data_manager.py` - Epoch validation in `_process_epoch()` method
- `batch_processing.py` - Existing epoch extraction validation
- `sleep_scoring_toolkit/tests/test_*.py` - Test validation utilities

**Specific validation functions to look for:**
- **Gap detection/validation**: Size thresholds, gap handling logic
- **Epoch boundary validation**: Sample count checks, duration validation
- **CSV format validation**: Data shape, timestamp validation, BrainFlow format checks
- **Sequential continuity**: Timestamp monotonicity, interval validation

**Document findings:** Create inventory of existing validation functions to reuse vs. what needs to be implemented

#### ✅ **Validation Audit Results**

**Existing Validation Functions We Can Reuse:**

1. **CSV Data Validation** (`realtime_with_restart/export/csv/validation.py`)
   - `validate_data_shape()` - 2D array validation, NaN/infinite checks
   - `validate_brainflow_data()` - BrainFlow format validation
   - `validate_timestamps_unique()` - Duplicate timestamp detection
   - `validate_file_path()` - File path validation

2. **Gap Detection & Validation** (`realtime_with_restart/core/gap_handler.py`)
   - `GapHandler.detect_gap()` - Core gap detection with thresholds
   - `GapHandler.validate_epoch_gaps()` - Epoch-specific gap validation
   - `GapHandler._validate_timestamps()` - Timestamp array validation (monotonicity, NaN checks)

3. **Epoch Validation** (Multiple files)
   - `data_manager._validate_epoch_duration()` - 30-second duration validation (0.1s tolerance)
   - **Sample count validation** - Present in `processor.py` and multiple test files
   - **`points_per_epoch`** calculation - Consistent across codebase (30 * sampling_rate)

4. **Test Utilities** (`tests/test_utils.py`)
   - `create_brainflow_test_data()` - Generate BrainFlow-compatible test data
   - **Timestamp validation utilities** - Format and interval validation

5. **Sampling Rate Validation** (`realtime_with_restart/utils/timestamp_utils.py`)
   - `validate_sample_rate()` - Validates sample rate within tolerance
   - `validate_inter_batch_sample_rate()` - Validates between batches

**Functions We Need to Create:**

1. **Gap Fail-Fast for Comparison Validity** (Only new function needed!)
   ```python
   def validate_no_large_gaps_for_comparison(detected_gaps, threshold=2.0):
       """Fail fast if gaps would affect model-researcher comparison validity"""
   ```
   
   **Implementation Note**: This function was implemented as `BatchProcessor._validate_gap_free_for_inference(timestamps, max_gap_seconds=2.0)` with an improved interface that takes timestamps directly rather than requiring pre-computed gaps. The functionality is identical but the implementation is more efficient.

**Validation Timing Clarification:**

**Once Per File (Upfront - Before Any Processing):**
- File path validation
- CSV format and BrainFlow data validation
- Complete timestamp array validation (monotonicity, duplicates, NaN)
- Sampling rate validation across entire dataset
- Gap detection and fail-fast validation

**Per Epoch (During Sequential Processing):**
- Individual epoch duration validation (30s ± 0.1s tolerance)
- Individual epoch sample count validation
- Epoch boundary timestamp validation

**Reusable Architecture Patterns:**
- **Fail-fast approach** - Raise specific exceptions with clear messages
- **Tolerance-based validation** - Allow small deviations (e.g., 0.1s, 50 samples)
- **Consistent logging** - Error messages with context
- **Composable functions** - Small, focused validation functions that can be combined

### Phase 2: Sequential Processing Workflow Design

**Core data flow with complete epoch processing:**

1. **Upfront validation** (once per file):
   - **File path validation**: Verify input file exists (existing)
   - **Load CSV data** and perform format validation
   - **BrainFlow data validation**: Check data format and shape (existing)  
   - **Timestamp validation**: Monotonicity, no duplicates, NaN checks (existing)
   - **Sampling rate validation**: Verify expected sample rate (existing)
   - **Gap detection + fail-fast**: Detect gaps and fail immediately if any >2s (existing + new)

2. **Generate epoch timestamps**: 30-second intervals from CSV start to end
3. **Sequential processing** (no hidden state resets since no gaps allowed):
   - Extract each epoch sequentially
   - Per-epoch validation (duration, sample count)
   - Process through continuous hidden state
4. **Save ALL model predictions** using existing CSV format

### Phase 3: BatchProcessor Implementation

**File:** `sleep_scoring_toolkit/batch_processor.py`

**Epoch Timestamp Generation:**
```python
def generate_epoch_timestamps(self, csv_start_time, csv_end_time):
    """Generate sequential 30-second epoch boundaries from CSV start to end.
    
    Args:
        csv_start_time: First timestamp in CSV data
        csv_end_time: Last timestamp in CSV data
        
    Returns:
        List of (start_time, end_time) tuples for each 30-second epoch
    """
    epoch_timestamps = []
    current_time = csv_start_time
    epoch_duration = 30.0  # 30 seconds
    
    while current_time + epoch_duration <= csv_end_time:
        start_time = current_time
        end_time = current_time + epoch_duration
        epoch_timestamps.append((start_time, end_time))
        current_time += epoch_duration  # Next epoch starts immediately after
    
    return epoch_timestamps
```

**Core workflow:**
```python
def process_csv_file(self, csv_path, output_path):
    # Phase 1: Upfront validation (once per file)
    validate_file_path(csv_path)  # Existing
    csv_data, timestamps = load_csv_data(csv_path)
    validate_brainflow_data(csv_data)  # Existing
    validate_timestamps_unique(timestamps)  # Existing 
    validate_sample_rate(timestamps, self.sampling_rate)  # Existing
    
    # Phase 2: Gap detection with fail-fast (combined step)
    gaps = self.gap_handler.detect_gaps(timestamps)  # Existing
    validate_no_large_gaps_for_comparison(gaps)  # New - fail if gaps > 2s
    
    # Phase 3: Generate all epoch timestamps (sequential 30s intervals)
    epoch_timestamps = self.generate_epoch_timestamps(timestamps[0], timestamps[-1])
    
    # Phase 4: Sequential processing (no hidden state resets - continuous processing)
    results = []
    for i, (start_time, end_time) in enumerate(epoch_timestamps):
        # Extract epoch with per-epoch validation
        epoch_data = extract_epoch_from_csv(csv_path, start_time)  # Existing
        validate_epoch_duration(epoch_data, expected_duration=30.0)  # Existing - per epoch
        
        # Process through continuous hidden state (no resets since no gaps)
        prediction = self.stateful_manager.process_epoch(epoch_data)  # Existing
        results.append(EpochResult(start_time, end_time, prediction, i))
    
    # Phase 5: Save all results using existing CSV format
    self.save_results(results, output_path)  # Reuse existing CSVManager format
```

**Key components:**
- Reuse ~90% existing validation functions (only need 1 new function)
- Use existing `GapHandler` for gap detection + new fail-fast validation
- Use existing `StatefulInferenceManager` for hidden state continuity
- Use existing `extract_epoch_from_csv()` for epoch extraction
- Use existing CSV output format for compatibility

### Phase 4: CLI Implementation

**File:** `sleep_scoring_toolkit/generate_model_scores.py`

```python
def main():
    parser = argparse.ArgumentParser(description='Generate complete sleep stage predictions from EEG CSV')
    parser.add_argument('--input', required=True, help='Input EEG CSV file (BrainFlow format)')
    parser.add_argument('--output', required=True, help='Output sleep stages CSV file')
    parser.add_argument('--montage', default='minimal_sleep_montage', help='Montage configuration')
    parser.add_argument('--sampling-rate', type=int, default=None, help='Sampling rate (Hz)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Initialize batch processor
    montage = get_montage_by_name(args.montage)
    processor = BatchProcessor(
        montage=montage, 
        sampling_rate=args.sampling_rate,
        show_progress=not args.quiet
    )

    # Process all epochs sequentially
    results = processor.process_csv_file(args.input, args.output)
    
    # Report results
    logger.info(f"Processing completed: {results['epochs_processed']} epochs")
    logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
    logger.info(f"Output saved to: {args.output}")
```

## Technical Specifications

### Input Requirements
- **Format**: BrainFlow CSV format with timestamp and EEG channels
- **Channels**: Must contain the channels required by the selected montage
- **Sampling Rate**: Configurable (default: uses BoardShim.get_sampling_rate())
- **Duration**: Any length (will be segmented into 30-second epochs)

### Output Format

**Complete Sleep Stage CSV:**
```csv
timestamp_start,timestamp_end,sleep_stage,buffer_id
1755354827.610700,1755354857.540740,0,0
1755354857.540740,1755354887.470780,1,1
1755354887.470780,1755354917.400820,2,2
```

### Gap Handling Strategy

**Fail-fast approach for model-researcher comparison validity:**

**Problem**: Model processing is gap-aware (resets hidden states after gaps) while researcher scoring is gap-unaware (assumes continuous temporal context). This creates systematic prediction differences after gaps that invalidate comparisons.

**Solution**: Fail-fast validation to ensure comparison validity:
1. **Detect gaps** in CSV timestamps using existing GapHandler
2. **Fail immediately** if any gaps > 2 seconds are detected
3. **Provide clear error messaging** explaining the comparison validity issue
4. **Recommend using continuous data** without gaps for comparison analysis

**Key Point**: Since we fail-fast on gaps, there are **no hidden state resets needed** during processing - the data is guaranteed to be continuous.

```python
def validate_no_large_gaps_for_comparison(detected_gaps, threshold=2.0):
    """Fail fast if gaps would affect model-researcher comparison validity."""
    large_gaps = [gap for gap in detected_gaps if abs(gap['size']) > threshold]
    
    if large_gaps:
        raise ValueError(
            f"Found {len(large_gaps)} significant gaps in data. "
            "Gaps > 2 seconds would cause hidden state resets, making model-researcher "
            "comparison invalid. Please use continuous data without gaps."
        )
```

**Benefits:**
- **Ensures comparison validity** - Model and researcher operate under same temporal assumptions
- **Research quality control** - Alerts to methodological issues with gapped data scoring
- **Clear error messaging** - Users understand why processing failed and how to fix it
- **Future flexibility** - Can add `--allow-gaps` flag later for non-comparison use cases

### Performance Targets

- **Speed**: Process 1 hour of data in < 2 minutes (no GUI/real-time overhead)
- **Memory**: Handle large files (several GB) efficiently
- **Completeness**: Generate predictions for every processable 30-second epoch
- **Accuracy**: Identical predictions to real-time system (same gap handling)

## Usage Examples

### Basic Usage
```bash
python sleep_scoring_toolkit/generate_model_scores.py \
    --input data/realtime_inference_test/8-16-25/BrainFlow-RAW_2025-08-16_04-33-14_0.csv \
    --output sleep_stages_2025-08-16_model_predictions.csv
```

### With Custom Configuration
```bash
python sleep_scoring_toolkit/generate_model_scores.py \
    --input data/eeg_recording.csv \
    --output model_predictions.csv \
    --montage eog_only_montage \
    --sampling-rate 500 \
    --quiet
```

## Integration with Comparison Analysis

The output CSV format is designed to be directly compatible with researcher scoring comparison:

```bash
# Step 1: Generate complete model predictions (this tool)
python sleep_scoring_toolkit/generate_model_scores.py \
    --input data/BrainFlow-RAW_2025-08-16_04-33-14_0.csv \
    --output sleep_stages_model_predictions.csv

# Step 2: Compare with researcher scores (separate tool)
python sleep_scoring_toolkit/score_comparison_viewer.py \
    --model-scores sleep_stages_model_predictions.csv \
    --researcher-scores data/8-16-25_scoring.mat \
    --recording-start-timestamp 1755354827.610700
```

## Benefits of Sequential Processing Approach

- **🎯 Accurate predictions**: Maintains temporal context through continuous hidden state (no resets)
- **📊 Complete dataset**: Every processable epoch included for comprehensive analysis
- **⚡ Speed improvement**: No GUI/real-time delays (hours → minutes)
- **🔄 Consistency**: Same inference logic as real-time system, but simplified (no gap handling)
- **🧩 Reuses proven components**: Leverages ~90% existing validation and processing logic
- **📈 Analysis ready**: Output directly compatible with comparison and visualization tools
- **🛡️ Robust validation**: Fail-fast approach ensures comparison validity
- **🚀 Simplified architecture**: No complex buffer management or hidden state resets

## Implementation Timeline (Revised)

- **Phase 1** (Validation Audit): ✅ **Completed**
- **Phase 2** (Workflow Design): ✅ **Completed**
- **Phase 3** (BatchProcessor): 1.0 days (simplified due to code reuse)
- **Phase 4** (CLI Implementation): 0.5 days
- **Testing & Integration**: 0.5 days

**Total**: ~2.0 days implementation time (reduced from 3.5 days due to extensive code reuse)