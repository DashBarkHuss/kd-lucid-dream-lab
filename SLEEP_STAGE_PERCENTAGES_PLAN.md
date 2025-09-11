# Plan: Add Sleep Stage Percentages to CSV Output

## Current Analysis

**Current Output Format:**
- `timestamp_start`, `timestamp_end`, `sleep_stage`, `buffer_id`
- Only shows the final predicted sleep stage (0-4 for Wake, N1, N2, N3, REM)

**Available Data:**
- GSSC model returns `class_probs` - a numpy array with probabilities for each sleep stage
- 5 sleep stages: Wake (0), N1 (1), N2 (2), N3 (3), REM (4)
- `class_probs` shape: [n_combinations, 5] where probabilities sum to 1.0 across stages

## Proposed Enhanced CSV Format

**New columns to add:**
- `wake_percent` - Percentage probability for Wake stage
- `n1_percent` - Percentage probability for N1 stage  
- `n2_percent` - Percentage probability for N2 stage
- `n3_percent` - Percentage probability for N3 stage
- `rem_percent` - Percentage probability for REM stage

**Final format:**
```csv
timestamp_start,timestamp_end,sleep_stage,buffer_id,wake_percent,n1_percent,n2_percent,n3_percent,rem_percent
1755354827.6107,1755354857.605726,0,0,85.2,5.1,7.3,1.8,0.6
```

## Implementation Approach

### 1. Modify EpochResult NamedTuple
- Add `class_probabilities` field to store the probability array
- Update `batch_processor.py:27-33`

### 2. Update BatchProcessor._process_all_epochs()
- Capture `class_probs` from `stateful_manager.process_epoch()`
- Pass probabilities to `EpochResult` constructor
- Update `batch_processor.py:449-460`

### 3. Enhance BatchProcessor._save_results()
- Extract percentage values from `class_probabilities`
- Convert probabilities to percentages (multiply by 100)
- Add new columns to DataFrame before saving
- Update `batch_processor.py:462-475`

### 4. Always Include Percentages
- Always include percentage columns in CSV output
- No command line flag needed - percentages provide valuable information
- Simpler implementation without optional behavior

## Key Design Decisions

1. **Always Include Percentages**: Percentages provide valuable information with no downside
2. **Percentage Format**: Convert 0.852 → 85.2 for readability
3. **Precision**: Use 1 decimal place for percentages (sufficient for sleep research)
4. **Column Naming**: Clear, consistent naming following existing patterns
5. **Data Flow**: Minimal changes to existing processing pipeline

## Files to Modify

1. `sleep_scoring_toolkit/batch_processor.py` - Core logic changes

## Testing Strategy

- Test with existing data to ensure no regressions
- Verify percentages sum to ~100% (accounting for rounding)
- Confirm final sleep stage matches highest percentage
- Test with percentage columns included

## Implementation Steps

- [x] 1. Modify EpochResult NamedTuple to include class_probabilities
- [x] 2. Update BatchProcessor._process_all_epochs() to capture class probabilities  
- [x] 3. Enhance BatchProcessor._save_results() to include percentage columns
- [x] 4. Always include percentages (no flag needed)
- [x] 5. Test the implementation with existing data

## ✅ Implementation Complete!

**Sample Output:**
```csv
timestamp_start,timestamp_end,sleep_stage,buffer_id,wake_percent,n1_percent,n2_percent,n3_percent,rem_percent
1755354827.6107,1755354857.605726,0,0,90.9,6.0,0.0,0.0,3.1
1755354857.61383,1755354887.608607,0,1,94.0,3.9,0.0,0.0,2.1
```

The feature successfully adds sleep stage percentage probabilities to the CSV output, providing valuable insight into the model's confidence and decision-making process.