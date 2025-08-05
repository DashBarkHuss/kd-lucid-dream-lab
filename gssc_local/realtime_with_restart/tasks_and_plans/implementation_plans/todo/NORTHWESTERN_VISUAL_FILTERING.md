# Northwestern Visual Filtering Implementation Plan

## Overview
Implement channel-specific visual filtering matching Northwestern University's clinical PSG standards. This includes bandpass filters (0.3-100 Hz for EEG/EOG, 10-100 Hz for EMG, etc.) plus 50/60 Hz notch filters for power line noise removal, while keeping GSSC processing pipeline separate and unaffected.

## Background

### Problem
- Current montage system incorrectly couples filter settings with channel configuration
- No visual filtering applied to EEG display - shows raw, noisy data
- Northwestern colleagues recommend clinical-standard filtering for professional PSG visualization

### Solution
- Decouple filtering from montage (separation of concerns)
- Implement Northwestern's 16-channel clinical filter configuration
- Apply filtering only to visual display, not GSSC sleep classification
- Use channel-specific filters rather than type-based for maximum flexibility

## Northwestern Filter Configuration

Based on Northwestern University clinical PSG standards:

```python
# Channel-specific bandpass filters
NORTHWESTERN_BANDPASS_FILTERS = {
    # EEG Channels (0.3-100 Hz)
    1: (0.3, 100.0),   # F3 - Frontal Left
    2: (0.3, 100.0),   # F4 - Frontal Right  
    3: (0.3, 100.0),   # C3 - Central Left
    4: (0.3, 100.0),   # C4 - Central Right
    5: (0.3, 100.0),   # O1 - Occipital Left
    6: (0.3, 100.0),   # O2 - Occipital Right
    7: (0.3, 100.0),   # T3/T5 - Temporal Left
    8: (0.3, 100.0),   # T4/T6 - Temporal Right
    
    # EOG Channels (0.3-100 Hz)
    9: (0.3, 100.0),    # ROC - Right Outer Canthus
    10: (0.3, 100.0),   # LOC - Left Outer Canthus
    11: (0.3, 100.0),   # R-EOG - Right EOG
    12: (0.3, 100.0),   # L-EOG - Left EOG
    
    # EMG Channels (10-100 Hz)  
    13: (10.0, 100.0),  # EMG1 - Chin EMG
    14: (10.0, 100.0),  # EMG2 - Leg EMG
    
    # Respiratory/Other Channels
    15: (0.3, 50.0),    # Airflow/Respiratory (lower high-cutoff)
    16: (10.0, 100.0)   # Snoring/Audio (higher low-cutoff)
}

# Power line notch filters (applied to ALL channels)
NORTHWESTERN_NOTCH_FILTERS = [50.0, 60.0]  # European + US power line frequencies
```

### Filter Rationale
- **EEG (0.3-100 Hz):** Captures all brain activity while removing DC drift
- **EOG (0.3-100 Hz):** Same as EEG, preserves eye movement dynamics
- **EMG (10-100 Hz):** Higher low-cutoff removes slow artifacts, preserves muscle activity
- **Respiratory (0.3-50 Hz):** Lower high-cutoff reduces noise, preserves breathing patterns
- **Snoring (10-100 Hz):** Higher low-cutoff for acoustic/vibrational signals
- **Notch (50+60 Hz):** Removes European and US power line interference

## Implementation Phases

### Phase 1: Decouple Filters from Montage

#### 1.1 Clean Up ChannelConfig
```python
@dataclass
class ChannelConfig:
    label: str
    location: str
    # filter_range: tuple[float, float]  # REMOVE THIS LINE
    channel_type: str  # 'EEG', 'EOG', 'EMG', 'OTHER'
    board: str  # 'CYTON_DAISY' or 'CYTON' or 'GANGLION'
    channel_number: int  # 1-16
```

#### 1.2 Remove Filter Methods from Montage
- Remove `get_filter_ranges()` method from Montage class
- Remove all `filter_range=` parameters from channel definitions

#### 1.3 Update Visualizer Classes
- Remove `self.filter_ranges = self.montage.get_filter_ranges()` from:
  - `PyQtVisualizer.__init__()`
  - Other visualizer classes

### Phase 2: Implement Northwestern Filtering System

#### 2.1 Add Filter Configuration to PyQtVisualizer
```python
class PyQtVisualizer:
    def __init__(self, ...):
        # ... existing code ...
        
        # Visual filtering configuration
        self.visual_filter_enabled = True  # Simple code parameter for testing
        
        # Northwestern clinical filter standards
        self.northwestern_bandpass_filters = {
            # ... filter configuration from above ...
        }
        self.northwestern_notch_filters = [50.0, 60.0]
```

#### 2.2 Core Filter Implementation Methods

```python
def apply_bandpass_filter(self, data, low_freq, high_freq, sampling_rate):
    """Apply 4th-order Butterworth bandpass filter with zero-phase filtering"""
    from scipy.signal import butter, filtfilt
    
    nyquist = sampling_rate / 2.0
    low_norm = low_freq / nyquist  
    high_norm = high_freq / nyquist
    
    # Design 4th-order Butterworth filter
    b, a = butter(4, [low_norm, high_norm], btype='band')
    
    # Apply zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_notch_filter(self, data, notch_freq, sampling_rate, Q=30):
    """Apply IIR notch filter for power line noise removal"""
    from scipy.signal import iirnotch, filtfilt
    
    # Design notch filter
    b, a = iirnotch(notch_freq, Q, sampling_rate)
    
    # Apply zero-phase filtering  
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_complete_filtering(self, epoch_data, sampling_rate):
    """Apply complete Northwestern filtering pipeline to all channels"""
    filtered_data = epoch_data.copy()
    
    for channel_idx in range(epoch_data.shape[0]):
        channel_num = channel_idx + 1  # Convert to 1-based indexing
        
        if channel_num in self.northwestern_bandpass_filters:
            try:
                # Step 1: Apply channel-specific bandpass filter
                low_freq, high_freq = self.northwestern_bandpass_filters[channel_num]
                filtered_data[channel_idx] = self.apply_bandpass_filter(
                    filtered_data[channel_idx], low_freq, high_freq, sampling_rate
                )
                
                # Step 2: Apply notch filters for power line noise
                for notch_freq in self.northwestern_notch_filters:
                    filtered_data[channel_idx] = self.apply_notch_filter(
                        filtered_data[channel_idx], notch_freq, sampling_rate
                    )
                    
            except Exception as e:
                logger.error(f"Filter failed on channel {channel_num} ({low_freq}-{high_freq} Hz): {e}")
                raise FilterError(f"Channel {channel_num} filtering failed. Check sampling rate and data quality.")
    
    return filtered_data
```

#### 2.3 Custom Exception
```python
class FilterError(Exception):
    """Raised when filtering operations fail"""
    pass
```

### Phase 3: Integrate into Display Pipeline

#### 3.1 Modify plot_polysomnograph() Method
```python
def plot_polysomnograph(self, epoch_data, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
    """Update polysomnograph plot with new data"""
    
    # Apply Northwestern filtering if enabled (for display only)
    if hasattr(self, 'visual_filter_enabled') and self.visual_filter_enabled:
        display_data = self.apply_complete_filtering(epoch_data, sampling_rate)
    else:
        display_data = epoch_data
    
    # ... continue with existing plotting logic using display_data ...
    # Original epoch_data remains unchanged for GSSC processing
```

### Phase 4: GUI Enhancement (Later Implementation)

#### 4.1 Replace Code Parameter with GUI Control
```python
# Replace self.visual_filter_enabled = True with:
self.filter_checkbox = QtWidgets.QCheckBox("Apply Clinical Filters (Northwestern Standard)")
self.filter_checkbox.setChecked(True)  # Default to ON
self.filter_checkbox.toggled.connect(self.toggle_visual_filters)
self.layout.addWidget(self.filter_checkbox)

def toggle_visual_filters(self, checked):
    self.visual_filter_enabled = checked
    # Update title to show filter status
    filter_status = "ON" if checked else "OFF"
    # Could update display immediately if needed
```

## Technical Specifications

### Filter Types
- **Bandpass Filter:** 4th-order Butterworth 
- **Notch Filter:** IIR notch filter
- **Processing Method:** scipy.signal.filtfilt (zero-phase, forward-backward)
- **Sampling Rate:** Uses actual data sampling rate (typically 250 Hz for OpenBCI)

### Error Handling Philosophy
- **Fail Fast:** No fallbacks or silent failures
- **Detailed Logging:** Specific error messages with channel and filter parameters
- **User Notification:** Clear error messages when filtering fails
- **Data Integrity:** Never show misleading "filtered" data that's actually raw

### Performance Considerations
- **Real-time Compatible:** Filtering applied only during display update
- **Memory Efficient:** Processes epoch data in-place where possible
- **GSSC Unaffected:** Original data preserved for sleep classification

## Benefits

### Clinical Accuracy
- **Northwestern Standard:** Matches established clinical PSG practices
- **Professional Visualization:** Clean, artifact-free display suitable for clinical review
- **Power Line Noise Removal:** 50/60 Hz notch filtering eliminates common interference

### Architecture Improvements  
- **Separation of Concerns:** Montage handles physical setup, filtering handles signal processing
- **Channel-Specific:** Appropriate filtering for EEG vs EMG vs respiratory signals
- **Maintainable:** Easy to modify individual channel filters without affecting montage
- **Testable:** Simple code parameter allows easy enable/disable for validation

### Data Integrity
- **Independent Processing:** Visual filtering doesn't affect GSSC sleep classification
- **Original Data Preserved:** Sleep stage prediction uses unfiltered data
- **No Silent Failures:** Problems are immediately visible and actionable

## Testing Plan

### Phase 1 Testing
- ✅ Verify montage functionality after filter removal
- ✅ Confirm no broken references to filter_range or get_filter_ranges()
- ✅ Test all montage types (default, minimal, eog_only)

### Phase 2 Testing  
- ✅ Test bandpass filtering on sample EEG data
- ✅ Verify notch filter effectiveness on 50/60 Hz noise
- ✅ Validate channel-specific filter application (channels 1-16)
- ✅ Test filter pipeline with different sampling rates
- ✅ Error condition testing (invalid parameters, corrupted data)

### Phase 3 Testing
- ✅ Visual comparison: raw vs filtered display
- ✅ Real-time performance during streaming
- ✅ GSSC classification accuracy unchanged
- ✅ Filter toggle functionality (code parameter)

### Phase 4 Testing (GUI)
- ✅ Button/checkbox responsiveness during real-time streaming
- ✅ Layout stability with new GUI controls
- ✅ Visual feedback showing filter status

## Files to Modify

### Primary Files
- `gssc_local/montage.py` - Remove filter coupling
- `gssc_local/pyqt_visualizer.py` - Add Northwestern filtering system
- `gssc_local/realtime_with_restart/visualizer.py` - Remove filter_ranges reference

### Dependencies
- Add `scipy.signal` imports: `butter`, `filtfilt`, `iirnotch`
- Ensure `numpy` available for array operations
- `logging` for error reporting

## Success Criteria

### Functional Requirements
- ✅ Visual display shows filtered EEG data with Northwestern clinical standards
- ✅ GSSC sleep classification uses original unfiltered data  
- ✅ Power line noise (50/60 Hz) effectively removed from display
- ✅ Channel-specific filtering applied correctly (EEG vs EMG vs respiratory)
- ✅ Real-time performance maintained during filtering

### Architecture Requirements  
- ✅ Montage completely decoupled from filtering concerns
- ✅ Filter configuration easily modifiable without montage changes
- ✅ No silent failures - all filter errors properly reported
- ✅ Clean separation between display filtering and processing pipeline

### Quality Requirements
- ✅ Professional clinical-grade visualization quality
- ✅ Consistent with Northwestern University PSG standards
- ✅ Robust error handling with actionable error messages
- ✅ Easy to enable/disable filtering for testing and debugging

## Future Enhancements

### Advanced Filtering Options
- **User-configurable filter ranges** - Allow customization per clinical site
- **Additional notch frequencies** - Support for other power line frequencies
- **Adaptive filtering** - Automatically adjust based on detected noise
- **Filter preview** - Show before/after comparison

### GUI Improvements
- **Filter status indicator** - Visual feedback on current filter state
- **Per-channel filter display** - Show which filters applied to each channel
- **Filter effectiveness meter** - Real-time noise reduction metrics
- **Presets** - Save/load different filter configurations

### Integration Features
- **Export filtered data** - Option to save filtered data to CSV
- **Filter logging** - Track filter usage and effectiveness over time
- **Clinical reporting** - Include filter settings in session reports