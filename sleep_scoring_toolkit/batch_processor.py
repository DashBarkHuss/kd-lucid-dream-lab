"""
Batch processor for generating complete sleep stage predictions from EEG CSV files.

This module provides fast, headless batch processing that generates sleep stage
predictions from EEG CSV files without GUI and real-time overhead.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, NamedTuple


from sleep_scoring_toolkit.realtime_with_restart.channel_mapping import NumPyDataWithBrainFlowDataKey, create_numpy_data_with_brainflow_keys
from sleep_scoring_toolkit.stateful_inference_manager import StatefulInferenceManager
from sleep_scoring_toolkit.realtime_with_restart.core.gap_handler import GapHandler
from sleep_scoring_toolkit.realtime_with_restart.processor import SignalProcessor
from sleep_scoring_toolkit.realtime_with_restart.export.csv.validation import (
    validate_file_path, validate_data_not_empty, validate_timestamps_unique
)
from sleep_scoring_toolkit.realtime_with_restart.utils.timestamp_utils import validate_sample_rate
from sleep_scoring_toolkit.utils.csv_processing_utils import load_brainflow_csv_raw

logger = logging.getLogger(__name__)


class EpochResult(NamedTuple):
    """Result from processing a single epoch."""
    start_time: float
    end_time: float
    sleep_stage: int
    epoch_index: int




class BatchProcessor:
    """Fast batch processor for generating complete sleep stage predictions."""
    
    def __init__(self, montage, sampling_rate: int, eeg_channels_for_scoring: List[str], eog_channels_for_scoring: List[str], 
                 timestamp_column: int, exg_columns: List[int], show_progress: bool = True):
        """Initialize batch processor.
        
        Args:
            montage: Montage configuration for signal processing
            sampling_rate: EEG sampling rate in Hz (required - must match your data)
            eeg_channels_for_scoring: EEG channel labels selected for sleep scoring (required)
            eog_channels_for_scoring: EOG channel labels selected for sleep scoring (required)
            timestamp_column: CSV column index containing timestamps (e.g. 22 for Cyton-Daisy)
            exg_columns: CSV column indices containing EXG data (e.g. [0-15] for Cyton-Daisy)
            show_progress: Whether to show processing progress
        """
        if sampling_rate is None:
            raise ValueError("sampling_rate is required - specify the actual sampling rate of your CSV data")
        if not eeg_channels_for_scoring and not eog_channels_for_scoring:
            raise ValueError("Either eeg_channels_for_scoring or eog_channels_for_scoring must be specified")
        
        self.montage = montage
        self.sampling_rate = sampling_rate
        self.eeg_channels_for_scoring = eeg_channels_for_scoring
        self.eog_channels_for_scoring = eog_channels_for_scoring
        self.timestamp_column = timestamp_column
        self.exg_columns = exg_columns
        self.show_progress = show_progress
        
        # Initialize components
        self.gap_handler = GapHandler(sampling_rate=self.sampling_rate)
        self.signal_processor = SignalProcessor()
        self.stateful_manager = StatefulInferenceManager(
            signal_processor=self.signal_processor, 
            montage=self.montage,
            eeg_channels_for_scoring=self.eeg_channels_for_scoring,
            eog_channels_for_scoring=self.eog_channels_for_scoring
        )
        
        logger.info(f"BatchProcessor initialized with sampling rate: {self.sampling_rate} Hz")

    def _create_etd_from_csv(self, csv_path: str) -> NumPyDataWithBrainFlowDataKey:
        """Create keyed electrode and timestamp data structure from CSV file."""
        transposed_all_channels = load_brainflow_csv_raw(csv_path)
        
        # Extract only the electrode and timestamp channels we need (EEG first, timestamp last)
        etd_board_keys = self.exg_columns + [self.timestamp_column]  # EXG = electrophysiological channels (EEG, EMG, ECG, etc.)
        etd_data = transposed_all_channels[etd_board_keys, :]  # Select only ETD channels (electrode and timestamp data)
        
        # Create keyed ETD structure (electrode and timestamp data only)
        etd_keyed = create_numpy_data_with_brainflow_keys(etd_data, etd_board_keys)
        
        logger.info(f"Loaded CSV with {len(self.exg_columns)} electrode channels (EXG) and {etd_keyed.shape[1]} samples")
        return etd_keyed

    def generate_epoch_timestamps(self, csv_start_time: float, csv_end_time: float) -> List[Tuple[float, float]]:
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
        
        logger.info(f"Generated {len(epoch_timestamps)} epoch timestamps")
        return epoch_timestamps


    def _extract_epoch_data(self, etd_keyed: NumPyDataWithBrainFlowDataKey, epoch_boundary: dict):
        """Extract epoch data using boundary information.
        
        Args:
            etd_keyed: Keyed electrode and timestamp data
            epoch_boundary: Pre-calculated boundary dictionary containing start_idx, end_idx, etc.
            
        Returns:
            NumPyDataWithBrainFlowDataKey: Keyed epoch data ready for processing
            
        Note:
            Timestamps are available directly from epoch_boundary - no need to return them again.
        """
        # Check for boundary extraction errors
        if 'error' in epoch_boundary:
            raise ValueError(f"Epoch {epoch_boundary['epoch_number']} boundary error: {epoch_boundary['details']}")
        
        # Use boundary indices directly
        start_idx = epoch_boundary['start_idx']
        end_idx = epoch_boundary['end_idx']
        
        # Extract epoch data directly from memory (EXG channels only) using keyed access
        epoch_data_raw = etd_keyed.get_by_keys(self.exg_columns)[:, start_idx:end_idx]
        
        # Create structured data format compatible with StatefulInferenceManager
        epoch_data_keyed = create_numpy_data_with_brainflow_keys(epoch_data_raw, self.exg_columns)
        
        return epoch_data_keyed



    def _extract_all_epoch_boundaries(self, etd_keyed: NumPyDataWithBrainFlowDataKey, all_epoch_timestamps: List[Tuple[float, float]]) -> List[dict]:
        """Extract actual epoch boundaries for all epochs upfront using time-based approach.
        
        Each boundary calculation uses np.searchsorted for both start and end times.
        
        Args:
            etd_keyed: Electrode and timestamp data structure
            all_epoch_timestamps: List of (estimated_start_time, estimated_end_time) tuples
            
        Returns:
            List of epoch boundary dictionaries containing:
            - estimated_start_time, estimated_end_time (theoretical perfect 30s intervals)
            - actual_start_time, actual_end_time (from real data timestamps)  
            - start_idx, end_idx (array indices for data extraction)
            - actual_duration (duration between actual timestamps)
            - epoch_number (1-based epoch numbering)
        """
        timestamps = etd_keyed.get_by_key(self.timestamp_column)
        epoch_boundaries = []
        
        logger.info(f"Extracting boundaries for {len(all_epoch_timestamps)} epochs using time-based approach...")
        
        for epoch_idx, (estimated_start_time, estimated_end_time) in enumerate(all_epoch_timestamps):
            # Pure time-based boundary detection
            start_idx = np.searchsorted(timestamps, estimated_start_time)
            end_idx = np.searchsorted(timestamps, estimated_end_time)
            
            # Handle edge cases
            if start_idx >= len(timestamps):
                # Start time is beyond available data
                boundary = {
                    'epoch_number': epoch_idx + 1,
                    'estimated_start_time': estimated_start_time,
                    'estimated_end_time': estimated_end_time,
                    'error': 'start_beyond_data',
                    'details': f"Start time {estimated_start_time} beyond last timestamp {timestamps[-1]}"
                }
                epoch_boundaries.append(boundary)
                continue
                
            if end_idx > len(timestamps):
                # End time is beyond available data, use last available sample
                end_idx = len(timestamps)
            
            if start_idx >= end_idx:
                # Invalid range (shouldn't happen with proper time windows)
                boundary = {
                    'epoch_number': epoch_idx + 1,
                    'estimated_start_time': estimated_start_time,
                    'estimated_end_time': estimated_end_time,
                    'error': 'invalid_range',
                    'details': f"start_idx {start_idx} >= end_idx {end_idx}"
                }
                epoch_boundaries.append(boundary)
                continue
            
            # Extract actual timestamps from the data
            actual_start_time = timestamps[start_idx]
            actual_end_time = timestamps[end_idx-1] if end_idx > 0 else timestamps[start_idx]
            actual_duration = actual_end_time - actual_start_time
            
            # Create boundary object
            boundary = {
                'epoch_number': epoch_idx + 1,
                'estimated_start_time': estimated_start_time,
                'estimated_end_time': estimated_end_time,
                'actual_start_time': actual_start_time,
                'actual_end_time': actual_end_time,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'actual_duration': actual_duration,
                'sample_count': end_idx - start_idx
            }
            epoch_boundaries.append(boundary)
        
        logger.info(f"✅ Extracted boundaries for {len(epoch_boundaries)} epochs")
        return epoch_boundaries

    def _validate_epoch_boundaries(self, epoch_boundaries: List[dict]) -> None:
        """Validate epoch boundaries.
        
        Args:
            epoch_boundaries: List of boundary dictionaries
            
        Raises:
            ValueError: If any epochs have validation issues, with details of all problems
        """
        expected_duration = 30.0
        tolerance = 0.5
        problematic_epochs = []
        
        logger.info(f"Validating {len(epoch_boundaries)} epoch boundaries...")
        
        for boundary in epoch_boundaries:
            epoch_number = boundary['epoch_number']
            
            # Check for extraction errors
            if 'error' in boundary:
                problematic_epochs.append({
                    'epoch': epoch_number,
                    'issue': boundary['error'],
                    'details': boundary['details']
                })
                continue
            
            # Check duration tolerance
            actual_duration = boundary['actual_duration']
            if abs(actual_duration - expected_duration) > tolerance:
                problematic_epochs.append({
                    'epoch': epoch_number,
                    'issue': 'duration_out_of_tolerance',
                    'actual_duration': actual_duration,
                    'expected_duration': expected_duration,
                    'tolerance': tolerance,
                    'details': f"Epoch duration {actual_duration:.3f}s outside tolerance (expected {expected_duration}s ± {tolerance}s)"
                })
                continue
            
            # Check for minimal sample count (should have some data)
            sample_count = boundary['sample_count']
            if sample_count < 10:  # Arbitrary minimum - should have some samples
                problematic_epochs.append({
                    'epoch': epoch_number,
                    'issue': 'insufficient_samples',
                    'sample_count': sample_count,
                    'details': f"Epoch has only {sample_count} samples"
                })
        
        if problematic_epochs:
            error_message = self._format_validation_error_message(problematic_epochs, len(epoch_boundaries))
            raise ValueError(error_message)
        
        logger.info(f"✅ All {len(epoch_boundaries)} epoch boundaries passed validation")

    def _format_validation_error_message(self, problematic_epochs: List[dict], total_epochs: int) -> str:
        """Format detailed validation error message grouped by issue type."""
        error_lines = [f"Found {len(problematic_epochs)} problematic epochs out of {total_epochs} total:"]
        
        # Group by issue type for better reporting
        duration_issues = [e for e in problematic_epochs if e['issue'] == 'duration_out_of_tolerance']
        extraction_errors = [e for e in problematic_epochs if e['issue'] in ['start_beyond_data', 'invalid_range']]
        sample_issues = [e for e in problematic_epochs if e['issue'] == 'insufficient_samples']
        
        if duration_issues:
            error_lines.append(f"\n⚠️  Duration Issues ({len(duration_issues)} epochs):")
            for epoch_info in duration_issues[:10]:  # Show first 10
                error_lines.append(f"   Epoch {epoch_info['epoch']}: {epoch_info['actual_duration']:.3f}s")
            if len(duration_issues) > 10:
                error_lines.append(f"   ... and {len(duration_issues) - 10} more duration issues")
        
        if extraction_errors:
            error_lines.append(f"\n❌ Extraction Errors ({len(extraction_errors)} epochs):")
            for epoch_info in extraction_errors[:5]:  # Show first 5
                error_lines.append(f"   Epoch {epoch_info['epoch']}: {epoch_info['details']}")
            if len(extraction_errors) > 5:
                error_lines.append(f"   ... and {len(extraction_errors) - 5} more extraction errors")
        
        if sample_issues:
            error_lines.append(f"\n📊 Sample Count Issues ({len(sample_issues)} epochs):")
            for epoch_info in sample_issues[:5]:  # Show first 5
                error_lines.append(f"   Epoch {epoch_info['epoch']}: {epoch_info['details']}")
            if len(sample_issues) > 5:
                error_lines.append(f"   ... and {len(sample_issues) - 5} more sample issues")
        
        error_lines.append(f"\n💡 Real-world data may have gaps - consider preprocessing to handle discontinuities")
        
        return '\n'.join(error_lines)

    def _validate_gap_free_for_inference(self, timestamps: np.ndarray, max_gap_seconds: float = 2.0) -> None:
        """Validate that data is gap-free for inference scenarios.
        
        Args:
            timestamps: Array of timestamp values to check for gaps
            max_gap_seconds: Maximum allowed gap size in seconds (default: 2.0)
            
        Raises:
            ValueError: If gaps larger than max_gap_seconds are found
            
        Note: Default parameter is acceptable here because:
        - Private method with single, specific purpose (inference validation)
        - 2.0s threshold is domain-appropriate for sleep research
        - No downstream configuration effects - validation either passes or fails
        - Most callers need the same threshold value
        """
        has_gap, gap_size, gap_start_idx, gap_end_idx = self.gap_handler.detect_largest_gap(timestamps)
        if has_gap and abs(gap_size) > max_gap_seconds:
            raise ValueError(
                f"Found significant gap of {gap_size:.2f}s at indices {gap_start_idx}-{gap_end_idx}. "
                "Model-researcher comparison requires continuous data without large gaps."
            )

    def _reorder_data_by_timestamps(self, etd_keyed: NumPyDataWithBrainFlowDataKey) -> tuple[NumPyDataWithBrainFlowDataKey, np.ndarray]:
        """Reorder data chronologically by timestamps if needed.
        
        Args:
            etd_keyed: Keyed electrode and timestamp data structure
            
        Returns:
            Tuple of (reordered_etd_keyed, sorted_timestamps)
        """
        timestamps = etd_keyed.get_by_key(self.timestamp_column)
        timestamps_sorted = np.sort(timestamps)
        
        if not np.array_equal(timestamps, timestamps_sorted):
            logger.warning("Timestamps not in chronological order - reordering data by timestamp")
            
            # Get sorting indices to reorder both timestamps and EEG data
            sort_indices = np.argsort(timestamps)
            
            # Reorder the ETD keyed data structure
            reordered_data = etd_keyed.data[:, sort_indices]
            etd_keyed_reordered = NumPyDataWithBrainFlowDataKey(
                data=reordered_data,
                channel_mapping=etd_keyed.channel_mapping
            )
            timestamps_reordered = timestamps[sort_indices]
            logger.info("Successfully reordered data by timestamp")
            
            return etd_keyed_reordered, timestamps_reordered
        else:
            return etd_keyed, timestamps

    def process_csv_file(self, csv_path: str, output_path: str) -> dict:
        """Process complete CSV file and generate all sleep stage predictions.
        
        Args:
            csv_path: Path to input EEG CSV file (BrainFlow format)
            output_path: Path for output sleep stages CSV file
            
        Returns:
            Dictionary with processing results and statistics
        """
        processing_start_time = pd.Timestamp.now()
        
        # Phase 1: Upfront validation and data loading (once per file)
        logger.info(f"Processing: {csv_path}")
        validate_file_path(csv_path)
        
        etd_keyed_unsanitized = self._create_etd_from_csv(csv_path)
        timestamps_unsanitized = etd_keyed_unsanitized.get_by_key(self.timestamp_column)
        validate_data_not_empty(etd_keyed_unsanitized.data)
        validate_timestamps_unique(timestamps_unsanitized)
        validate_sample_rate(timestamps_unsanitized, self.sampling_rate)
        
        # Phase 2: Reorder samples by timestamp if needed (same as main pipeline)
        etd_keyed_ordered, timestamps_ordered = self._reorder_data_by_timestamps(etd_keyed_unsanitized)
        
        # Phase 3: Gap detection with fail-fast (only check for research-significant gaps >2s)
        self._validate_gap_free_for_inference(timestamps_ordered)
        
        # Phase 4: Generate all epoch timestamps (sequential 30s intervals)
        all_epoch_timestamps = self.generate_epoch_timestamps(timestamps_ordered[0], timestamps_ordered[-1])
        
        # Phase 4.5: Extract all epoch boundaries upfront (time-based, single calculation)
        epoch_boundaries = self._extract_all_epoch_boundaries(etd_keyed_ordered, all_epoch_timestamps)
        
        # Phase 4.6: Validate all boundaries (fail-fast with comprehensive reporting)
        self._validate_epoch_boundaries(epoch_boundaries)
        
        # Phase 5: Sequential processing using boundaries
        results_of_processed_epochs = self._process_all_epochs(etd_keyed_ordered, epoch_boundaries)
        
        # Phase 5: Save all results using existing CSV format
        self._save_results(results_of_processed_epochs, output_path)
        
        # Calculate processing statistics
        processing_end_time = pd.Timestamp.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()
        
        stats = {
            'epochs_processed': len(results_of_processed_epochs),
            'processing_time': processing_time,
            'output_path': output_path,
            'data_duration': timestamps_ordered[-1] - timestamps_ordered[0],
            'speed_ratio': (timestamps_ordered[-1] - timestamps_ordered[0]) / processing_time
        }
        
        logger.info(f"Processing completed: {stats['epochs_processed']} epochs in {stats['processing_time']:.2f}s")
        logger.info(f"Speed ratio: {stats['speed_ratio']:.1f}x real-time")
        
        return stats

    def _process_all_epochs(self, etd_keyed_ordered: NumPyDataWithBrainFlowDataKey, epoch_boundaries: List[dict]) -> List[EpochResult]:
        """Process all epochs sequentially using pre-validated boundaries.
        
        Args:
            etd_keyed_ordered: Keyed electrode and timestamp data in chronological order
            epoch_boundaries: Pre-validated list of epoch boundary dictionaries
            
        Returns:
            List[EpochResult]: Results from processing each epoch
        """
        results_of_processed_epochs = []
        
        for boundary in epoch_boundaries:
            if self.show_progress:
                logger.info(f"Processing epoch {boundary['epoch_number']}/{len(epoch_boundaries)}")
            
            # Extract epoch data using boundary
            epoch_data = self._extract_epoch_data(etd_keyed_ordered, boundary)
            # Note: No need for _validate_epoch_duration since validation already done upfront
            
            # Process through continuous hidden state (no resets since no gaps)
            predicted_class, class_probs, new_hidden_states = self.stateful_manager.process_epoch(epoch_data)
            
            # Use timestamps directly from boundary
            results_of_processed_epochs.append(EpochResult(
                boundary['actual_start_time'], 
                boundary['actual_end_time'], 
                predicted_class, 
                boundary['epoch_number'] - 1
            ))
        
        return results_of_processed_epochs

    def _save_results(self, results: List[EpochResult], output_path: str) -> None:
        """Save processing results to CSV file."""
        df_data = []
        for result in results:
            df_data.append({
                'timestamp_start': result.start_time,
                'timestamp_end': result.end_time,
                'sleep_stage': result.sleep_stage,
                'buffer_id': result.epoch_index
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")