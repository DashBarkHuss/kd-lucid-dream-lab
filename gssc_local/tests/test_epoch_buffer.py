import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

"""
Tests for the epoch buffer functionality in DataManager.

This module contains tests for:
1. Buffer trimming
2. Index translation
3. Processing continuity
4. Round-robin integration

The tests verify that the buffer management system correctly handles:
- Buffer size limits
- Data continuity
- Index tracking
- Round-robin processing
"""

import pytest
import numpy as np
from gssc_local.realtime_with_restart.data_manager import DataManager
from gssc_local.realtime_with_restart.etd_buffer_manager import ETDBufferManager
from gssc_local.tests.test_utils import create_brainflow_test_data, transform_to_stream_format
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
import tempfile
import os

class TestEpochBuffer:
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager instance with playback board."""
        master_board_id = BoardIds.CYTON_DAISY_BOARD
        playback_board_id = BoardIds.PLAYBACK_FILE_BOARD
        sampling_rate = BoardShim.get_sampling_rate(master_board_id)
        
        # Create a temporary file for playback data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            # Generate test data and save it to the temp file
            data, metadata = create_brainflow_test_data(
                duration_seconds=120,
                sampling_rate=sampling_rate,
                add_noise=False,
                board_id=master_board_id
            )
            # Update metadata to use exg_channels instead of eeg_channels
            metadata['eeg_channels'] = BoardShim.get_exg_channels(master_board_id)
            # Save data in BrainFlow format
            np.savetxt(temp_file.name, data, delimiter=',')
            
        # Create BrainFlowInputParams for playback
        params = BrainFlowInputParams()
        params.file = temp_file.name
        params.master_board = master_board_id
        params.file_operation = 'read'
        params.file_data_type = 'timestamp'
        
        # Initialize the playback board
        board = BoardShim(playback_board_id, params)
        board.prepare_session()
        board.start_stream()
        
        # Create DataManager with the playback board
        data_manager = DataManager(board, sampling_rate)
        
        # Clean up the temp file after the test
        def cleanup():
            board.stop_stream()
            board.release_session()
            os.unlink(temp_file.name)
            
        # Register cleanup
        data_manager.cleanup = cleanup
        
        return data_manager

    @pytest.fixture
    def buffer_manager(self, test_data):
        """Create a buffer manager instance for testing."""
        _, metadata = test_data
        # Get channel information from the test data
        eeg_channels = metadata['eeg_channels']
        timestamp_channel = metadata['timestamp_channel']
        electrode_and_timestamp_channels = list(eeg_channels)
        if timestamp_channel not in electrode_and_timestamp_channels:
            electrode_and_timestamp_channels.append(timestamp_channel)
            
        return ETDBufferManager(
            max_buffer_size=35 * metadata['sampling_rate'],  # 35 seconds of data
            timestamp_channel_index=len(electrode_and_timestamp_channels) - 1,  # Timestamp is last channel
            channel_count=len(electrode_and_timestamp_channels),
            electrode_and_timestamp_channels=electrode_and_timestamp_channels
        )

    @pytest.fixture
    def test_data(self, data_manager):
        """Create test data with the correct number of channels."""
        master_board_id = BoardIds.CYTON_DAISY_BOARD
        sampling_rate = BoardShim.get_sampling_rate(master_board_id)
        
        # Generate test data
        data, metadata = create_brainflow_test_data(
            duration_seconds=120,
            sampling_rate=sampling_rate,
            add_noise=False,
            board_id=master_board_id
        )
        
        # Update metadata
        metadata['eeg_channels'] = data_manager.electrode_channels
        metadata['timestamp_channel'] = data_manager.board_timestamp_channel
        
        return data, metadata

    def test_buffer_trimming(self, data_manager, test_data):
        """Test that buffer is trimmed correctly when it exceeds 35 seconds.
        Data should only be trimmed after it has been fully processed by all round-robin buffers.
        """
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Calculate expected buffer sizes
        min_buffer_size = 35 * sampling_rate  # 35 seconds worth of data
        initial_data_size = 40 * sampling_rate  # 40 seconds of data to start
        
        # Add initial data to buffer (40 seconds worth)
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        initial_data_stream = transform_to_stream_format(initial_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(initial_data_stream)
        
        # Verify initial buffer size
        initial_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
        assert initial_buffer_size == initial_data_size, \
            f"Initial buffer size should be {initial_data_size}, got {initial_buffer_size}"
            
        # Add more data to exceed 35 seconds
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(additional_data_stream)
        
        # Try to trim buffer before processing any epochs - should not trim
        data_manager.etd_buffer_manager.trim_buffer(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs, data_manager.points_per_step)
        assert data_manager.etd_buffer_manager._get_total_data_points() == initial_data_size + 20 * sampling_rate, \
            "Buffer should not be trimmed before any epochs are processed"
            
        # Process first epoch for each buffer
        for buffer_id in range(6):
            epoch_start = buffer_id * data_manager.points_per_step
            epoch_end = epoch_start + data_manager.points_per_epoch
            data_manager.manage_epoch(
                buffer_id=buffer_id,
                epoch_start_idx_abs=epoch_start,
                epoch_end_idx_abs=epoch_end
            )
            
            # After each epoch is processed, trim buffer
            data_manager.etd_buffer_manager.trim_buffer(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs, data_manager.points_per_step)
            
            # Verify buffer size after each trim
            current_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
            assert current_buffer_size >= min_buffer_size, \
                f"Buffer should maintain at least {min_buffer_size} points after processing buffer {buffer_id}, got {current_buffer_size}"
            
            # Verify data continuity
            timestamps = data_manager.etd_buffer_manager.electrode_and_timestamp_data[data_manager.etd_buffer_manager.timestamp_channel_index]
            data_manager.etd_buffer_manager._verify_timestamp_continuity(timestamps)
            
            # Verify offset tracking
            assert data_manager.etd_buffer_manager.offset >= 0, "Buffer offset should be non-negative"
            
            # Verify total streamed samples tracking
            assert data_manager.etd_buffer_manager.total_streamed_samples == initial_data_size + 20 * sampling_rate, \
                f"Total streamed samples should be {initial_data_size + 20 * sampling_rate}, got {data_manager.etd_buffer_manager.total_streamed_samples}"
            
            # assert that the last channel is the timestamp channel
            assert data_manager.etd_buffer_manager.electrode_and_timestamp_channels[-1] == metadata['timestamp_channel'], \
                "Last channel should be the timestamp channel"
            # assert that the last channels first value is a timestamp
            first_timestamp = data_manager.etd_buffer_manager.electrode_and_timestamp_data[-1][0]
            assert first_timestamp > 1700000000, \
                f"First value of last channel should be a Unix timestamp after 2023, got {first_timestamp}"
            
            # Calculate theoretical voltage range based on Cyton Daisy board specifications
            # 24-bit ADC, 4.5V reference, gain of 24, zero point at 8192
            min_adc = 0
            max_adc = 2**24 - 1  # 24-bit ADC
            zero_point = 8192
            ref_voltage = 4.5
            gain = 24
            
            # Calculate theoretical voltage range in microvolts
            min_voltage = (min_adc - zero_point) * (ref_voltage / gain) / gain
            max_voltage = (max_adc - zero_point) * (ref_voltage / gain) / gain
            
            # assert that the last channels first value is not an eeg value (should be outside theoretical range)
            theoretical_range = max(abs(min_voltage), abs(max_voltage))
            assert abs(first_timestamp) > theoretical_range, \
                f"First value of last channel should be outside theoretical EEG range (±{theoretical_range:.2f} µV), got {first_timestamp}"

    def test_index_translation(self, data_manager, test_data):
        """Test that index translation works correctly with buffer trimming.
        Trimming should only happen after all epochs that need the oldest data have been processed.
        If we try to access an absolute index that has been trimmed, we should expect a ValueError.
        """
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Add initial data (40 seconds)
        initial_data_size = 40 * sampling_rate
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        initial_data_stream = transform_to_stream_format(initial_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(initial_data_stream)
        
        # Test absolute to relative conversion before trimming
        test_absolute_idx = 20 * sampling_rate  # 20 seconds into the data
        relative_idx = data_manager.etd_buffer_manager._adjust_index_with_offset(test_absolute_idx, to_etd=True)
        assert relative_idx == test_absolute_idx, \
            f"Before trimming: absolute index {test_absolute_idx} should map to relative index {test_absolute_idx}, got {relative_idx}"
        
        # Add more data and trim buffer
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(additional_data_stream)
        data_manager.etd_buffer_manager.trim_buffer(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs, data_manager.points_per_step)  # Trim buffer to max_buffer_size
        
        # After trimming, test_absolute_idx may have been trimmed away
        if test_absolute_idx < data_manager.etd_buffer_manager.offset:
            # Should raise ValueError if we try to access an index that has been trimmed
            with pytest.raises(ValueError):
                data_manager.etd_buffer_manager._adjust_index_with_offset(test_absolute_idx, to_etd=True)
        else:
            # Otherwise, translation should work
            relative_idx = data_manager.etd_buffer_manager._adjust_index_with_offset(test_absolute_idx, to_etd=True)
            expected_relative_idx = test_absolute_idx - data_manager.etd_buffer_manager.offset
            assert relative_idx == expected_relative_idx, \
                f"After trimming: absolute index {test_absolute_idx} should map to relative index {expected_relative_idx}, got {relative_idx}"
            # Test relative to absolute conversion
            absolute_idx = data_manager.etd_buffer_manager._adjust_index_with_offset(relative_idx, to_etd=False)
            assert absolute_idx == test_absolute_idx, \
                f"Relative index {relative_idx} should map back to absolute index {test_absolute_idx}, got {absolute_idx}"
        
        # Test edge cases
        # Test index at buffer start
        start_relative = data_manager.etd_buffer_manager._adjust_index_with_offset(data_manager.etd_buffer_manager.offset, to_etd=True)
        assert start_relative == 0, \
            f"Buffer start should map to relative index 0, got {start_relative}"
        # Test index at buffer end
        end_absolute = data_manager.etd_buffer_manager.total_streamed_samples - 1
        if end_absolute < data_manager.etd_buffer_manager.offset:
            with pytest.raises(ValueError):
                data_manager.etd_buffer_manager._adjust_index_with_offset(end_absolute, to_etd=True)
        else:
            end_relative = data_manager.etd_buffer_manager._adjust_index_with_offset(end_absolute, to_etd=True)
            expected_end_relative = data_manager._get_total_data_points_etd() - 1
            assert end_relative == expected_end_relative, \
                f"Buffer end should map to relative index {expected_end_relative}, got {end_relative}"
        # Test invalid indices
        with pytest.raises(ValueError):
            data_manager.etd_buffer_manager._adjust_index_with_offset(-1, to_etd=True)
        # Note: index == total_streamed_samples is now valid for exclusive end slicing
        # Test an index that is definitely invalid (beyond the end)
        with pytest.raises(ValueError):
            data_manager.etd_buffer_manager._adjust_index_with_offset(data_manager.etd_buffer_manager.total_streamed_samples + 1, to_etd=True)

    def test_processing_continuity(self, data_manager, test_data):
        """Test that epoch processing continues correctly after buffer trimming.
        Trimming should only happen after all epochs that need the old data have been processed.
        """
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        points_per_epoch = data_manager.points_per_epoch
    
        # Add initial data (40 seconds)
        initial_data_size = 40 * sampling_rate
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        initial_data_stream = transform_to_stream_format(initial_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(initial_data_stream)
    
        # Process first epoch
        first_epoch_start = 0
        first_epoch_end = points_per_epoch
        initial_epochs_count = data_manager.epochs_scored
        initial_matrix_length = len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[0])
        
        data_manager.manage_epoch(
            buffer_id=0,
            epoch_start_idx_abs=first_epoch_start,
            epoch_end_idx_abs=first_epoch_end
        )
    
        # Add more data and process second epoch before trimming
        additional_data = data[initial_data_size:initial_data_size + 25 * sampling_rate, :]  # shape: (n_samples, n_channels) - increased to 25 seconds
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(additional_data_stream)
    
        # Process second epoch before trimming  
        second_epoch_start = points_per_epoch
        second_epoch_end = 2 * points_per_epoch
        
        data_manager.manage_epoch(
            buffer_id=0,
            epoch_start_idx_abs=second_epoch_start,
            epoch_end_idx_abs=second_epoch_end
        )
    
        # Now trim buffer (after all epochs that need the old data have been processed)
        data_manager.etd_buffer_manager.trim_buffer(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs, data_manager.points_per_step)
    
        # Verify sleep stage processing continues by checking tracking matrix and epoch count
        assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[0]) == initial_matrix_length + 2, \
            "Both epochs should be tracked in the processing matrix"
        assert data_manager.epochs_scored == initial_epochs_count + 2, \
            "Both epochs should have been scored successfully"
    
        # Test gap detection after trimming
        # Get the current last timestamp from the buffer to calculate where the gap should start
        current_timestamps = np.array(data_manager.etd_buffer_manager._get_timestamps())
        last_existing_timestamp = current_timestamps[-1]
        
        # Create data with a gap - start 60 seconds after the last existing timestamp  
        gap_start_time = last_existing_timestamp + 60.0  # 60 second gap
        gap_data, gap_metadata = create_brainflow_test_data(
            duration_seconds=10,
            sampling_rate=sampling_rate,
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD,
            start_time=gap_start_time
        )
    
        # Add gap data
        gap_data_stream = transform_to_stream_format(gap_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(gap_data_stream)
    
        # Test gap detection
        # Get updated timestamps from the buffer
        timestamps = np.array(data_manager.etd_buffer_manager._get_timestamps())
        
        # Find where the gap occurs - look for the largest timestamp difference
        timestamp_diffs = np.diff(timestamps)
        expected_interval = 1.0 / sampling_rate
        
        # Find the index where the gap occurs (largest difference)
        gap_idx = np.argmax(timestamp_diffs)
        gap_size = timestamp_diffs[gap_idx] - expected_interval
        
        # Verify that the gap is detected
        assert gap_size > 2.0, f"Gap size should be greater than 2 seconds, got {gap_size}"
        
        # Now verify that the gap is detected by the gap handler
        # gap_idx is the index in the timestamp array where the gap occurs
        # Convert this to absolute indices by adding the buffer offset
        gap_idx_abs = gap_idx + data_manager.etd_buffer_manager.offset
        
        # Create an epoch that spans across the gap  
        # We want to test gap detection, so make sure the epoch crosses the gap
        epoch_start_idx_abs = gap_idx_abs - 100  # Start well before the gap
        epoch_end_idx_abs = gap_idx_abs + 100    # End well after the gap
        
        # Make sure we don't exceed buffer bounds
        epoch_start_idx_abs = max(data_manager.etd_buffer_manager.offset, epoch_start_idx_abs)
        epoch_end_idx_abs = min(data_manager.etd_buffer_manager.total_streamed_samples - 1, epoch_end_idx_abs)
        
        has_gap, detected_gap_size = data_manager.validate_epoch_gaps(
            buffer_id=0,
            epoch_start_idx_abs=epoch_start_idx_abs,
            epoch_end_idx_abs=epoch_end_idx_abs
        )
        
        # The gap should be detected
        assert has_gap, "Should detect gap in data"
        
        # The detected gap size should be close to the actual gap size
        assert abs(detected_gap_size - gap_size) < 0.1, f"Detected gap size {detected_gap_size} should be close to actual gap size {gap_size}"
        
        # Verify matrix of processed epochs is updated correctly  
        assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[0]) == 2, \
            "Should track both processed epochs"
            
        # Verify epoch indices are correct after trimming
        first_epoch_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[0][0]
        second_epoch_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[0][1]
        assert first_epoch_idx == first_epoch_start, \
            f"First epoch start index should be {first_epoch_start}, got {first_epoch_idx}"
        assert second_epoch_idx == second_epoch_start, \
            f"Second epoch start index should be {second_epoch_start}, got {second_epoch_idx}"

    def test_round_robin_integration(self, data_manager, test_data):
        """Test full round-robin cycle with buffer trimming.
        Trimming should only happen after all round-robin epochs that need the oldest data have been processed.
        """
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        points_per_epoch = data_manager.points_per_epoch
        points_per_step = data_manager.points_per_step
        
        # Add initial data (60 seconds to ensure enough data for all buffers)
        initial_data_size = 60 * sampling_rate
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        initial_data_stream = transform_to_stream_format(initial_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(initial_data_stream)
        
        # Process each buffer in sequence before trimming
        for buffer_id in range(6):
            # Calculate epoch indices for this buffer
            epoch_start = buffer_id * points_per_step
            epoch_end = epoch_start + points_per_epoch
            # Process epoch using manage_epoch (which handles the full pipeline)
            data_manager.manage_epoch(
                buffer_id=buffer_id,
                epoch_start_idx_abs=epoch_start,
                epoch_end_idx_abs=epoch_end
            )
            # Verify buffer tracking
            assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id]) == 1, \
                f"Buffer {buffer_id} should track its processed epoch"
            # Verify epoch indices
            processed_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id][0]
            assert processed_idx == epoch_start, \
                f"Buffer {buffer_id} should track correct start index {epoch_start}, got {processed_idx}"
        
        # Now trim buffer (after all round-robin epochs that need the old data have been processed)
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(additional_data_stream)
        data_manager.etd_buffer_manager.trim_buffer(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs, data_manager.points_per_step)  # Trim buffer to max_buffer_size
        
        # Process second round of epochs after trimming - only process buffers that have enough data
        for buffer_id in range(6):
            # Calculate epoch indices for second round - continue from where the first epoch ended
            # Each buffer processes epochs separated by points_per_epoch
            first_epoch_start = buffer_id * points_per_step
            epoch_start = first_epoch_start + points_per_epoch  # Move to next epoch for this buffer
            epoch_end = epoch_start + points_per_epoch
            
            # Only process if we have enough data for this epoch
            total_available_samples = data_manager.etd_buffer_manager.total_streamed_samples
            if epoch_end <= total_available_samples:
                # Process epoch using manage_epoch
                data_manager.manage_epoch(
                    buffer_id=buffer_id,
                    epoch_start_idx_abs=epoch_start,
                    epoch_end_idx_abs=epoch_end
                )
                # Verify buffer tracking
                assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id]) == 2, \
                    f"Buffer {buffer_id} should track both processed epochs"
                # Verify epoch indices after trimming
                second_processed_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id][1]
                assert second_processed_idx == epoch_start, \
                    f"Buffer {buffer_id} should track correct second start index {epoch_start}, got {second_processed_idx}"
            else:
                # Skip this buffer as we don't have enough data - this is expected for later buffers
                print(f"Skipping buffer {buffer_id} second epoch: not enough data ({epoch_end} > {total_available_samples})")
        
        # Verify buffer size is reasonable 
        # Note: Buffer may exceed max_buffer_size temporarily before trimming can safely remove all old data
        final_buffer_size = data_manager._get_total_data_points_etd()
        max_buffer_size = data_manager.etd_buffer_manager.max_buffer_size
        # Allow some flexibility since trimming can only remove data that's been fully processed by all buffers
        assert final_buffer_size <= max_buffer_size * 3, \
            f"Buffer should not exceed 3x max size {max_buffer_size * 3}, got {final_buffer_size}"
        # Should also have some reasonable amount of data
        assert final_buffer_size > 0, "Buffer should contain some data"
        # Test total streamed samples tracking
        expected_total_samples = initial_data_size + 20 * sampling_rate
        assert data_manager.etd_buffer_manager.total_streamed_samples == expected_total_samples, \
            f"Total streamed samples should be {expected_total_samples}, got {data_manager.etd_buffer_manager.total_streamed_samples}"

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v']) 