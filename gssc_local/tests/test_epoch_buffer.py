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
from gssc_local.tests.test_utils import create_brainflow_test_data
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
    def test_data(self):
        """Create test data with known characteristics."""
        # Create 2 minutes of data to test buffer trimming
        data, metadata = create_brainflow_test_data(
            duration_seconds=120,
            sampling_rate=125,  # Cyton Daisy sampling rate
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD
        )
        return data, metadata

    def test_buffer_trimming(self, data_manager, test_data):
        """Test that buffer is trimmed correctly when it exceeds 35 seconds."""
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Calculate expected buffer sizes
        min_buffer_size = 35 * sampling_rate  # 35 seconds worth of data
        initial_data_size = 40 * sampling_rate  # 40 seconds of data to start
        
        # Add initial data to buffer (40 seconds worth)
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Verify initial buffer size
        initial_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
        assert initial_buffer_size == initial_data_size, \
            f"Initial buffer size should be {initial_data_size}, got {initial_buffer_size}"
            
        # Add more data to exceed 35 seconds
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        data_manager.add_to_data_processing_buffer(additional_data)
        
        # Call buffer trimming
        points_to_remove = initial_buffer_size + 20 * sampling_rate - min_buffer_size
        data_manager.etd_buffer_manager.trim_buffer(points_to_remove)
        
        # Verify buffer was trimmed to correct size
        final_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
        assert final_buffer_size == min_buffer_size, \
            f"Buffer should be trimmed to {min_buffer_size}, got {final_buffer_size}"
            
        # Verify data continuity
        # Get timestamps from buffer
        timestamps = data_manager.etd_buffer_manager.electrode_and_timestamp_data[data_manager.etd_buffer_manager.timestamp_channel_index]
        data_manager.etd_buffer_manager._verify_timestamp_continuity(timestamps)
        
        # Verify offset tracking
        assert data_manager.etd_buffer_manager.offset > 0, "Buffer offset should be updated after trimming"
        
        # Test total streamed samples tracking
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
        """Test that index translation works correctly with buffer trimming."""
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Add initial data (40 seconds)
        initial_data_size = 40 * sampling_rate
        initial_data = data[:initial_data_size, :]  # shape: (n_samples, n_channels)
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Test absolute to relative conversion before trimming
        test_absolute_idx = 20 * sampling_rate  # 20 seconds into the data
        relative_idx = data_manager.etd_buffer_manager._adjust_index_with_offset(test_absolute_idx, to_etd=True)
        assert relative_idx == test_absolute_idx, \
            f"Before trimming: absolute index {test_absolute_idx} should map to relative index {test_absolute_idx}, got {relative_idx}"
            
        # Add more data and trim buffer
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        data_manager.add_to_data_processing_buffer(additional_data)
        data_manager.etd_buffer_manager.trim_buffer(points_to_remove=20 * sampling_rate)  # Trim 20 seconds of data
        
        # Test absolute to relative conversion after trimming
        # The absolute index should be adjusted by the offset
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
        end_relative = data_manager.etd_buffer_manager._adjust_index_with_offset(end_absolute, to_etd=True)
        expected_end_relative = data_manager._get_total_data_points_etd() - 1
        assert end_relative == expected_end_relative, \
            f"Buffer end should map to relative index {expected_end_relative}, got {end_relative}"
            
        # Test invalid indices
        with pytest.raises(ValueError):
            data_manager.etd_buffer_manager._adjust_index_with_offset(-1, to_etd=True)
            
        with pytest.raises(ValueError):
            data_manager.etd_buffer_manager._adjust_index_with_offset(data_manager.etd_buffer_manager.total_streamed_samples, to_etd=True)

    def test_processing_continuity(self, data_manager, test_data):
        """Test that epoch processing continues correctly after buffer trimming."""
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        points_per_epoch = data_manager.points_per_epoch
        
        # Add initial data (40 seconds)
        initial_data_size = 40 * sampling_rate
        initial_data = data[:, :initial_data_size]
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Process first epoch
        first_epoch_start = 0
        first_epoch_end = points_per_epoch
        first_sleep_stage = data_manager._process_epoch(
            start_idx_abs=first_epoch_start,
            end_idx_abs=first_epoch_end,
            buffer_id=0
        )
        
        # Add more data and trim buffer
        additional_data = data[:, initial_data_size:initial_data_size + 20 * sampling_rate]
        data_manager.add_to_data_processing_buffer(additional_data)
        data_manager.etd_buffer_manager.trim_buffer(points_to_remove=20 * sampling_rate)  # Trim 20 seconds of data
        
        # Process second epoch after trimming
        second_epoch_start = points_per_epoch
        second_epoch_end = 2 * points_per_epoch
        second_sleep_stage = data_manager._process_epoch(
            start_idx_abs=second_epoch_start,
            end_idx_abs=second_epoch_end,
            buffer_id=0
        )
        
        # Verify sleep stage processing continues
        assert first_sleep_stage is not None, "First epoch should be processed successfully"
        assert second_sleep_stage is not None, "Second epoch should be processed after trimming"
        
        # Test gap detection after trimming
        # Create data with a gap
        gap_data, gap_metadata = create_brainflow_test_data(
            duration_seconds=10,
            sampling_rate=sampling_rate,
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD,
            start_time=metadata['start_time'] + 60  # Start 60 seconds after first data
        )
        
        # Add gap data
        data_manager.add_to_data_processing_buffer(gap_data)
        
        # Test gap detection
        has_gap, gap_size = data_manager.validate_epoch_gaps(
            buffer_id=0,
            epoch_start_idx_abs=data_manager.etd_buffer_manager.total_streamed_samples - points_per_epoch,
            epoch_end_idx_abs=data_manager.etd_buffer_manager.total_streamed_samples
        )
        
        assert has_gap, "Should detect gap in data"
        assert gap_size > 0, "Gap size should be positive"
        
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
        """Test full round-robin cycle with buffer trimming."""
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        points_per_epoch = data_manager.points_per_epoch
        points_per_step = data_manager.points_per_step
        
        # Calculate buffer positions
        # Buffer 0: 0-30s
        # Buffer 1: 5-35s
        # Buffer 2: 10-40s
        # Buffer 3: 15-45s
        # Buffer 4: 20-50s
        # Buffer 5: 25-55s
        
        # Add initial data (60 seconds to ensure enough data for all buffers)
        initial_data_size = 60 * sampling_rate
        initial_data = data[:, :initial_data_size]
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Process each buffer in sequence
        sleep_stages = []
        for buffer_id in range(6):
            # Calculate epoch indices for this buffer
            epoch_start = buffer_id * points_per_step
            epoch_end = epoch_start + points_per_epoch
            
            # Process epoch
            sleep_stage = data_manager._process_epoch(
                start_idx_abs=epoch_start,
                end_idx_abs=epoch_end,
                buffer_id=buffer_id
            )
            sleep_stages.append(sleep_stage)
            
            # Verify sleep stage was processed
            assert sleep_stage is not None, f"Buffer {buffer_id} should process epoch successfully"
            
            # Verify buffer tracking
            assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id]) == 1, \
                f"Buffer {buffer_id} should track its processed epoch"
                
            # Verify epoch indices
            processed_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id][0]
            assert processed_idx == epoch_start, \
                f"Buffer {buffer_id} should track correct start index {epoch_start}, got {processed_idx}"
        
        # Add more data and trim buffer
        additional_data = data[:, initial_data_size:initial_data_size + 20 * sampling_rate]
        data_manager.add_to_data_processing_buffer(additional_data)
        data_manager.etd_buffer_manager.trim_buffer(points_to_remove=20 * sampling_rate)  # Trim 20 seconds of data
        
        # Process second round of epochs after trimming
        second_round_sleep_stages = []
        for buffer_id in range(6):
            # Calculate epoch indices for second round
            epoch_start = initial_data_size + (buffer_id * points_per_step)
            epoch_end = epoch_start + points_per_epoch
            
            # Process epoch
            sleep_stage = data_manager._process_epoch(
                start_idx_abs=epoch_start,
                end_idx_abs=epoch_end,
                buffer_id=buffer_id
            )
            second_round_sleep_stages.append(sleep_stage)
            
            # Verify sleep stage was processed
            assert sleep_stage is not None, f"Buffer {buffer_id} should process second epoch successfully"
            
            # Verify buffer tracking
            assert len(data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id]) == 2, \
                f"Buffer {buffer_id} should track both processed epochs"
                
            # Verify epoch indices after trimming
            second_processed_idx = data_manager.matrix_of_round_robin_processed_epoch_start_indices_abs[buffer_id][1]
            assert second_processed_idx == epoch_start, \
                f"Buffer {buffer_id} should track correct second start index {epoch_start}, got {second_processed_idx}"
        
        # Verify overlap data is preserved
        # Check that each buffer can access its required overlap data
        for buffer_id in range(6):
            # Get the overlap period for this buffer
            overlap_start = initial_data_size + (buffer_id * points_per_step)
            overlap_end = overlap_start + points_per_epoch
            
            # Verify we can access the overlap data
            timestamps = data_manager._get_etd_timestamps()
            overlap_timestamps = timestamps[overlap_start:overlap_end]
            
            # Check timestamp continuity in overlap period
            expected_interval = 1.0 / sampling_rate
            for i in range(1, len(overlap_timestamps)):
                actual_interval = overlap_timestamps[i] - overlap_timestamps[i-1]
                assert abs(actual_interval - expected_interval) < 1e-6, \
                    f"Buffer {buffer_id} overlap period has timestamp discontinuity at index {i}"
        
        # Verify buffer size is maintained
        final_buffer_size = data_manager._get_total_data_points_etd()
        expected_buffer_size = 35 * sampling_rate  # 35 seconds
        assert final_buffer_size == expected_buffer_size, \
            f"Buffer should maintain size of {expected_buffer_size}, got {final_buffer_size}"
        
        # Test total streamed samples tracking
        expected_total_samples = initial_data_size + 20 * sampling_rate
        assert data_manager.etd_buffer_manager.total_streamed_samples == expected_total_samples, \
            f"Total streamed samples should be {expected_total_samples}, got {data_manager.etd_buffer_manager.total_streamed_samples}"

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v']) 