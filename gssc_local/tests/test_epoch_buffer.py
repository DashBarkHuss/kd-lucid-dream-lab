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
from gssc_local.tests.test_utils import create_brainflow_test_data
from brainflow.board_shim import BoardShim, BoardIds
from unittest.mock import MagicMock

class TestEpochBuffer:
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager instance with mock board."""
        board_id = BoardIds.CYTON_DAISY_BOARD
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        
        # Create a mock board with the real board's channel configuration
        mock_board = MagicMock()
        mock_board.get_board_id.return_value = board_id
        mock_board.get_exg_channels.return_value = eeg_channels
        mock_board.get_timestamp_channel.return_value = timestamp_channel
        
        return DataManager(mock_board, sampling_rate)

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
        initial_data = data[:initial_data_size, :].T  # shape: (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Verify initial buffer size
        initial_buffer_size = data_manager._get_total_data_points_etd()
        assert initial_buffer_size == initial_data_size, \
            f"Initial buffer size should be {initial_data_size}, got {initial_buffer_size}"
            
        # Add more data to exceed 35 seconds
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :].T  # shape: (n_channels, n_samples)
        data_manager.add_to_data_processing_buffer(additional_data)
        
        # Call buffer trimming
        data_manager._trim_etd_buffer()
        
        # Verify buffer was trimmed to correct size
        final_buffer_size = data_manager._get_total_data_points_etd()
        assert final_buffer_size == min_buffer_size, \
            f"Buffer should be trimmed to {min_buffer_size}, got {final_buffer_size}"
            
        # Verify data continuity
        # Get timestamps from buffer
        timestamps = data_manager._get_etd_timestamps()
        data_manager._verify_timestamp_continuity(timestamps)
        
        # Verify offset tracking
        assert data_manager.etd_offset > 0, "Buffer offset should be updated after trimming"
        
        # Verify total streamed samples tracking
        assert data_manager.total_streamed_samples_since_start == initial_data_size + 20 * sampling_rate, \
            "Total streamed samples should include all data, even after trimming"

    def test_index_translation(self, data_manager, test_data):
        """Test that index translation works correctly with buffer trimming."""
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Add initial data (40 seconds)
        initial_data_size = 40 * sampling_rate
        initial_data = data[:, :initial_data_size]
        data_manager.add_to_data_processing_buffer(initial_data)
        
        # Test absolute to relative conversion before trimming
        test_absolute_idx = 20 * sampling_rate  # 20 seconds into the data
        relative_idx = data_manager._adjust_index_with_etd_offset(test_absolute_idx, to_etd=True)
        assert relative_idx == test_absolute_idx, \
            f"Before trimming: absolute index {test_absolute_idx} should map to relative index {test_absolute_idx}, got {relative_idx}"
            
        # Add more data and trim buffer
        additional_data = data[:, initial_data_size:initial_data_size + 20 * sampling_rate]
        data_manager.add_to_data_processing_buffer(additional_data)
        data_manager._trim_etd_buffer()
        
        # Test absolute to relative conversion after trimming
        # The absolute index should be adjusted by the offset
        relative_idx = data_manager._adjust_index_with_etd_offset(test_absolute_idx, to_etd=True)
        expected_relative_idx = test_absolute_idx - data_manager.etd_offset
        assert relative_idx == expected_relative_idx, \
            f"After trimming: absolute index {test_absolute_idx} should map to relative index {expected_relative_idx}, got {relative_idx}"
            
        # Test relative to absolute conversion
        absolute_idx = data_manager._adjust_index_with_etd_offset(relative_idx, to_etd=False)
        assert absolute_idx == test_absolute_idx, \
            f"Relative index {relative_idx} should map back to absolute index {test_absolute_idx}, got {absolute_idx}"
            
        # Test edge cases
        # Test index at buffer start
        start_relative = data_manager._adjust_index_with_etd_offset(data_manager.etd_offset, to_etd=True)
        assert start_relative == 0, \
            f"Buffer start should map to relative index 0, got {start_relative}"
            
        # Test index at buffer end
        end_absolute = data_manager.total_streamed_samples_since_start - 1
        end_relative = data_manager._adjust_index_with_etd_offset(end_absolute, to_etd=True)
        expected_end_relative = data_manager._get_total_data_points_etd() - 1
        assert end_relative == expected_end_relative, \
            f"Buffer end should map to relative index {expected_end_relative}, got {end_relative}"
            
        # Test invalid indices
        with pytest.raises(ValueError):
            data_manager._adjust_index_with_etd_offset(-1, to_etd=True)
            
        with pytest.raises(ValueError):
            data_manager._adjust_index_with_etd_offset(data_manager.total_streamed_samples_since_start, to_etd=True)

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
        data_manager._trim_etd_buffer()
        
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
            epoch_start_idx_abs=data_manager.total_streamed_samples_since_start - points_per_epoch,
            epoch_end_idx_abs=data_manager.total_streamed_samples_since_start
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
        data_manager._trim_etd_buffer()
        
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
        
        # Verify total streamed samples tracking
        expected_total_samples = initial_data_size + 20 * sampling_rate
        assert data_manager.total_streamed_samples_since_start == expected_total_samples, \
            f"Total streamed samples should be {expected_total_samples}, got {data_manager.total_streamed_samples_since_start}" 