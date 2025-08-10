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
        from gssc_local.montage import Montage
        data_manager = DataManager(board, sampling_rate, Montage.minimal_sleep_montage())
        
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
            timestamp_board_key=timestamp_channel,
            channel_count=len(electrode_and_timestamp_channels),
            electrode_and_timestamp_board_keys=electrode_and_timestamp_channels
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
        data_manager.etd_buffer_manager.select_channel_data_and_add(initial_data_stream)
        
        # Verify initial buffer size
        initial_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
        assert initial_buffer_size == initial_data_size, \
            f"Initial buffer size should be {initial_data_size}, got {initial_buffer_size}"
            
        # Add more data to exceed 35 seconds
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.etd_buffer_manager.select_channel_data_and_add(additional_data_stream)
        
        # Try to trim buffer before processing any epochs - should not trim
        next_epoch_start_idx_abs, _ = data_manager._get_next_epoch_indices(0)  # Next buffer to process
        data_manager.etd_buffer_manager.trim_buffer(next_epoch_start_idx_abs)
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
            next_epoch_start_idx_abs, _ = data_manager._get_next_epoch_indices(data_manager.last_processed_buffer + 1)
            data_manager.etd_buffer_manager.trim_buffer(next_epoch_start_idx_abs)
            
            # Verify buffer size after each trim
            current_buffer_size = data_manager.etd_buffer_manager._get_total_data_points()
            assert current_buffer_size >= min_buffer_size, \
                f"Buffer should maintain at least {min_buffer_size} points after processing buffer {buffer_id}, got {current_buffer_size}"
            
            # Verify data continuity
            timestamps = data_manager.etd_buffer_manager.electrode_and_timestamp_data_keyed.get_by_key(data_manager.etd_buffer_manager.timestamp_board_key)
            data_manager.etd_buffer_manager._verify_timestamp_continuity(timestamps)
            
            # Verify offset tracking
            assert data_manager.etd_buffer_manager.offset >= 0, "Buffer offset should be non-negative"
            
            # Verify total streamed samples tracking
            assert data_manager.etd_buffer_manager.total_streamed_samples == initial_data_size + 20 * sampling_rate, \
                f"Total streamed samples should be {initial_data_size + 20 * sampling_rate}, got {data_manager.etd_buffer_manager.total_streamed_samples}"
            
            # assert that the timestamp channel is included
            assert data_manager.etd_buffer_manager.timestamp_board_key == metadata['timestamp_channel'], \
                "Timestamp board key should match expected timestamp channel"
            # assert that the timestamp channel first value is a timestamp
            timestamp_data = data_manager.etd_buffer_manager.electrode_and_timestamp_data_keyed.get_by_key(data_manager.etd_buffer_manager.timestamp_board_key)
            first_timestamp = timestamp_data[0]
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
        data_manager.etd_buffer_manager.select_channel_data_and_add(initial_data_stream)
        
        # Test absolute to relative conversion before trimming
        test_absolute_idx = 20 * sampling_rate  # 20 seconds into the data
        relative_idx = data_manager.etd_buffer_manager._adjust_index_with_offset(test_absolute_idx, to_etd=True)
        assert relative_idx == test_absolute_idx, \
            f"Before trimming: absolute index {test_absolute_idx} should map to relative index {test_absolute_idx}, got {relative_idx}"
        
        # Add more data and trim buffer
        additional_data = data[initial_data_size:initial_data_size + 20 * sampling_rate, :]  # shape: (n_samples, n_channels)
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.etd_buffer_manager.select_channel_data_and_add(additional_data_stream)
        next_epoch_start_idx_abs, _ = data_manager._get_next_epoch_indices(data_manager.last_processed_buffer + 1)
        data_manager.etd_buffer_manager.trim_buffer(next_epoch_start_idx_abs)  # Trim buffer to max_buffer_size
        
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
        data_manager.etd_buffer_manager.select_channel_data_and_add(initial_data_stream)
    
        # Process first epoch
        first_epoch_start = 0
        first_epoch_end = points_per_epoch
        initial_epochs_count = data_manager.epochs_scored
        initial_buffer_epoch_count = data_manager.epochs_processed_count_per_buffer[0]
        
        data_manager.manage_epoch(
            buffer_id=0,
            epoch_start_idx_abs=first_epoch_start,
            epoch_end_idx_abs=first_epoch_end
        )
    
        # Add more data and process second epoch before trimming
        additional_data = data[initial_data_size:initial_data_size + 25 * sampling_rate, :]  # shape: (n_samples, n_channels) - increased to 25 seconds
        additional_data_stream = transform_to_stream_format(additional_data)  # transform to (n_channels, n_samples)
        data_manager.etd_buffer_manager.select_channel_data_and_add(additional_data_stream)
    
        # Process second epoch before trimming  
        second_epoch_start = points_per_epoch
        second_epoch_end = 2 * points_per_epoch
        
        data_manager.manage_epoch(
            buffer_id=0,
            epoch_start_idx_abs=second_epoch_start,
            epoch_end_idx_abs=second_epoch_end
        )
    
        # Now trim buffer (after all epochs that need the old data have been processed)
        next_epoch_start_idx_abs, _ = data_manager._get_next_epoch_indices(data_manager.last_processed_buffer + 1)
        data_manager.etd_buffer_manager.trim_buffer(next_epoch_start_idx_abs)
    
        # Verify sleep stage processing continues by checking epoch count
        assert data_manager.epochs_processed_count_per_buffer[0] == initial_buffer_epoch_count + 2, \
            "Both epochs should be tracked in buffer 0"
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
        data_manager.etd_buffer_manager.select_channel_data_and_add(gap_data_stream)
    
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
        
        # Verify processed epochs count is updated correctly  
        assert data_manager.epochs_processed_count_per_buffer[0] == 2, \
            "Should track both processed epochs in buffer 0"
            
        # Verify last epoch indices are correct after trimming
        last_epoch_tuple = data_manager.last_processed_epoch_per_buffer[0]
        assert last_epoch_tuple is not None, "Buffer 0 should have a last processed epoch"
        last_epoch_start_idx = last_epoch_tuple[0]  # Extract start index from tuple
        assert last_epoch_start_idx == second_epoch_start, \
            f"Last epoch start index should be {second_epoch_start}, got {last_epoch_start_idx}"

    def test_round_robin_integration(self, data_manager, test_data):
        """Test realistic streaming with round-robin processing and buffer trimming.
        Simulates how epochs would be processed in real-time streaming scenarios.
        """
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        points_per_epoch = data_manager.points_per_epoch
        points_per_step = data_manager.points_per_step
        
        # Track processed epochs per buffer
        epochs_processed_per_buffer = [0] * 6
        data_start_idx = 0
        chunk_size = 10 * sampling_rate  # Add data in 10-second chunks
        
        # Simulate streaming: add data chunks and process epochs as they become available
        for cycle in range(8):  # 8 cycles = 80 seconds of streaming
            # Simulate getting new data from the stream
            chunk_end_idx = data_start_idx + chunk_size
            if chunk_end_idx > len(data):
                break
                
            chunk_data = data[data_start_idx:chunk_end_idx, :]
            chunk_data_stream = transform_to_stream_format(chunk_data)

            # add simulated stream data to buffer
            data_manager.etd_buffer_manager.select_channel_data_and_add(chunk_data_stream)
            data_start_idx = chunk_end_idx
            
            print(f"\nCycle {cycle + 1}: Added data chunk, total samples: {data_manager.etd_buffer_manager.total_streamed_samples}")
            
            # Check which buffers are ready to process their next epoch
            for buffer_id in range(6):
                # Calculate the next epoch for this buffer
                epoch_number = epochs_processed_per_buffer[buffer_id]
                epoch_start = buffer_id * points_per_step + epoch_number * points_per_epoch
                epoch_end = epoch_start + points_per_epoch
                
                # Process epoch if we have enough data
                if epoch_end <= data_manager.etd_buffer_manager.total_streamed_samples:
                    print(f"  Processing buffer {buffer_id}, epoch {epoch_number + 1} (samples {epoch_start}-{epoch_end})")
                    
                    data_manager.manage_epoch(
                        buffer_id=buffer_id,
                        epoch_start_idx_abs=epoch_start,
                        epoch_end_idx_abs=epoch_end
                    )
                    
                    epochs_processed_per_buffer[buffer_id] += 1
                    
                    # Verify tracking
                    expected_epochs = epochs_processed_per_buffer[buffer_id]
                    actual_epochs = data_manager.epochs_processed_count_per_buffer[buffer_id]
                    assert actual_epochs == expected_epochs, \
                        f"Buffer {buffer_id} should track {expected_epochs} epochs, got {actual_epochs}"
                    
                    # Try to trim buffer after each epoch is processed (realistic streaming)
                    pre_trim_size = data_manager._get_total_data_points_etd()
                    next_epoch_start_idx_abs, _ = data_manager._get_next_epoch_indices(data_manager.last_processed_buffer + 1)
                    data_manager.etd_buffer_manager.trim_buffer(next_epoch_start_idx_abs)
                    post_trim_size = data_manager._get_total_data_points_etd()
                    
                    if post_trim_size < pre_trim_size:
                        print(f"    Trimmed buffer after epoch: {pre_trim_size} -> {post_trim_size} samples")
                    
                    # Monitor buffer size after each epoch (realistic streaming)
                    max_buffer_size = data_manager.etd_buffer_manager.max_buffer_size
                    if post_trim_size > max_buffer_size * 2:
                        print(f"    Warning: Buffer size {post_trim_size} exceeds 2x max ({max_buffer_size * 2})")
        
        # Verify final state
        print(f"\nFinal state:")
        print(f"  Epochs processed per buffer: {epochs_processed_per_buffer}")
        print(f"  Total buffer size: {data_manager._get_total_data_points_etd()}")
        print(f"  Total streamed samples: {data_manager.etd_buffer_manager.total_streamed_samples}")
        
        # All buffers should have processed at least one epoch
        assert all(count > 0 for count in epochs_processed_per_buffer), \
            f"All buffers should process at least one epoch, got {epochs_processed_per_buffer}"
        
        # Buffer 0 should have processed the most epochs (starts first)
        assert epochs_processed_per_buffer[0] >= max(epochs_processed_per_buffer), \
            f"Buffer 0 should process the most epochs, got {epochs_processed_per_buffer}"
        
        # Verify epoch indices are tracked correctly
        for buffer_id in range(6):
            expected_epochs = epochs_processed_per_buffer[buffer_id]
            actual_epochs = data_manager.epochs_processed_count_per_buffer[buffer_id]
            assert actual_epochs == expected_epochs, \
                f"Buffer {buffer_id} should have processed {expected_epochs} epochs, got {actual_epochs}"
            
            if expected_epochs > 0:
                # Check that last processed epoch exists and has reasonable indices
                last_epoch_tuple = data_manager.last_processed_epoch_per_buffer[buffer_id]
                assert last_epoch_tuple is not None, f"Buffer {buffer_id} should have a last processed epoch"
                
                last_epoch_start = last_epoch_tuple[0]
                last_epoch_end = last_epoch_tuple[1]
                
                # Verify epoch is properly sized (30 seconds worth of data)
                assert last_epoch_end - last_epoch_start == points_per_epoch, \
                    f"Buffer {buffer_id} last epoch should span {points_per_epoch} points, got {last_epoch_end - last_epoch_start}"
                
                # Verify epoch start is reasonable for this buffer (accounting for processed epochs)
                expected_last_start = buffer_id * points_per_step + (expected_epochs - 1) * points_per_epoch
                assert last_epoch_start == expected_last_start, \
                    f"Buffer {buffer_id} last epoch should start at {expected_last_start}, got {last_epoch_start}"
        
        # Verify final buffer state
        final_buffer_size = data_manager._get_total_data_points_etd()
        assert final_buffer_size > 0, "Buffer should contain some data"
        assert final_buffer_size <= data_manager.etd_buffer_manager.max_buffer_size * 3, \
            f"Buffer should not exceed 3x max size, got {final_buffer_size}"

    def test_realistic_streaming_integration(self, test_data):
        """Integration test using ReceivedStreamedDataHandler - tests the actual production code path.
        
        This test simulates realistic streaming by:
        1. Using the same ReceivedStreamedDataHandler pattern as production
        2. Adding data in small chunks (like real OpenBCI streaming)
        3. Validating buffer constraints and processing outcomes
        4. Catching bugs that only manifest in the real integration path
        """
        from gssc_local.realtime_with_restart.received_stream_data_handler import ReceivedStreamedDataHandler
        from gssc_local.realtime_with_restart.board_manager import BoardManager
        import logging
        
        data, metadata = test_data
        sampling_rate = metadata['sampling_rate']
        
        # Create a mock board manager for the handler
        class MockBoardManager:
            def __init__(self, board_shim, sampling_rate):
                self.board_shim = board_shim
                self.sampling_rate = sampling_rate
                self.board_timestamp_channel = BoardShim.get_timestamp_channel(master_board_id)
        
        # Create mock logger
        logger = logging.getLogger(__name__)
        
        # Use the same board setup as the data_manager fixture
        master_board_id = BoardIds.CYTON_DAISY_BOARD
        playback_board_id = BoardIds.PLAYBACK_FILE_BOARD
        
        # Create a temporary file for playback data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            # Save test data to temp file
            np.savetxt(temp_file.name, data.T, delimiter=',')  # Transpose for BrainFlow format
            
        try:
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
            
            # Create mock board manager and handler
            from gssc_local.montage import Montage
            test_montage = Montage.minimal_sleep_montage()
            mock_board_manager = MockBoardManager(board, sampling_rate)
            handler = ReceivedStreamedDataHandler(mock_board_manager, logger, test_montage)
            
            # Track metrics
            initial_buffer_size = handler.data_manager.etd_buffer_manager.max_buffer_size
            max_observed_buffer_size = 0
            epochs_processed_count = 0
            trim_operations = 0
            
            # Simulate realistic streaming: add data in small chunks
            chunk_size = int(0.5 * sampling_rate)  # 0.5 second chunks (realistic for OpenBCI)
            total_chunks = min(len(data) // chunk_size, 120)  # Limit to ~60 seconds of data
            
            print(f"\nRunning realistic streaming simulation:")
            print(f"  Chunk size: {chunk_size} samples ({chunk_size/sampling_rate:.1f}s)")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Max buffer size: {initial_buffer_size}")
            
            for chunk_idx in range(total_chunks):
                # Get chunk data
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(data))
                chunk_data = data[start_idx:end_idx, :].T  # Transpose to (n_channels, n_samples)
                
                # Track buffer size before processing
                pre_process_buffer_size = handler.data_manager._get_total_data_points_etd()
                pre_process_epochs = handler.data_manager.epochs_scored
                
                # Process chunk using production code path
                handler.process_board_data_chunk(chunk_data)
                
                # Track metrics after processing
                post_process_buffer_size = handler.data_manager._get_total_data_points_etd()
                post_process_epochs = handler.data_manager.epochs_scored
                max_observed_buffer_size = max(max_observed_buffer_size, post_process_buffer_size)
                
                # Count trim operations
                if post_process_buffer_size < pre_process_buffer_size:
                    trim_operations += 1
                
                # Count epochs processed
                if post_process_epochs > pre_process_epochs:
                    epochs_processed_count += 1
                    print(f"  Chunk {chunk_idx+1}: Processed epoch, buffer size: {post_process_buffer_size}")
                
                # CRITICAL ASSERTIONS - These would have caught the bugs
                assert post_process_buffer_size <= initial_buffer_size * 2.0, \
                    f"Buffer size {post_process_buffer_size} exceeds 2x max ({initial_buffer_size * 2.0}) at chunk {chunk_idx+1}"
                
                assert handler.data_manager.etd_buffer_manager.offset >= 0, \
                    f"Buffer offset should be non-negative, got {handler.data_manager.etd_buffer_manager.offset}"
                
                assert handler.data_manager.etd_buffer_manager.total_streamed_samples >= handler.data_manager.etd_buffer_manager.offset, \
                    f"Total streamed samples ({handler.data_manager.etd_buffer_manager.total_streamed_samples}) should be >= offset ({handler.data_manager.etd_buffer_manager.offset})"
                
                # Verify index consistency
                if post_process_buffer_size > 0:
                    # Test that we can still access valid indices
                    try:
                        last_valid_abs_idx = handler.data_manager.etd_buffer_manager.total_streamed_samples - 1
                        relative_idx = handler.data_manager.etd_buffer_manager._adjust_index_with_offset(last_valid_abs_idx, to_etd=True)
                        assert 0 <= relative_idx < post_process_buffer_size, \
                            f"Index translation failed: abs {last_valid_abs_idx} -> rel {relative_idx}, buffer size {post_process_buffer_size}"
                    except ValueError as e:
                        pytest.fail(f"Index translation error at chunk {chunk_idx+1}: {e}")
            
            # Final validation
            print(f"\nSimulation completed successfully:")
            print(f"  Epochs processed: {epochs_processed_count}")
            print(f"  Trim operations: {trim_operations}")
            print(f"  Max buffer size observed: {max_observed_buffer_size}")
            print(f"  Final buffer size: {handler.data_manager._get_total_data_points_etd()}")
            
            # Verify we processed some epochs
            assert epochs_processed_count > 0, "Should have processed at least one epoch"
            
            # Verify trimming occurred (indicates buffer management is working)
            assert trim_operations > 0, "Should have performed buffer trimming operations"
            
            # Verify buffer stayed within reasonable bounds
            assert max_observed_buffer_size <= initial_buffer_size * 2.5, \
                f"Max buffer size {max_observed_buffer_size} should stay within 2.5x limit ({initial_buffer_size * 2.5})"
            
            # Test round-robin progression by checking that multiple buffers processed epochs
            buffers_with_epochs = sum(1 for count in handler.data_manager.epochs_processed_count_per_buffer if count > 0)
            assert buffers_with_epochs >= 2, f"Expected at least 2 buffers to process epochs, got {buffers_with_epochs}"
            
        finally:
            # Cleanup
            try:
                board.stop_stream()
                board.release_session()
            except:
                pass
            try:
                os.unlink(temp_file.name)
            except:
                pass
            try:
                handler.data_manager.cleanup()
            except:
                pass

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v']) 