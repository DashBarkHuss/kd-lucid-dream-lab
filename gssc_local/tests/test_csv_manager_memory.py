"""
Memory usage tests for CSVManager class.

This module contains test cases for:
- Peak memory usage monitoring
- Memory leak detection
- Buffer size compliance
"""

import pytest
import numpy as np
import pandas as pd
import psutil
import os
import gc
import tempfile
import logging
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.board_shim import BrainFlowInputParams

from gssc_local.realtime_with_restart.export.csv.manager import CSVManager
from gssc_local.tests.test_utils import create_brainflow_test_data

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

def get_process_memory():
    """Get the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

@pytest.fixture
def csv_manager():
    """Fixture providing a CSVManager instance with small buffer sizes."""
    input_params = BrainFlowInputParams()
    return CSVManager(
        board_shim=BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params),
        main_buffer_size=1000,  # Small buffer for testing
        sleep_stage_buffer_size=10  # Small buffer for testing
    )

def test_peak_memory_usage(csv_manager):
    """Test that memory usage stays below configured limits during large data processing."""
    gc.collect()
    before = get_process_memory()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path

        # Generate 10 minutes of test data at 125 Hz
        data, _ = create_brainflow_test_data(
            duration_seconds=600,  # 10 minutes
            sampling_rate=125,     # 125 Hz
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD
        )

        # Process data in chunks to simulate real-time processing
        chunk_size = 1250  # 10 seconds of data
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            csv_manager.add_data_to_buffer(chunk.T, is_initial=(i == 0))

            # Add sleep stage data every minute
            if i % (125 * 60) == 0:  # Every minute
                start_time = data[i, BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
                end_time = data[min(i + chunk_size, len(data) - 1), 
                              BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
                csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i // chunk_size, start_time, end_time)

        # Final cleanup
        csv_manager.save_all_and_cleanup()
    gc.collect()
    after = get_process_memory()
    print(f"[test_peak_memory_usage] Memory used: {after - before:.2f} MB (before: {before:.2f}, after: {after:.2f})")
    assert (after - before) < 100  # Adjust this limit as needed

def test_memory_leak_detection(csv_manager):
    """Test for memory leaks by repeatedly processing data and checking memory usage."""
    memory_usage = []
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths BEFORE generating data
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path

        # Generate different chunks of test data for each iteration
        # This better simulates real usage where we process different data each time
        for i in range(5):  # Process 5 times
            gc.collect()
            before = get_process_memory()
            
            # Generate new data for each iteration with sequential timestamps
            data, _ = create_brainflow_test_data(
                duration_seconds=60,  # 1 minute
                sampling_rate=125,    # 125 Hz
                add_noise=False,
                board_id=BoardIds.CYTON_DAISY_BOARD,
                start_time=1700000000.1 + (i * 60)  # Each chunk starts 60 seconds after the previous
            )
            
            # Process data - each chunk is treated as initial since it's new data
            csv_manager.add_data_to_buffer(data.T, is_initial=True)
            csv_manager.save_main_buffer_to_csv()
            
            # Add sleep stage data
            start_time = data[0, BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
            end_time = data[-1, BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
            csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i, start_time, end_time)
            csv_manager.save_sleep_stages_to_csv()
            
            gc.collect()
            after = get_process_memory()
            memory_usage.append(after - before)
            print(f"[test_memory_leak_detection] Iteration {i+1}/5 memory used: {after - before:.2f} MB (before: {before:.2f}, after: {after:.2f})")
    
    max_memory_increase = max(memory_usage)
    assert max_memory_increase < 10.0, f"Memory leak detected: {max_memory_increase:.2f} MB increase"

def test_buffer_size_compliance(csv_manager):
    """Test that buffer sizes are strictly enforced."""
    gc.collect()
    before = get_process_memory()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path

        # Generate data that would exceed buffer size
        data, _ = create_brainflow_test_data(
            duration_seconds=10,  # 10 seconds
            sampling_rate=125,    # 125 Hz
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD
        )

        # Process data and verify buffer size compliance
        csv_manager.add_data_to_buffer(data.T, is_initial=True)
        assert len(csv_manager.main_csv_buffer) <= csv_manager.main_buffer_size, \
            f"Main buffer size exceeded: {len(csv_manager.main_csv_buffer)} > {csv_manager.main_buffer_size}"

        # Add sleep stages and verify buffer size compliance
        for i in range(20):  # Try to add more than buffer size
            start_time = i * 30.0
            end_time = (i + 1) * 30.0
            csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i, start_time, end_time)
            assert len(csv_manager.sleep_stage_buffer) <= csv_manager.sleep_stage_buffer_size, \
                f"Sleep stage buffer size exceeded: {len(csv_manager.sleep_stage_buffer)} > {csv_manager.sleep_stage_buffer_size}"
    gc.collect()
    after = get_process_memory()
    print(f"[test_buffer_size_compliance] Memory used: {after - before:.2f} MB (before: {before:.2f}, after: {after:.2f})")

def test_large_file_handling(csv_manager):
    """Test memory usage when handling very large files."""
    gc.collect()
    before = get_process_memory()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        final_output_path = os.path.join(temp_dir, "test_data.merged.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path

        # Generate 30 minutes of test data
        data, _ = create_brainflow_test_data(
            duration_seconds=1800,  # 30 minutes
            sampling_rate=125,      # 125 Hz
            add_noise=False,
            board_id=BoardIds.CYTON_DAISY_BOARD
        )

        # Process data in chunks
        chunk_size = 1250  # 10 seconds of data
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            csv_manager.add_data_to_buffer(chunk.T, is_initial=(i == 0))

        # Add sleep stages every minute
        for i in range(0, len(data), 125 * 60):  # Every minute
            start_time = data[i, BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
            end_time = data[min(i + 125 * 60, len(data) - 1), 
                          BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD)]
            csv_manager.add_sleep_stage_to_sleep_stage_csv(2.0, i // (125 * 60), start_time, end_time)

        # Merge files
        csv_manager.merge_files(main_csv_path, sleep_stage_csv_path, final_output_path)

        # Verify final file exists and has correct content
        assert os.path.exists(final_output_path)
        merged_df = pd.read_csv(final_output_path, delimiter='\t')
        assert len(merged_df) == len(data)
        assert 'sleep_stage' in merged_df.columns
        assert 'buffer_id' in merged_df.columns
    gc.collect()
    after = get_process_memory()
    print(f"[test_large_file_handling] Memory used: {after - before:.2f} MB (before: {before:.2f}, after: {after:.2f})") 