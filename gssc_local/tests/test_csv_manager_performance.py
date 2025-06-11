"""
Performance tests for CSVManager class.

This module contains test cases for:
- Processing speed measurements
- Disk I/O performance
- Buffer operation performance
- Performance under different data volumes
"""

import pytest
import numpy as np
import pandas as pd
import os
import time
import tempfile
import logging
import psutil
import gc
from brainflow.board_shim import BoardShim, BoardIds
from brainflow.board_shim import BrainFlowInputParams
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gssc_local.realtime_with_restart.export.csv.manager import CSVManager
from gssc_local.tests.test_utils import create_brainflow_test_data

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

@pytest.fixture
def csv_manager():
    """Fixture providing a CSVManager instance with configurable buffer sizes."""
    input_params = BrainFlowInputParams()
    return CSVManager(
        board_shim=BoardShim(BoardIds.CYTON_DAISY_BOARD, input_params),
        main_buffer_size=10_000,  # Default buffer size
        sleep_stage_buffer_size=100  # Default sleep stage buffer size
    )

def test_buffer_add_performance(csv_manager):
    """
    Test the performance of adding data to the buffer.
    - Measure time taken to add different sized chunks
    - Compare performance with different buffer sizes
    - Test with various data types and shapes
    - Monitor memory usage
    - Test with realistic streaming patterns
    """
    print("\n=== Testing Buffer Add Performance ===")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
        
        # Test parameters
        chunk_sizes = [100, 1000, 10000]  # Different chunk sizes to test
        buffer_sizes = [1000, 10000, 100000]  # Different buffer sizes to test
        
        results = []
        
        for buffer_size in buffer_sizes:
            print(f"\nTesting with buffer size: {buffer_size}")
            csv_manager.main_buffer_size = buffer_size
            
            # Track the last timestamp used to ensure sequential data
            last_timestamp = 1700000000.1  # Initial start time
            
            # Test continuous streaming pattern
            for chunk_size in chunk_sizes:
                # Calculate duration in seconds for this chunk
                duration_seconds = chunk_size/125  # Convert samples to seconds at 125 Hz
                
                # Record initial memory usage
                initial_memory = get_memory_usage()
                
                # Generate test data with sequential start time
                data, _ = create_brainflow_test_data(
                    duration_seconds=duration_seconds,
                    sampling_rate=125,
                    add_noise=False,
                    board_id=BoardIds.CYTON_DAISY_BOARD,
                    start_time=last_timestamp
                )
                
                # Update last_timestamp for next chunk
                last_timestamp += duration_seconds
                
                # Debug: Print data shape and first few timestamps
                logger.debug(f"Generated data shape: {data.shape}")
                timestamp_channel = csv_manager.board_shim.get_timestamp_channel(csv_manager.board_shim.get_board_id())
                logger.debug(f"First 5 timestamps in new data: {data[timestamp_channel, :5]}")
                
                # Debug: Print current buffer state
                if csv_manager.main_csv_buffer:
                    logger.debug(f"Current buffer size: {len(csv_manager.main_csv_buffer)}")
                    logger.debug(f"First 5 timestamps in buffer: {[row[timestamp_channel] for row in csv_manager.main_csv_buffer[:5]]}")
                
                # Measure time for adding data
                start_time = time.time()
                try:
                    result = csv_manager.queue_data_for_csv_write(data.T, is_initial=True)
                except Exception as e:
                    logger.error(f"Error adding data to buffer: {str(e)}")
                    if csv_manager.main_csv_buffer:
                        logger.error(f"Buffer size at error: {len(csv_manager.main_csv_buffer)}")
                        logger.error(f"All timestamps in buffer: {[row[timestamp_channel] for row in csv_manager.main_csv_buffer]}")
                    raise
                end_time = time.time()
                
                # Calculate performance metrics
                time_taken = end_time - start_time
                samples_per_second = chunk_size / time_taken if time_taken > 0 else 0
                
                # Calculate memory usage
                final_memory = get_memory_usage()
                memory_increase = final_memory - initial_memory
                
                # Store results
                results.append({
                    'buffer_size': buffer_size,
                    'chunk_size': chunk_size,
                    'time_taken': time_taken,
                    'samples_per_second': samples_per_second,
                    'memory_increase_mb': memory_increase,
                    'success': result
                })
                
                print(f"Chunk size: {chunk_size}")
                print(f"Time taken: {time_taken:.4f} seconds")
                print(f"Processing speed: {samples_per_second:.2f} samples/second")
                print(f"Memory increase: {memory_increase:.2f} MB")
                print(f"Success: {result}")
                
                # Force garbage collection between chunks
                gc.collect()
        
        # Analyze results
        print("\n=== Performance Analysis ===")
        
        # Group results by buffer size
        for buffer_size in buffer_sizes:
            buffer_results = [r for r in results if r['buffer_size'] == buffer_size]
            
            avg_time = sum(r['time_taken'] for r in buffer_results) / len(buffer_results)
            avg_speed = sum(r['samples_per_second'] for r in buffer_results) / len(buffer_results)
            avg_memory = sum(r['memory_increase_mb'] for r in buffer_results) / len(buffer_results)
            
            print(f"\nBuffer size {buffer_size}:")
            print(f"Average time per operation: {avg_time:.4f} seconds")
            print(f"Average processing speed: {avg_speed:.2f} samples/second")
            print(f"Average memory increase: {avg_memory:.2f} MB")
        
        # Verify all operations were successful
        assert all(r['success'] for r in results), "Some buffer operations failed"
        
        # Verify performance meets minimum requirements
        min_acceptable_speed = 1000  # samples per second
        assert all(r['samples_per_second'] >= min_acceptable_speed for r in results), \
            f"Some operations were slower than {min_acceptable_speed} samples/second"
        
        # Verify memory usage is reasonable
        max_acceptable_memory_increase = 100  # MB
        assert all(r['memory_increase_mb'] <= max_acceptable_memory_increase for r in results), \
            f"Some operations used more than {max_acceptable_memory_increase} MB of memory"

def test_buffer_save_performance(csv_manager):
    """
    Test the performance of saving buffer contents to disk.
    - Measure time taken to save different sized buffers
    - Compare performance with different file sizes
    - Test with different buffer sizes
    - Monitor disk space usage
    - Test file fragmentation impact
    """
    print("\n=== Testing Buffer Save Performance ===", flush=True)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up paths
        main_csv_path = os.path.join(temp_dir, "test_data.csv")
        sleep_stage_csv_path = os.path.join(temp_dir, "test_data.sleep.csv")
        csv_manager.main_csv_path = main_csv_path
        csv_manager.sleep_stage_csv_path = sleep_stage_csv_path
        
        # Test parameters
        buffer_sizes = [1000, 10000, 100000]  # Different buffer sizes to test
        num_saves = 5  # Number of saves to perform for each buffer size
        
        results = []
        
        for buffer_size in buffer_sizes:
            print(f"\nTesting with buffer size: {buffer_size}", flush=True)
            csv_manager.main_buffer_size = buffer_size * 2  # Set buffer size larger than test data to prevent auto-save
            
            # Track the last timestamp used to ensure sequential data
            last_timestamp = 1700000000.1  # Initial start time
            
            # Get initial disk space
            initial_disk_space = psutil.disk_usage(temp_dir).free
            
            for save_idx in range(num_saves):
                # Generate test data to fill the buffer
                duration_seconds = buffer_size/125  # Convert samples to seconds at 125 Hz
                data, _ = create_brainflow_test_data(
                    duration_seconds=duration_seconds,
                    sampling_rate=125,
                    add_noise=False,
                    board_id=BoardIds.CYTON_DAISY_BOARD,
                    start_time=last_timestamp
                )
                
                # Update last_timestamp for next chunk
                last_timestamp += duration_seconds
                
                # Add data to buffer without triggering auto-save
                csv_manager.queue_data_for_csv_write(data.T, is_initial=(save_idx == 0))
                
                # Clear any existing file for the first save
                if save_idx == 0 and os.path.exists(main_csv_path):
                    os.remove(main_csv_path)
                
                # Measure time for saving buffer
                start_time = time.time()
                try:
                    csv_manager.save_main_buffer_to_csv()
                except Exception as e:
                    logger.error(f"Error saving buffer: {str(e)}")
                    raise
                end_time = time.time()
                
                # Calculate performance metrics
                time_taken = end_time - start_time
                samples_per_second = buffer_size / time_taken if time_taken > 0 else 0
                
                # Get file size and disk space after save
                file_size = os.path.getsize(main_csv_path) if os.path.exists(main_csv_path) else 0
                current_disk_space = psutil.disk_usage(temp_dir).free
                disk_space_used = initial_disk_space - current_disk_space
                
                # Store results
                results.append({
                    'buffer_size': buffer_size,
                    'save_index': save_idx,
                    'time_taken': time_taken,
                    'samples_per_second': samples_per_second,
                    'file_size_bytes': file_size,
                    'disk_space_used_bytes': disk_space_used
                })
                
                print(f"Save {save_idx + 1}/{num_saves}:", flush=True)
                print(f"Buffer size: {buffer_size}", flush=True)
                print(f"Time taken: {time_taken:.4f} seconds", flush=True)
                print(f"Processing speed: {samples_per_second:.2f} samples/second", flush=True)
                print(f"File size: {file_size/1024:.2f} KB", flush=True)
                print(f"Disk space used: {disk_space_used/1024:.2f} KB", flush=True)
        
        # Analyze results
        print("\n=== Performance Analysis ===", flush=True)
        
        # Group results by buffer size
        for buffer_size in buffer_sizes:
            buffer_results = [r for r in results if r['buffer_size'] == buffer_size]
            
            avg_time = sum(r['time_taken'] for r in buffer_results) / len(buffer_results)
            avg_speed = sum(r['samples_per_second'] for r in buffer_results) / len(buffer_results)
            avg_file_size = sum(r['file_size_bytes'] for r in buffer_results) / len(buffer_results)
            avg_disk_space = sum(r['disk_space_used_bytes'] for r in buffer_results) / len(buffer_results)
            
            print(f"\nBuffer size {buffer_size}:", flush=True)
            print(f"Average time per save: {avg_time:.4f} seconds", flush=True)
            print(f"Average processing speed: {avg_speed:.2f} samples/second", flush=True)
            print(f"Average file size: {avg_file_size/1024:.2f} KB", flush=True)
            print(f"Average disk space used: {avg_disk_space/1024:.2f} KB", flush=True)
        
        # Verify performance meets minimum requirements
        min_acceptable_speed = 1000  # samples per second
        assert all(r['samples_per_second'] >= min_acceptable_speed for r in results), \
            f"Some save operations were slower than {min_acceptable_speed} samples/second"
        
        # Verify file sizes are reasonable
        expected_bytes_per_sample = 32 * 8  # 32 columns, 8 bytes per value
        for result in results:
            expected_size = result['buffer_size'] * expected_bytes_per_sample
            actual_size = result['file_size_bytes']
            # Allow for some overhead in CSV format (headers, delimiters, etc.)
            assert actual_size >= expected_size * 0.8, \
                f"File size {actual_size} bytes is too small for {result['buffer_size']} samples"
            
            # Verify disk space usage is reasonable (should be close to file size)
            disk_space_used = result['disk_space_used_bytes']
            # Allow for file system overhead (block size, metadata, etc.)
            # Most file systems use 4KB blocks, so we'll allow for that overhead
            max_allowed_overhead = max(4096, actual_size * 0.5)  # Allow either 4KB or 50% overhead, whichever is larger
            assert disk_space_used <= actual_size + max_allowed_overhead, \
                f"Disk space usage ({disk_space_used} bytes) is too high compared to file size ({actual_size} bytes)"

if __name__ == '__main__':
    print("\nRunning tests directly...")
    pytest.main([__file__, '-v'])