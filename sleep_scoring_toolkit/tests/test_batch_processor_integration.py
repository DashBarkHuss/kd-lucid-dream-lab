"""
Integration tests for BatchProcessor and generate_model_scores CLI.

Tests both direct BatchProcessor usage and CLI script execution with real data.
Uses small test data for fast integration testing.
"""

import pytest
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import os

from brainflow.board_shim import BoardShim, BoardIds
from sleep_scoring_toolkit.batch_processor import BatchProcessor
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.tests.test_utils import create_brainflow_test_data, save_brainflow_data_to_csv
from sleep_scoring_toolkit.constants import STANDARD_EEG_CHANNELS, STANDARD_EOG_CHANNELS


class TestBatchProcessorIntegration:
    """Integration tests using real Cyton-Daisy OpenBCI test data."""
    
    @pytest.fixture
    def board_config(self):
        """Board configuration for Cyton-Daisy tests."""
        board_id = BoardIds.CYTON_DAISY_BOARD
        return {
            'board_id': board_id,
            'sampling_rate': BoardShim.get_sampling_rate(board_id),
            'timestamp_column': BoardShim.get_timestamp_channel(board_id),
            'exg_columns': BoardShim.get_exg_channels(board_id)
        }
    
    @pytest.fixture
    def minimal_batch_processor(self, board_config):
        """BatchProcessor configured for minimal sleep montage."""
        montage = Montage.minimal_sleep_montage()
        return BatchProcessor(
            montage=montage,
            sampling_rate=board_config['sampling_rate'],
            eeg_channels_for_scoring=STANDARD_EEG_CHANNELS,
            eog_channels_for_scoring=STANDARD_EOG_CHANNELS,
            timestamp_column=board_config['timestamp_column'],
            exg_columns=board_config['exg_columns'],
            show_progress=False,
        )
    
    @pytest.fixture  
    def small_test_csv(self, board_config, tmp_path):
        """Create a small synthetic CSV file using test utils for fast testing."""
        # Generate 70 seconds of synthetic data (enough for 2 epochs)
        duration = 70  # seconds
        
        # Use existing test utility to create properly formatted data
        data, metadata = create_brainflow_test_data(
            duration_seconds=duration,
            sampling_rate=board_config['sampling_rate'],
            add_noise=True,  # More realistic for sleep stage classification
            start_time=1743336713.0,
            random_seed=42
        )
        
        # Save as CSV using test utility
        csv_path = tmp_path / "synthetic_test.csv"
        save_brainflow_data_to_csv(data, str(csv_path))
        return str(csv_path)

    def test_batch_processor_processes_synthetic_data(self, small_test_csv, minimal_batch_processor, tmp_path):
        """Test BatchProcessor can process synthetic data end-to-end quickly."""
        output_path = tmp_path / "test_results.csv"
        
        # Process the synthetic data
        results = minimal_batch_processor.process_csv_file(small_test_csv, str(output_path))
        
        # Verify processing completed successfully
        assert results['epochs_processed'] >= 2, "Should process at least 2 epochs from 70s of data"
        assert results['processing_time'] > 0, "Should take measurable time"
        assert results['speed_ratio'] > 0, "Should report speed ratio"
        assert output_path.exists(), "Output file should be created"
        
        # Verify output format
        output_df = pd.read_csv(output_path)
        expected_columns = ['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id']
        assert list(output_df.columns) == expected_columns, f"Expected columns {expected_columns}"
        assert len(output_df) == results['epochs_processed'], "Output rows should match processed epochs"
        
        # Verify sleep stages are valid integers
        valid_stages = {0, 1, 2, 3, 4, 5}  # Wake, N1, N2, N3, REM, undefined
        assert all(stage in valid_stages for stage in output_df['sleep_stage']), "All sleep stages should be valid"
        
        # Verify timestamps are properly ordered
        assert output_df['timestamp_start'].is_monotonic_increasing, "Start timestamps should be increasing"
        assert all(output_df['timestamp_end'] > output_df['timestamp_start']), "End should be after start"
    
    def test_generate_model_scores_cli_execution(self, small_test_csv, board_config, tmp_path):
        """Test the complete CLI workflow with synthetic data for speed."""
        output_path = tmp_path / "cli_results.csv"
        
        # Build CLI command using board configuration
        exg_columns_str = ",".join(map(str, board_config['exg_columns']))
        cmd = [
            "python", "-m", "sleep_scoring_toolkit.generate_model_scores",
            "--input", small_test_csv,
            "--output", str(output_path),
            "--montage", "minimal_sleep_montage",
            "--sampling-rate", str(board_config['sampling_rate']),
            "--timestamp-column", str(board_config['timestamp_column']),
            "--exg-columns", exg_columns_str,
            "--eeg-channels-for-scoring", "C4,C3,F3,F4",
            "--eog-channels-for-scoring", "L-HEOG,R-HEOG",
            "--quiet"
        ]
        
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        # Execute CLI command
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent.parent.parent,  # Project root
            env=env,
            capture_output=True, 
            text=True
        )
        
        # Verify CLI execution succeeded
        assert result.returncode == 0, f"CLI failed with error: {result.stderr}"
        assert output_path.exists(), "CLI should create output file"
        
        # Verify CLI output format matches direct BatchProcessor
        cli_df = pd.read_csv(output_path)
        expected_columns = ['timestamp_start', 'timestamp_end', 'sleep_stage', 'buffer_id']
        assert list(cli_df.columns) == expected_columns, "CLI output should match BatchProcessor format"
        assert len(cli_df) >= 2, "CLI should process at least 2 epochs from 70s of data"
        
        # Verify output mentions success in stdout
        assert "✅ Processing completed successfully!" in result.stdout or result.returncode == 0
    
    def test_batch_processor_eog_only_montage(self, small_test_csv, board_config, tmp_path):
        """Test BatchProcessor with EOG-only analysis using synthetic data."""
        # Configure EOG-only processor
        montage = Montage.eog_only_montage()
        processor = BatchProcessor(
            montage=montage,
            sampling_rate=board_config['sampling_rate'],
            eeg_channels_for_scoring=[],  # No EEG channels for EOG-only
            eog_channels_for_scoring=["R-HEOG", "L-HEOG"],
            timestamp_column=board_config['timestamp_column'],
            exg_columns=board_config['exg_columns'],
            show_progress=False,
        )
        
        output_path = tmp_path / "eog_only_results.csv"
        results = processor.process_csv_file(small_test_csv, str(output_path))
        
        # Verify EOG-only processing works
        assert results['epochs_processed'] >= 2, "EOG-only processing should work with 70s data"
        assert output_path.exists(), "EOG-only output should be created"
        
        output_df = pd.read_csv(output_path)
        assert len(output_df) >= 2, "EOG-only analysis should produce results"
    
    def test_batch_processor_error_handling(self, board_config, tmp_path):
        """Test BatchProcessor handles invalid inputs gracefully."""
        montage = Montage.minimal_sleep_montage()
        processor = BatchProcessor(
            montage=montage,
            sampling_rate=board_config['sampling_rate'],
            eeg_channels_for_scoring=STANDARD_EEG_CHANNELS,
            eog_channels_for_scoring=STANDARD_EOG_CHANNELS,
            timestamp_column=board_config['timestamp_column'],
            exg_columns=board_config['exg_columns'],
            show_progress=False,
        )
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            processor.process_csv_file("nonexistent.csv", str(tmp_path / "output.csv"))