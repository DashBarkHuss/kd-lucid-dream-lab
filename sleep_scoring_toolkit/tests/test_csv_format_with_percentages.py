"""
Test for CSV output format with sleep stage percentages.
"""

import pytest
import pandas as pd
import tempfile
import os

from brainflow.board_shim import BoardShim, BoardIds
from sleep_scoring_toolkit.batch_processor import BatchProcessor
from sleep_scoring_toolkit.montage import Montage
from sleep_scoring_toolkit.constants import GSSCStages


def test_csv_format_with_percentages():
    """Test that CSV output includes percentage columns when enabled."""
    test_data_path = os.path.abspath("small_test_8_16_data.csv")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Required test data file {test_data_path} not found. Please ensure test data is available.")
    
    # Setup processor with BoardShim API configuration
    board_id = BoardIds.CYTON_DAISY_BOARD
    montage = Montage.minimal_sleep_montage()
    
    # Get board configuration from BoardShim API
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    exg_channels = BoardShim.get_exg_channels(board_id)
    
    processor = BatchProcessor(
        montage=montage,
        sampling_rate=sampling_rate,
        eeg_channels_for_scoring=["C4", "C3", "F3", "F4"],
        eog_channels_for_scoring=["L-HEOG", "R-HEOG"],
        timestamp_column=timestamp_channel,
        exg_columns=exg_channels,
        show_progress=False
    )
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        output_path = temp_file.name
    
    try:
        # Process the data
        processor.process_csv_file(test_data_path, output_path)
        
        # Load and validate CSV
        df = pd.read_csv(output_path)
        
        # Basic validation
        assert len(df) > 0
        assert 'sleep_stage' in df.columns
        assert df['sleep_stage'].min() >= GSSCStages.MIN_STAGE
        assert df['sleep_stage'].max() <= GSSCStages.MAX_STAGE
        
        # Validate percentage columns are present
        expected_cols = ['wake_percent', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent']
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column: {col}"
        
        # Validate percentage values
        percentage_cols = ['wake_percent', 'n1_percent', 'n2_percent', 'n3_percent', 'rem_percent']
        for idx, row in df.iterrows():
            # Check percentages sum to approximately 100%
            total_percent = sum(row[col] for col in percentage_cols)
            assert 99.0 <= total_percent <= 101.0, f"Percentages should sum to ~100%, got {total_percent}"
            
            # Check all percentages are non-negative
            for col in percentage_cols:
                assert row[col] >= 0, f"Percentage {col} should be non-negative, got {row[col]}"
                assert row[col] <= 100, f"Percentage {col} should be <= 100%, got {row[col]}"
            
            # Verify final sleep stage matches highest percentage
            max_percent_stage = max(enumerate(row[col] for col in percentage_cols), key=lambda x: x[1])[0]
            assert row['sleep_stage'] == max_percent_stage, f"Final stage {row['sleep_stage']} should match highest percentage stage {max_percent_stage}"
        
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)