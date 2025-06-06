import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

import numpy as np
import pandas as pd
from montage import Montage
from pyqt_visualizer import PyQtVisualizer

def test_pyqt_visualizer_with_real_data():
    """Test the PyQtVisualizer with a 30-second chunk of real data from a BrainFlow recording"""
    # Check if running in CI environment
    is_ci = os.environ.get('CI') == 'true'
    
    # Path to the test data file
    test_data_path = os.path.join(workspace_root, 'gssc_local', 'tests', 'test_data', 'BrainFlow-RAW_test.csv')
    
    # Read the data
    df = pd.read_csv(test_data_path, sep='\t', header=None)
    
    # Get sampling rate (125 Hz for Cyton+Daisy)
    sampling_rate = 125
    
    # Calculate number of points for 30 seconds
    points_per_epoch = 30 * sampling_rate
    
    # Create montage - using minimal montage without temporal and top/bottom EOG
    montage = Montage.minimal_sleep_montage()
    
    # Get the montage channel indices in order
    montage_channels = sorted(montage.channels.keys())
    
    # Map montage channels to data file columns
    selected_columns = [ch for ch in montage_channels]
    
    # Select the data for the epoch
    start_idx = points_per_epoch  # Start at 3750 (30 seconds in)
    end_idx = 2 * points_per_epoch  # End at 7500 (60 seconds in)
    
    # Get the epoch data
    epoch_data = df.iloc[start_idx:end_idx, selected_columns].values.T
    
    # Create visualizer with headless mode in CI
    visualizer = PyQtVisualizer(
        seconds_per_epoch=30,
        board_shim=None,
        montage=montage,
        headless=is_ci
    )
    
    # Plot the data
    visualizer.plot_polysomnograph(
        epoch_data=epoch_data,
        sampling_rate=sampling_rate,
        sleep_stage=0,  # Wake stage
        time_offset=0,
        epoch_start_time=0
    )
    
    # Verify the visualization was created correctly
    assert len(visualizer.plots) == len(montage.get_channel_labels()), \
        "Number of plots should match number of channels"
    
    # Check that each plot has data
    for curve in visualizer.curves:
        assert curve.xData is not None and curve.yData is not None, \
            "Each curve should have data points"
        assert len(curve.xData) == points_per_epoch, \
            "Each curve should have the correct number of data points"
    
    # Only show window if not in CI
    if not is_ci:
        visualizer.app.exec()
    else:
        # In CI, just close immediately
        visualizer.close()

if __name__ == '__main__':
    test_pyqt_visualizer_with_real_data() 