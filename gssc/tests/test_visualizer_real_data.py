import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from montage import Montage
from cyton_realtime_round_robin_accurate_time_buffer import Visualizer

def test_visualizer_with_real_data():
    """Test the Visualizer with a 30-second chunk of real data"""
    # Path to the test data file
    test_data_path = os.path.join('data', 'realtime_inference_test', 'BrainFlow-RAW_2025-03-29_23-14-54_0.csv')
    
    # Read the data
    df = pd.read_csv(test_data_path, sep='\t', header=None)
    
    # Get sampling rate (125 Hz for Cyton+Daisy)
    sampling_rate = 125
    
    # Calculate number of points for 30 seconds
    points_per_epoch = 30 * sampling_rate
    
    # Take second 30 seconds of data
    epoch_data = df.iloc[points_per_epoch:2*points_per_epoch, 1:].values.T  # Skip timestamp column
    
    # Create montage
    montage = Montage.default_sleep_montage()
    
    # Create visualizer
    visualizer = Visualizer(
        seconds_per_epoch=30,
        board_shim=None,
        montage=montage
    )
    
    # Initialize the plot
    fig, axes = visualizer.init_polysomnograph()
    
    # Plot the data
    visualizer.plot_polysomnograph(
        epoch_data=epoch_data,
        sampling_rate=sampling_rate,
        sleep_stage=0,  # Wake stage
        time_offset=0,
        epoch_start_time=0
    )
    
    # Show the plot
    plt.show()
    
    # Verify the data was plotted correctly
    assert len(visualizer.axes) == len(montage.get_channel_labels()), \
        "Number of axes should match number of channels"
    
    # Check that each axis has data
    for ax in visualizer.axes:
        assert len(ax.lines) > 0, "Each axis should have at least one line"
        
    # Print min/max values for each channel
    for i, (ax, label, ch_type) in enumerate(zip(visualizer.axes, montage.get_channel_labels(), montage.get_channel_types())):
        data = epoch_data[i]
        y_min = np.min(data)
        y_max = np.max(data)
        print(f"\nChannel {label} ({ch_type}):")
        print(f"  Min value: {y_min:.2f} µV")
        print(f"  Max value: {y_max:.2f} µV")
        print(f"  Data range: {y_max - y_min:.2f} µV")

if __name__ == '__main__':
    test_visualizer_with_real_data() 