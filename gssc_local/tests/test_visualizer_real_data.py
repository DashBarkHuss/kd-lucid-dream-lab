import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add workspace root to path
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workspace_root)

import numpy as np
import pandas as pd
# Set matplotlib backend to 'Agg' in CI to prevent GUI
if os.environ.get('CI'):
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from montage import Montage
from gssc_local.realtime_with_restart.visualizer import Visualizer
from realtime_with_restart.channel_mapping import ChannelIndexMapping, DataWithBrainFlowDataKey

def test_visualizer_with_real_data():
    """Test the Visualizer with a 30-second chunk of real data"""
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
    
    # Create channel mapping for the data
    num_channels = epoch_data.shape[0]
    channel_mapping = [
        ChannelIndexMapping(board_position=i+1)  # Board positions 1 to N
        for i in range(num_channels)
    ]
    
    # Wrap data with channel mapping
    epoch_data_wrapper = DataWithBrainFlowDataKey(
        data=epoch_data,
        channel_mapping=channel_mapping
    )
    
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
        epoch_data_wrapper=epoch_data_wrapper,
        sampling_rate=sampling_rate,
        sleep_stage=0,  # Wake stage
        time_offset=0,
        epoch_start_time=0
    )
    
    # Only show the plot in non-CI environments
    if not os.environ.get('CI'):
        plt.show(block=True)
    else:
        # In CI, just verify the plot was created correctly
        plt.close('all')  # Clean up
    
    # Verify the data was plotted correctly
    assert len(visualizer.axes) == len(montage.get_channel_labels()), \
        "Number of axes should match number of channels"
    
    # Check that each axis has data
    for ax in visualizer.axes:
        assert len(ax.lines) > 0, "Each axis should have at least one line"
        
    # Clean up
    plt.close('all')

if __name__ == '__main__':
    test_visualizer_with_real_data() 