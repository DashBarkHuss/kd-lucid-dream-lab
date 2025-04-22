import numpy as np
import matplotlib.pyplot as plt
from montage import Montage
from cyton_realtime_round_robin_accurate_time_buffer import Visualizer

def generate_sample_data(num_channels=16, num_points=3750, sampling_rate=125):
    """Generate sample data for testing
    3750 points = 30 seconds at 125Hz sampling rate"""
    time = np.linspace(0, 30, num_points)
    data = []
    
    # Generate different waveforms for each channel type
    for i in range(num_channels):
        if i < 8:  # EEG channels: mixture of frequencies
            signal = (np.sin(2 * np.pi * 10 * time) +  # 10 Hz
                     0.5 * np.sin(2 * np.pi * 5 * time) +  # 5 Hz
                     0.25 * np.random.randn(len(time)))  # noise
        elif i < 12:  # EOG channels: slower waves
            signal = 2 * np.sin(2 * np.pi * 0.3 * time) + 0.1 * np.random.randn(len(time))
        elif i < 14:  # EMG channels: higher frequency
            signal = 0.5 * np.sin(2 * np.pi * 50 * time) + 0.5 * np.random.randn(len(time))
        else:  # Other channels
            signal = np.sin(2 * np.pi * time) + 0.2 * np.random.randn(len(time))
        data.append(signal)
    
    return np.array(data)

def test_visualizer():
    # Create montage
    montage = Montage.default_sleep_montage()
    
    # Create visualizer
    visualizer = Visualizer(seconds_per_epoch=30, montage=montage)
    
    # Generate sample data
    data = generate_sample_data()
    
    # Test visualization
    visualizer.plot_polysomnograph(
        epoch_data=data,
        sampling_rate=125,
        sleep_stage=2,  # N2 sleep stage
        time_offset=0,
        epoch_start_time=0
    )
    
    # Keep the plot window open
    plt.show(block=True)

if __name__ == "__main__":
    test_visualizer() 