import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from montage import Montage
from brainflow.board_shim import BoardShim

from .queue import VisualizationQueue

class Visualizer:
    """Handles visualization of polysomnograph data and sleep stages"""
    def __init__(self, seconds_per_epoch=30, board_shim=None, montage: Montage = None):
        self.fig = None
        self.axes = None
        self.recording_start_time = None
        self.seconds_per_epoch = seconds_per_epoch
        self.board_shim = board_shim
        self.viz_queue = VisualizationQueue()
        
        # Use provided montage or create default
        self.montage = montage if montage is not None else Montage.default_sleep_montage()
        self.channel_labels = self.montage.get_channel_labels()
        self.channel_types = self.montage.get_channel_types()
        self.filter_ranges = self.montage.get_filter_ranges()
        
        # Get channel information from board if available
        if board_shim is not None:
            self.electrode_channels = board_shim.get_exg_channels(board_shim.get_board_id())
        else:
            # Default to 16 channels for Cyton+Daisy
            self.electrode_channels = list(range(16))
            
        # Initialize figure in the main thread
        self.init_polysomnograph()
        # Start processing visualization updates
        self.viz_queue.start_processing(self)
        
    def init_polysomnograph(self):
        """Initialize the polysomnograph figure and axes"""
        if self.fig is None:
            plt.ion()  # Turn on interactive mode
            n_channels = len(self.channel_labels)
            
            # Create figure with balanced size
            self.fig = plt.figure(figsize=(12, 10))  # Reduced height from 16 to 10
            
            # Create a gridspec that leaves room for the title and adds spacing between channels
            gs = self.fig.add_gridspec(n_channels + 1, 1, height_ratios=[0.5] + [1.2]*n_channels)  # Adjusted channel height ratio from 1.5 to 1.2
            gs.update(left=0.1, right=0.95, bottom=0.05, top=0.95, hspace=0.5)
            
            # Create title axes
            self.title_ax = self.fig.add_subplot(gs[0])
            self.title_ax.set_xticks([])
            self.title_ax.set_yticks([])
            self.title_ax.spines['top'].set_visible(False)
            self.title_ax.spines['right'].set_visible(False)
            self.title_ax.spines['bottom'].set_visible(False)
            self.title_ax.spines['left'].set_visible(False)
            
            # Create axes for channels
            self.axes = []
            for i in range(n_channels):
                ax = self.fig.add_subplot(gs[i+1])
                self.axes.append(ax)
                
                # Setup axis
                ax.set_ylabel(self.channel_labels[i], fontsize=8, rotation=0, ha='right', va='center')
                ax.grid(True, alpha=0.3)  # Lighter grid
                ax.tick_params(axis='y', labelsize=8)
                
                # Hide unnecessary spines and set colors for visible ones
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                
                # Only show x-axis for bottom subplot
                if i < n_channels - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Time (seconds)', fontsize=8)
                    ax.tick_params(axis='x', labelsize=8)
        
        return self.fig, self.axes
    
    @staticmethod
    def get_sleep_stage_text(sleep_stage):
        """Convert sleep stage number to text representation"""
        stages = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM'
        }
        return stages.get(sleep_stage, 'Unknown')
    
    def plot_polysomnograph(self, epoch_data, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
        """Add visualization update to queue instead of plotting directly"""
        self.viz_queue.add_update(epoch_data, sampling_rate, sleep_stage, time_offset, epoch_start_time)
        
    def _plot_polysomnograph(self, epoch_data, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
        """Internal method to actually perform the plotting (called from main thread)"""
        # Create time axis with offset
        time_axis = np.arange(epoch_data.shape[1]) / sampling_rate + time_offset
        
        # Calculate elapsed time
        elapsed_seconds = (epoch_start_time - self.recording_start_time 
                         if self.recording_start_time is not None and epoch_start_time is not None 
                         else time_offset)
        
        # Format time string
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        relative_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Update title
        title_text = f'Sleep Stage: {self.get_sleep_stage_text(sleep_stage)} | Time from Start: {relative_time_str}'
        self.title_ax.clear()
        self.title_ax.text(0.5, 0.5, title_text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          fontsize=10)
        self.title_ax.set_xticks([])
        self.title_ax.set_yticks([])
        
        # Plot each channel
        for ax, data, label, ch_type in zip(self.axes, epoch_data, self.channel_labels, self.channel_types):
            ax.clear()  # Clear previous data
            
            # Add units based on channel type
            if ch_type in ['EEG', 'EOG', 'EMG']:
                unit = 'µV'
            else:
                unit = 'a.u.'  # arbitrary units
            
            # Plot the data
            ax.plot(time_axis, data, 'b-', linewidth=0.5)
            
            # Set y-axis label with units
            ax.set_ylabel(f'{label}\n({unit})', fontsize=7, rotation=0, ha='right', va='center')
            ax.grid(True, alpha=0.3)  # Lighter grid
            ax.tick_params(axis='y', labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Calculate y-axis limits based on the actual data range
            y_min = np.min(data)
            y_max = np.max(data)
            y_range = y_max - y_min
            margin = y_range * 0.1  # 10% margin
            y_limits = (y_min - margin, y_max + margin)
            ax.set_ylim(y_limits)
            
            # Create clean tick marks that include the range
            tick_range = y_max - y_min
            if tick_range > 0:
                # Choose a reasonable number of ticks (3-5)
                n_ticks = 5 if tick_range > 3 else 3
                ax.yaxis.set_major_locator(plt.LinearLocator(n_ticks))
            
            if y_min == y_max:
                # For zero-value channels, set y limits explicitly to match other channels
                ax.set_ylim(-0.1, 0.1)
                # Add red text indicating all zeros
                ax.text(0.02, 0.7, f"All values are {y_min}", 
                       transform=ax.transAxes,
                       color='red',
                       fontsize=8,
                       fontweight='bold')
                # Draw the zero line in light grey
                ax.axhline(y=0, color='grey', linewidth=0.5, alpha=0.3)
            
            # Add horizontal lines at the top and bottom of each channel's plot area
            ax.axhline(y=ax.get_ylim()[1], color='black', linewidth=1)  # Black line at top
            ax.axhline(y=ax.get_ylim()[0], color='black', linewidth=1)  # Black line at bottom
            
            # Format y-axis ticks to show one decimal place
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
            
            # Only show x-axis for bottom subplot
            if ax != self.axes[-1]:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time (seconds)', fontsize=8)
                ax.tick_params(axis='x', labelsize=8)
            
            # Set x-axis limits
            ax.set_xlim(time_offset, time_offset + self.seconds_per_epoch)
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def release(self):
        """Clean up visualization resources"""
        self.viz_queue.stop_processing()
        if self.fig is not None:
            plt.close(self.fig) 