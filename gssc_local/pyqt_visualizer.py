import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import sys, os
import logging
from scipy.signal import butter, filtfilt, iirnotch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gssc_local.montage import Montage

# Set up logger
logger = logging.getLogger(__name__)

class FilterError(Exception):
    """Raised when filtering operations fail"""
    pass

class PyQtVisualizer:
    """Handles visualization of polysomnograph data and sleep stages using PyQtGraph"""
    
    # Window and Layout Constants
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 1000
    LAYOUT_MARGINS = 5
    LAYOUT_SPACING = 0
    TITLE_LABEL_HEIGHT = 70 # does this do anything?
    PLOT_WIDGET_MARGINS = 90 # does this do anything?
    PLOT_VERTICAL_SPACING = 100 # does this do anything?
    TIMER_UPDATE_INTERVAL_MS = 50
    
    # Plot Configuration Constants
    PLOT_FIXED_HEIGHT = 44
    PLOT_SPACING = 10
    DATA_PEN_WIDTH = 0.75
    Y_AXIS_MARGIN_FACTOR = 0.15
    FLAT_LINE_DEFAULT_RANGE = 0.2
    FLAT_LINE_MARGIN = 0.1
    Y_AXIS_TICK_COUNT = 5
    LEFT_AXIS_WIDTH = 60  # Fixed width for left axis in pixels
    
    # Font Size Constants
    TITLE_FONT_SIZE = 24
    AXIS_LABEL_FONT_SIZE = 8
    TICK_FONT_SIZE = 7
    
    # Sleep Stage Constants
    SLEEP_STAGE_WAKE = 0
    SLEEP_STAGE_N1 = 1
    SLEEP_STAGE_N2 = 2
    SLEEP_STAGE_N3 = 3
    SLEEP_STAGE_REM = 4
    
    def __init__(self, seconds_per_epoch=30, board_shim=None, montage: Montage = None, headless=False):
        self.recording_start_time = None
        self.seconds_per_epoch = seconds_per_epoch
        self.board_shim = board_shim
        self.countdown_remaining = seconds_per_epoch  # Add countdown counter
        self.is_counting_down = True  # Flag to track if we're still counting down
        self.headless = headless
        
        # Require explicit montage parameter
        if montage is None:
            raise ValueError("montage parameter is required - no default montage will be assumed")
        self.montage = montage
        self.channel_labels = self.montage.get_channel_labels()
        self.channel_types = self.montage.get_channel_types()
        
        # Get channel information from board if available
        if board_shim is not None:
            self.electrode_channels = board_shim.get_exg_channels(board_shim.get_board_id())
        else:
            # Default to 16 channels for Cyton+Daisy
            self.electrode_channels = list(range(16))
            
        # Set PyQtGraph global configuration
        pg.setConfigOption('background', 'w')  # White background
        pg.setConfigOption('foreground', 'k')  # Black foreground
            
        # Initialize PyQtGraph components
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        
        if not self.headless:
            # Create main window
            self.win = QtWidgets.QMainWindow()
            self.win.setWindowTitle('Polysomnograph - Filters: ON')  # Initial title with filter status
            # Increase height to ensure all channels have enough space
            # 16 channels * 100px per channel + some margin for title and spacing
            self.win.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            
            # Create central widget and layout
            self.central_widget = QtWidgets.QWidget()
            self.win.setCentralWidget(self.central_widget)
            self.layout = QtWidgets.QVBoxLayout()
            self.layout.setContentsMargins(self.LAYOUT_MARGINS, self.LAYOUT_MARGINS, self.LAYOUT_MARGINS, self.LAYOUT_MARGINS)  # Reduce margins
            self.layout.setSpacing(self.LAYOUT_SPACING)  # Minimize spacing between widgets
            self.central_widget.setLayout(self.layout)
            
            # Create title label
            title_font = QtGui.QFont()
            title_font.setPointSize(self.TITLE_FONT_SIZE)
            self.title_label = QtWidgets.QLabel("")
            self.title_label.setFont(title_font)
            self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.title_label.setMaximumHeight(self.TITLE_LABEL_HEIGHT)  # Limit title height
            self.layout.addWidget(self.title_label)
            
            # Set initial loading message
            self.title_label.setText(f"Waiting {self.seconds_per_epoch} seconds for initial data...")
            
            # Create filter control checkbox
            self.filter_checkbox = QtWidgets.QCheckBox("Apply Clinical Filters (Northwestern Standard)")
            self.filter_checkbox.setChecked(True)  # Default to ON
            self.filter_checkbox.toggled.connect(self.toggle_visual_filters)
            self.layout.addWidget(self.filter_checkbox)
            
            # Create GraphicsLayoutWidget for plots
            self.plot_widget = pg.GraphicsLayoutWidget()
            self.plot_widget.setContentsMargins(self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS)  # Remove internal margins
            self.plot_widget.ci.layout.setVerticalSpacing(self.PLOT_VERTICAL_SPACING)  # Set a small vertical gap between plots
            self.layout.addWidget(self.plot_widget)
            
            # Show the window
            self.win.show()
        else:
            # In headless mode, create a hidden GraphicsLayoutWidget
            self.plot_widget = pg.GraphicsLayoutWidget()
            self.title_label = None
        
        # Initialize plots and curves
        self.plots = []
        self.curves = []
        
        # Initialize the visualization
        self._init_polysomnograph()
        
        # Set up update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.TIMER_UPDATE_INTERVAL_MS)  # Update every 50ms
        
        # Set up countdown timer
        self.countdown_timer = QtCore.QTimer()
        self.countdown_timer.timeout.connect(self._update_countdown)
        self.countdown_timer.start(1000)  # Update every second
        
        # Visual filtering configuration - initialized based on GUI checkbox
        if not self.headless and hasattr(self, 'filter_checkbox'):
            self.visual_filter_enabled = self.filter_checkbox.isChecked()
        else:
            self.visual_filter_enabled = True  # Default to ON for headless mode
        
        # Cache for immediate filter toggle response
        self._cached_epoch_data = None
        self._cached_sampling_rate = None
        self._cached_sleep_stage = None
        self._cached_time_offset = None
        self._cached_epoch_start_time = None
        
        # Northwestern clinical filter standards
        self.northwestern_bandpass_filters = {
            # EEG Channels (0.3-100 Hz)
            1: (0.3, 100.0),   # F3 - Frontal Left
            2: (0.3, 100.0),   # F4 - Frontal Right  
            3: (0.3, 100.0),   # C3 - Central Left
            4: (0.3, 100.0),   # C4 - Central Right
            5: (0.3, 100.0),   # O1 - Occipital Left
            6: (0.3, 100.0),   # O2 - Occipital Right
            7: (0.3, 100.0),   # T3/T5 - Temporal Left
            8: (0.3, 100.0),   # T4/T6 - Temporal Right
            
            # EOG Channels (0.3-100 Hz)
            9: (0.3, 100.0),    # ROC - Right Outer Canthus
            10: (0.3, 100.0),   # LOC - Left Outer Canthus
            11: (0.3, 100.0),   # R-EOG - Right EOG
            12: (0.3, 100.0),   # L-EOG - Left EOG
            
            # EMG Channels (10-100 Hz)  
            13: (10.0, 100.0),  # EMG1 - Chin EMG
            14: (10.0, 100.0),  # EMG2 - Leg EMG
            
            # Respiratory/Other Channels
            15: (0.3, 50.0),    # Airflow/Respiratory (lower high-cutoff)
            16: (10.0, 100.0)   # Snoring/Audio (higher low-cutoff)
        }
        self.northwestern_notch_filters = [50.0, 60.0]
        
    def _init_polysomnograph(self):
        """Initialize the polysomnograph figure and plots"""
        n_channels = len(self.channel_labels)
        
        # Create plots for each channel
        for i in range(n_channels):
            # Add plot
            p = self.plot_widget.addPlot(row=i, col=0)
            
            # Set up plot margins
            p.setContentsMargins(0, 0, 0, 0)
            p.getViewBox().setDefaultPadding(0)  # Remove padding around the plot
         
            
            # Set up axis labels with proper units
            unit = 'µV'  # All OpenBCI channels output in microvolts
            p.setLabel('left', f'{self.channel_labels[i]}\n({unit})', **{
                'font-size': f'{self.AXIS_LABEL_FONT_SIZE}pt',
                'color': '#000000'
            })
            
            # Show x-axis for all plots but only label the bottom one
            p.showAxis('bottom')
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (seconds)', **{
                    'font-size': f'{self.AXIS_LABEL_FONT_SIZE}pt',
                    'color': '#000000'
                })

            p.showAxis('top')
            p.getAxis('top').setStyle(showValues=False)

            # Set up axis styling
            for axis in ['left', 'bottom']:
                ax = p.getAxis(axis)
                ax.setTextPen('k')
                ax.setPen('k')
                ax.setTickFont(QtGui.QFont('Arial', self.TICK_FONT_SIZE))
                if axis == 'left':
                    # Set fixed width for left axis
                    ax.setWidth(self.LEFT_AXIS_WIDTH)
            
            # Configure grid and axes
            p.showGrid(x=True, y=True)
            p.getAxis('left').setGrid(100)
            p.getAxis('bottom').setGrid(100)
            
            # Add plot to list with blue pen for data
            self.plots.append(p)
            self.curves.append(p.plot(pen=pg.mkPen('b', width=self.DATA_PEN_WIDTH)))
            
            # Set fixed height and ensure ViewBox uses full height
            p.setFixedHeight(self.PLOT_FIXED_HEIGHT)
            view_box = p.getViewBox()
            view_box.setMinimumHeight(self.PLOT_FIXED_HEIGHT)
            view_box.setMaximumHeight(self.PLOT_FIXED_HEIGHT)
            
            # Add small spacing between plots (10 pixels)
            if i < n_channels - 1:
                self.plot_widget.ci.layout.setSpacing(self.PLOT_SPACING)
                self.plot_widget.nextRow()
        
    @staticmethod
    def get_sleep_stage_text(sleep_stage):
        """Convert sleep stage number to text representation"""
        stages = {
            PyQtVisualizer.SLEEP_STAGE_WAKE: 'Wake',
            PyQtVisualizer.SLEEP_STAGE_N1: 'N1',
            PyQtVisualizer.SLEEP_STAGE_N2: 'N2',
            PyQtVisualizer.SLEEP_STAGE_N3: 'N3',
            PyQtVisualizer.SLEEP_STAGE_REM: 'REM'
        }
        return stages.get(sleep_stage, 'Unknown')
    
    def _update_countdown(self):
        """Update the countdown timer and title"""
        if self.is_counting_down and self.countdown_remaining > 0:
            self.countdown_remaining -= 1
            if not self.headless and self.title_label:
                self.title_label.setText(f"Waiting {self.countdown_remaining} seconds for initial data...")
        elif self.countdown_remaining == 0:
            self.is_counting_down = False
            self.countdown_timer.stop()
    
    def toggle_visual_filters(self, checked):
        """Handle checkbox state changes for visual filtering"""
        self.visual_filter_enabled = checked
        filter_status = "ON" if checked else "OFF"
        logger.info(f"Visual filtering toggled: {filter_status}")
        
        # Update window title to show filter status if not in headless mode
        if not self.headless and hasattr(self, 'win'):
            base_title = 'Polysomnograph'
            filter_suffix = f' - Filters: {filter_status}'
            self.win.setWindowTitle(base_title + filter_suffix)
        
        # Immediately replot with current data if available
        if self._cached_epoch_data is not None:
            self.plot_polysomnograph(
                epoch_data_all_electrode_channels_on_board=self._cached_epoch_data,
                sampling_rate=self._cached_sampling_rate,
                sleep_stage=self._cached_sleep_stage,
                time_offset=self._cached_time_offset,
                epoch_start_time=self._cached_epoch_start_time
            )
    
    def apply_bandpass_filter(self, channel_data, low_freq, high_freq, sampling_rate):
        """Apply 4th-order Butterworth bandpass filter with zero-phase filtering"""
        nyquist = sampling_rate / 2.0
        
        # Ensure frequencies are within valid range (0 < freq < nyquist)
        low_freq = max(0.1, min(low_freq, nyquist * 0.95))
        high_freq = max(low_freq + 0.1, min(high_freq, nyquist * 0.95))
        
        low_norm = low_freq / nyquist  
        high_norm = high_freq / nyquist
        
        # Design 4th-order Butterworth filter
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        # Apply zero-phase filtering
        filtered_data = filtfilt(b, a, channel_data)
        return filtered_data

    def apply_notch_filter(self, data, notch_freq, sampling_rate, Q=30):
        """Apply IIR notch filter for power line noise removal"""
        nyquist = sampling_rate / 2.0
        
        # Skip notch filter if frequency is too close to Nyquist
        if notch_freq >= nyquist * 0.9:
            logger.warning(f"Skipping notch filter at {notch_freq} Hz (too close to Nyquist {nyquist} Hz)")
            return data
            
        # Design notch filter
        b, a = iirnotch(notch_freq, Q, sampling_rate)
        
        # Apply zero-phase filtering  
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def apply_complete_filtering(self, epoch_data, channel_numbers, sampling_rate):
        """Apply complete Northwestern filtering pipeline to selected channels
        
        Args:
            epoch_data: Array of shape (n_channels, n_samples) containing data for selected channels
            channel_numbers: List of actual channel numbers (1-based) corresponding to each row in epoch_data
            sampling_rate: Sampling rate in Hz
        """
        filtered_data = epoch_data.copy()
        
        for channel_idx, channel_num in enumerate(channel_numbers):
            
            if channel_num in self.northwestern_bandpass_filters:
                try:
                    # Step 1: Apply channel-specific bandpass filter
                    low_freq, high_freq = self.northwestern_bandpass_filters[channel_num]
                    filtered_data[channel_idx] = self.apply_bandpass_filter(
                        filtered_data[channel_idx], low_freq, high_freq, sampling_rate
                    )
                    
                    # Step 2: Apply notch filters for power line noise
                    for notch_freq in self.northwestern_notch_filters:
                        filtered_data[channel_idx] = self.apply_notch_filter(
                            filtered_data[channel_idx], notch_freq, sampling_rate
                        )
                        
                except Exception as e:
                    logger.error(f"Filter failed on channel {channel_num} ({low_freq}-{high_freq} Hz): {e}")
                    raise FilterError(f"Channel {channel_num} filtering failed. Check sampling rate and data quality.")
        
        return filtered_data
            
    def plot_polysomnograph(self, epoch_data_all_electrode_channels_on_board, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
        """Update polysomnograph plot with new data"""
        # Cache the current display parameters for immediate filter toggling
        self._cached_epoch_data = epoch_data_all_electrode_channels_on_board.copy()
        self._cached_sampling_rate = sampling_rate
        self._cached_sleep_stage = sleep_stage
        self._cached_time_offset = time_offset
        self._cached_epoch_start_time = epoch_start_time
        
        # Stop countdown if it's still running
        if self.is_counting_down:
            self.is_counting_down = False
            self.countdown_timer.stop()
            
        # Calculate elapsed time
        elapsed_seconds = (epoch_start_time - self.recording_start_time 
                         if self.recording_start_time is not None and epoch_start_time is not None 
                         else time_offset)
        
        # Format time string
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        relative_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Update title with proper styling
        title_text = f'Sleep Stage: {self.get_sleep_stage_text(sleep_stage)} | Time from Start: {relative_time_str}'
        if not self.headless and self.title_label:
            self.title_label.setText(title_text)
        
        # Create time axis
        time_axis = np.arange(epoch_data_all_electrode_channels_on_board.shape[1]) / sampling_rate + time_offset

        # Extract only the channels defined in the montage from the full epoch_data
        electrode_indices = self.montage.get_electrode_channel_indices()
        montage_electrode_data = epoch_data_all_electrode_channels_on_board[electrode_indices]

        # Apply Northwestern filtering if enabled (for display only)
        if hasattr(self, 'visual_filter_enabled') and self.visual_filter_enabled:
            # Get actual channel numbers for proper filter mapping
            channel_numbers = sorted(self.montage.channels.keys())
            filtered_display_montage_electrode_data = self.apply_complete_filtering(montage_electrode_data, channel_numbers, sampling_rate)
        else:
            filtered_display_montage_electrode_data = montage_electrode_data

        # Update each channel's plot
        for data, plot, curve in zip(filtered_display_montage_electrode_data, self.plots, self.curves):
            # Update curve data
            curve.setData(time_axis, data)
            
            # Calculate y-axis range with extra margin for border lines
            y_min = np.min(data)
            y_max = np.max(data)
            y_range = y_max - y_min
            if y_range == 0:  # Handle flat line case
                y_range = self.FLAT_LINE_DEFAULT_RANGE  # Small range to show flat line
                y_min -= self.FLAT_LINE_MARGIN
                y_max += self.FLAT_LINE_MARGIN
            
            # Add larger margin for border lines
            margin = y_range * self.Y_AXIS_MARGIN_FACTOR  # Increased from 0.1 to 0.15
            y_min_with_margin = y_min - margin
            y_max_with_margin = y_max + margin
            
            # Set axis ranges with proper tick marks
            plot.setYRange(y_min_with_margin, y_max_with_margin, padding=0)  # Added padding=0
            plot.setXRange(time_offset, time_offset + self.seconds_per_epoch, padding=0)  # Added padding=0
            
            # Update axis ticks
            y_ticks = np.linspace(y_min_with_margin, y_max_with_margin, self.Y_AXIS_TICK_COUNT)
            plot.getAxis('left').setTicks([[(v, f'{v:.1f}') for v in y_ticks]])
            
            # Force the y-axis to show exactly 5 ticks
            plot.getAxis('left').setStyle(tickTextOffset=5)  # Adjust text offset for better visibility
            plot.getAxis('left').setTickSpacing(major=(y_max_with_margin - y_min_with_margin) / (self.Y_AXIS_TICK_COUNT - 1))
            
            # Show "All values are ---" text if needed
            if np.allclose(data, data[0]):
                # Create text item for "All values are ---" message
                text = pg.TextItem(f"All values are {data[0]:.1f}", color='r', anchor=(0, 0))  # Changed anchor to (0, 0) for left alignment
                plot.addItem(text)
                # Position the text at the left side of the plot
                text.setPos(time_offset, y_max_with_margin - margin/2)  # Changed x position to time_offset
            else:
                # Remove any existing text items
                for item in plot.items:
                    if isinstance(item, pg.TextItem):
                        plot.removeItem(item)

            

    
    def update(self):
        """Update the visualization (called by timer)"""
        if not self.headless:
            self.app.processEvents()
        
    def close(self):
        """Close the visualization window and clean up Qt resources"""
        try:
            # Stop timers first
            if hasattr(self, 'timer'):
                self.timer.stop()
            if hasattr(self, 'countdown_timer'):
                self.countdown_timer.stop()
            
            # Clean up plots and curves first
            if hasattr(self, 'curves'):
                self.curves.clear()
            if hasattr(self, 'plots'):
                for plot in self.plots:
                    if plot is not None:
                        plot.clear()  # Clear plot contents before closing
                self.plots.clear()
            
            # Close window if not in headless mode
            if not self.headless and hasattr(self, 'win'):
                self.win.close()
            
            # Clean up plot widget last
            if hasattr(self, 'plot_widget'):
                self.plot_widget.clear()  # Clear widget contents before closing
                self.plot_widget.close()
            
            # Process any pending events
            if hasattr(self, 'app'):
                self.app.processEvents()
                
            # Quit the application if we're in CI
            if os.environ.get('CI') == 'true' and hasattr(self, 'app'):
                self.app.quit()
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Try to process events one last time
            if hasattr(self, 'app'):
                self.app.processEvents() 