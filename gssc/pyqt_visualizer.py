import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from montage import Montage

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
    
    def __init__(self, seconds_per_epoch=30, board_shim=None, montage: Montage = None):
        self.recording_start_time = None
        self.seconds_per_epoch = seconds_per_epoch
        self.board_shim = board_shim
        
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
            
        # Set PyQtGraph global configuration
        pg.setConfigOption('background', 'w')  # White background
        pg.setConfigOption('foreground', 'k')  # Black foreground
            
        # Initialize PyQtGraph components
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        
        # Create main window
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('Polysomnograph')
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
        
        # Create GraphicsLayoutWidget for plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setContentsMargins(self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS, self.PLOT_WIDGET_MARGINS)  # Remove internal margins
        self.plot_widget.ci.layout.setVerticalSpacing(self.PLOT_VERTICAL_SPACING)  # Set a small vertical gap between plots
        self.layout.addWidget(self.plot_widget)
        
        # Initialize plots and curves
        self.plots = []
        self.curves = []
        
        # Initialize the visualization
        self._init_polysomnograph()
        
        # Show the window
        self.win.show()
        
        # Set up update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.TIMER_UPDATE_INTERVAL_MS)  # Update every 50ms
        
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

            # # Set up axis styling
            for axis in ['left', 'bottom']:
                ax = p.getAxis(axis)
                ax.setTextPen('k')
                ax.setPen('k')
                ax.setTickFont(QtGui.QFont('Arial', self.TICK_FONT_SIZE))
            
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
    
    def plot_polysomnograph(self, epoch_data, sampling_rate, sleep_stage, time_offset=0, epoch_start_time=None):
        """Update polysomnograph plot with new data"""
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
        self.title_label.setText(title_text)
        
        # Create time axis
        time_axis = np.arange(epoch_data.shape[1]) / sampling_rate + time_offset

        # Log min/max values and height for each channel
        print("\nChannel ranges and heights:")
        for i, (data, label) in enumerate(zip(epoch_data, self.channel_labels)):
            y_min = np.min(data)
            y_max = np.max(data)
            y_range = y_max - y_min
            plot_height = self.plots[i].height()
            print(f"{label}: min={y_min:.2f}, max={y_max:.2f}, range={y_range:.2f}, plot_height={plot_height}px")
        
        # Update each channel's plot
        for i, (data, plot, curve) in enumerate(zip(epoch_data, self.plots, self.curves)):
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
            
            # # Add black lines at top and bottom of plot
            # if not hasattr(plot, 'border_lines'):
            #     plot.border_lines = [
            #         pg.InfiniteLine(pos=y_min_with_margin, angle=0, pen=pg.mkPen('r', width=1)),
            #         pg.InfiniteLine(pos=y_max_with_margin, angle=0, pen=pg.mkPen('r', width=1))
            #     ]
            #     for line in plot.border_lines:
            #         plot.addItem(line)
            # else:
            #     plot.border_lines[0].setPos(y_min_with_margin)
            #     plot.border_lines[1].setPos(y_max_with_margin)
            
            # Get scene coordinates
            view_box = plot.getViewBox()
            scene_rect = view_box.sceneBoundingRect()
            
            # Log positions in scene coordinates
            print(f"\nChannel {self.channel_labels[i]} positions:")
            print(f"Scene Y start: {scene_rect.top():.1f}")
            print(f"Scene Y end: {scene_rect.bottom():.1f}")
            print(f"Scene height: {scene_rect.height():.1f}")
            
            # Update axis ticks
            y_ticks = np.linspace(y_min_with_margin, y_max_with_margin, self.Y_AXIS_TICK_COUNT)
            plot.getAxis('left').setTicks([[(v, f'{v:.1f}') for v in y_ticks]])
            
            # Show "All values are 0.0" text if needed
            if np.allclose(data, 0):
                plot.setTitle("All values are 0.0", color='r', size='8pt')
            else:
                plot.setTitle("")
    
    def update(self):
        """Update the visualization (called by timer)"""
        self.app.processEvents()
        
    def close(self):
        """Close the visualization window"""
        self.win.close()
        self.timer.stop() 