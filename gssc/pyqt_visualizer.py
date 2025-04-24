import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from montage import Montage

class PyQtVisualizer:
    """Handles visualization of polysomnograph data and sleep stages using PyQtGraph"""
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
        self.win.resize(1500,2000)
        
        # Create central widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins
        self.layout.setSpacing(0)  # Minimize spacing between widgets
        self.central_widget.setLayout(self.layout)
        
        # Create title label
        title_font = QtGui.QFont()
        title_font.setPointSize(16)
        self.title_label = QtWidgets.QLabel("")
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setMaximumHeight(30)  # Limit title height
        self.layout.addWidget(self.title_label)
        
        # Create GraphicsLayoutWidget for plots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
        self.plot_widget.ci.layout.setVerticalSpacing(-30)  # Set a small vertical gap between plots
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
        self.timer.start(50)  # Update every 50ms
        
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
            
            # p.hideAxis('left')
            # p.showAxis('top')
            # p.hideAxis('right')
            # p.hideAxis('bottom')
            # # Set up plot styling
            # p.showGrid(x=True, y=True, alpha=0.3)
            # p.getAxis('left').setWidth(60)
            
            # # Show axis lines with proper styling
            # p.getAxis('left').setStyle(showValues=True, tickLength=5)
            # p.getAxis('bottom').setStyle(showValues=True, tickLength=5)
            # p.showAxis('top', show=False)
            # p.showAxis('right', show=False)
            
            # Set up axis labels with proper units
            unit = 'µV' if self.channel_types[i] in ['EEG', 'EOG', 'EMG'] else 'a.u.'
            p.setLabel('left', f'{self.channel_labels[i]}\n({unit})', **{
                'font-size': '8pt',
                'color': '#000000'
            })
            
            # Show x-axis for all plots but only label the bottom one
            p.showAxis('bottom')
            if i == n_channels - 1:
                p.setLabel('bottom', 'Time (seconds)', **{
                    'font-size': '8pt',
                    'color': '#000000'
                })

            p.showAxis('top')
            p.getAxis('top').setStyle(showValues=False)

            # # Set up axis styling
            for axis in ['left', 'bottom']:
                ax = p.getAxis(axis)
                ax.setTextPen('k')
                ax.setPen('k')
                ax.setTickFont(QtGui.QFont('Arial', 7))
            
            # Configure grid and axes
            p.showGrid(x=True, y=True)
            p.getAxis('left').setGrid(100)
            p.getAxis('bottom').setGrid(100)
            
            # Add plot to list with blue pen for data
            self.plots.append(p)
            self.curves.append(p.plot(pen=pg.mkPen('b', width=0.75)))
            
            # Set fixed height and ensure ViewBox uses full height
            p.setFixedHeight(70)
            view_box = p.getViewBox()
            view_box.setMinimumHeight(70)
            view_box.setMaximumHeight(70)
            
            # Add small spacing between plots (10 pixels)
            if i < n_channels - 1:
                self.plot_widget.ci.layout.setSpacing(10)
                self.plot_widget.nextRow()
        
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
                y_range = 0.2  # Small range to show flat line
                y_min -= 0.1
                y_max += 0.1
            
            # Add larger margin for border lines
            margin = y_range * 0.15  # Increased from 0.1 to 0.15
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
            y_ticks = np.linspace(y_min_with_margin, y_max_with_margin, 5)
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