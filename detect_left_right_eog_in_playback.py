# This is not perfec. We neeed to tweak thr scrip to better detect the left and right eye movements

import logging
from scipy.signal import find_peaks, detrend

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import brainflow
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import argparse
import time
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from brainflow.data_filter import DataFilter, DetrendOperations
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import find_peaks

class Marker:
    def __init__(self, x, color, name):
        self.x = x
        self.color = color
        self.name = name

class MarkerLegendItem(pg.GraphicsObject):
    def __init__(self, color):
        pg.GraphicsObject.__init__(self)
        self.color = color
        
    def paint(self, p, *args):
        p.setPen(pg.mkPen(self.color))
        p.drawLine(0, 10, 20, 10)
        
    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

def detect_left_eye_movements(csv_file, channel_index=0, threshold_uv=400, peak_width=50, return_width=450, start_of_return_width=50):
    # Read the CSV file
    df = pd.read_csv(csv_file, sep='\t', header=None)
    
    print("Columns in the CSV file:")
    print(df.columns)
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Extract the relevant channel data (assuming channel_index is 0-based)
    signal = df.iloc[:, channel_index + 1].values  # +1 because first column is usually timestamp
    # timestamps = df.iloc[:, 13].values
    x_axises = []
    
    # Find peaks above the threshold
    peaks, _ = find_peaks(signal, height=threshold_uv, width=peak_width, distance=100, prominence=50) # prominence is the minimum height of a peak and height is
    
    left_eye_movements = []
    
    for peak in peaks:
        # Check if the signal returns to normal within the specified width
        end_index = min(len(signal), peak + return_width)
        start_of_return_index = min(len(signal), peak + start_of_return_width)
        if np.all(signal[start_of_return_index:end_index] < threshold_uv):
            # Get the timestamp of the start of the eye movement
            # start_index = max(0, peak - peak_width)
            # timestamp = timestamps[peak]
            left_eye_movements.append((peak))
    
    return signal, left_eye_movements

# similar to detect_left_eye_movements but for right eye movements so we detect negative peaks
def detect_right_eye_movements(csv_file, channel_index=0, threshold_uv=-400, peak_width=50, return_width=450, start_of_return_width=50):
    # Read the CSV file
    df = pd.read_csv(csv_file, sep='\t', header=None)
    
    print("Columns in the CSV file:")
    print(df.columns)
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Extract the relevant channel data (assuming channel_index is 0-based)
    signal = df.iloc[:, channel_index + 1].values  # +1 because first column is usually timestamp
    # timestamps = df.iloc[:, 13].values
    x_axises = []
    
    # Find peaks below the threshold
    peaks, _ = find_peaks(-signal, height=threshold_uv, width=peak_width, distance=100, prominence=50)
    
    right_eye_movements = []
    
    for peak in peaks:
        # Check if the signal returns to normal within the specified width
        end_index = min(len(signal), peak + return_width)
        start_of_return_index = min(len(signal), peak + start_of_return_width)
        if np.all(signal[start_of_return_index:end_index] > threshold_uv):
            # Get the timestamp of the start of the eye movement
            # start_index = max(0, peak - peak_width)   
            right_eye_movements.append((peak))
    
    return signal, right_eye_movements


class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        
        self.data_buffers = {channel: [] for channel in self.exg_channels}
        self.max_points = 3000  # Maximum number of points to keep


        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='BrainFlow Plot')
        self.win.resize(800, 600)
        self.win.setWindowTitle('BrainFlow Plot')

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points) #num_points is the number of data points to get
        if data.size == 0:
            print("No data received from the board")
            return

        self.remove_all_infinite_lines() #if we don't do this, the markers will stack on top of each other
        for count, channel in enumerate(self.`exg_channels): # this line means for each channel in the exg_channels list, enumerate returns a tuple containing the index and the value
            channel_data = data[channel]
            
            # Add new data to our buffer
            self.data_buffers[channel].extend(channel_data.tolist())
            # Keep only the most recent max_points
            self.data_buffers[channel] = self.data_buffers[channel][-self.max_points:]
            
            # Update the plot with all data in our buffer
            self.curves[count].setData(self.data_buffers[channel])

            if channel_data.size == 0:
                print(f"No data for channel {channel}")
                continue

            try:
                # detect eye movements
                left_movements = self.detect_eye_movements(data[channel], threshold=400, direction='left')
                right_movements = self.detect_eye_movements(data[channel], threshold=-400, direction='right')

                infinite_lines = [item for item in self.plots[count].items if isinstance(item, pg.InfiniteLine)]
                # print(f"Channel {channel}: {len(infinite_lines)} markers. Left: {len(le)}, Right: {len(re)}")
            

                # add markers for eye movements
                for movement in left_movements:
                    self.plots[count].addItem(pg.InfiniteLine(pos=movement, angle=90, pen='g'))
                for movement in right_movements:
                    self.plots[count].addItem(pg.InfiniteLine(pos=movement, angle=90, pen='r'))

                # Apply filters
                # DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                # DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                #                             FilterTypes.BUTTERWORTH.value, 0)
                # DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                #                             FilterTypes.BUTTERWORTH.value, 0)
                # DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                #                             FilterTypes.BUTTERWORTH.value, 0)
                self.curves[count].setData(channel_data.tolist()) 
            except brainflow.BrainFlowError as e:
                print(f"Error processing channel {channel}: {str(e)}")

        self.app.processEvents()

    def detect_eye_movements(self, data, threshold=400, peak_width=10, peak_distance=100, direction='left'):
        if direction == 'left':
            peaks, _ = find_peaks(data, height=400, width=10, distance=100)
        else:  # right
            peaks, _ = find_peaks(-data,  height=400, width=10, distance=100)
        return peaks
    
    def remove_all_infinite_lines(self):
        for count, channel in enumerate(self.exg_channels):
            items_to_remove = [item for item in self.plots[count].items if isinstance(item, pg.InfiniteLine)]
            for item in items_to_remove:
                self.plots[count].removeItem(item)
            print(f"Removed {len(items_to_remove)} InfiniteLines from Channel {channel}")

def plot_data_with_eye_movements(file_path):
    # Load data from CSV file
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    
    # Assuming the EOG data is in the second column (index 1)
    eog_data = data[:, 1]
    
    # Detrend the data using scipy instead of BrainFlow
    eog_data = detrend(eog_data)
    # Detect eye movements
    left_movements, _ = find_peaks(eog_data, height=400, width=10, distance=100)
    right_movements, _ = find_peaks(-eog_data, height=400, width=10, distance=100)
    
    # Create plot
    app = pg.mkQApp("EOG Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="EOG Plot with Eye Movements")
    win.resize(1000, 600)
    plot = win.addPlot(title="EOG Signal")
    
    # Plot EOG data
    plot.plot(eog_data, pen='b')
    
    # Add markers for eye movements
    for left in left_movements:
        plot.addItem(pg.InfiniteLine(pos=left, angle=90, pen='g', label='Left'))
    for right in right_movements:
        plot.addItem(pg.InfiniteLine(pos=right, angle=90, pen='r', label='Right'))
    
    # Add legend
    legend = plot.addLegend()
    legend.addItem(pg.PlotDataItem(pen='g'), 'Left Eye Movement')
    legend.addItem(pg.PlotDataItem(pen='r'), 'Right Eye Movement')
    
    # Start Qt event loop
    pg.exec()

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)


    params = BrainFlowInputParams()
    # play from file
    # params board_id PLAYBACK_FILE_BOARD
    params.board_id = BoardIds.PLAYBACK_FILE_BOARD
    params.master_board = BoardIds.GANGLION_BOARD
    # I needed to use the full path or the path from the root because I run the script from the root directory using .vscode/launch.json   
    params.file = "data/BrainFlow-RAW.csv"
    # params.file = "/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/BrainFlow-RAW.csv"


    board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim)
        plot_data_with_eye_movements(params.file)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()