"""
Real-time EEG visualization script using BrainFlow and PyQtGraph that applies filters to the data. Connects to a Ganglion board 
via streaming protocol and displays both filtered and unfiltered EEG data in separate windows.
The Graph class handles real-time plotting with configurable filter functions. Filtering options 
include notch filters (50-60Hz) and channel-specific bandpass filters (0.3-35Hz for channels 1,2,4 
and 10-100Hz for channel 3). The script creates two parallel visualizations to compare raw and 
processed signals, with automatic updates every 50ms in a 4-second sliding window.
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/Users/dashiellbarkhuss/anaconda3/plugins/platforms'
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore


class Graph:
    def __init__(self, board_shim, apply_filters_func, title):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.apply_filters_func = apply_filters_func

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(800, 600)
        self.win.setWindowTitle(title)

        self._init_timeseries()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

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
        data = self.board_shim.get_current_board_data(self.num_points)
        if data.size == 0:
            print("No data received from the board")
            return

        for count, channel in enumerate(self.exg_channels):
            channel_data = data[channel]
            if channel_data.size == 0:
                print(f"No data for channel {channel}")
                continue

            try:
                # Apply custom filters
                filtered_data = self.apply_filters_func(channel_data, self.sampling_rate, count)
                self.curves[count].setData(filtered_data.tolist())
                # log the original data and the filtered data
                # print(f"Original data: {channel_data[:5]}")
                # print(f"Filtered data: {filtered_data[:5]}")
            except Exception as e:
                print(f"Error processing channel {channel}: {str(e)}")

        self.app.processEvents()

def apply_filters(data, sampling_rate, channel):
    # Create a copy of the data to avoid modifying the original
    filtered_data = data.copy()
    
    # Apply notch filters for 50 Hz and 60 Hz
    DataFilter.perform_bandstop(filtered_data, sampling_rate, 50, 60, 4, FilterTypes.BUTTERWORTH.value, 0)
    
    # Apply bandpass filter
    if channel in [0, 1, 3]:  # Channels 1, 2, and 4 (0-indexed)
        DataFilter.perform_bandpass(filtered_data, sampling_rate, 0.3, 35.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    elif channel == 2:  # Channel 3 (0-indexed)
        DataFilter.perform_bandpass(filtered_data, sampling_rate, 10.0, 100.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    
    return filtered_data

def no_filter(data, sampling_rate, channel):
    return data  # Return the data without any filtering

def main():
    # Board setup
    params = BrainFlowInputParams()
    params.ip_port = 6677
    params.ip_address = "225.1.1.1"
    params.master_board = BoardIds.GANGLION_BOARD

    board = BoardShim(BoardIds.STREAMING_BOARD, params)

    # graph the data

    try:
        board.prepare_session()
        board.start_stream()

        # Create two Graph instances: one for filtered and one for unfiltered data
        graph_filtered = Graph(board, apply_filters, "Filtered EEG Data")
        graph_unfiltered = Graph(board, no_filter, "Unfiltered EEG Data")

        # Start the Qt event loop
        QtWidgets.QApplication.instance().exec_()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    main()