# This script plays back previously recorded EEG data from a BrainFlow CSV file.
# It visualizes the data in real-time using PyQtGraph, applying various filters
# (detrend, bandpass, bandstop) to clean the signal. The script is configured to
# work with Ganglion board recordings and displays each EEG channel in separate
# plots. To use, place your BrainFlow-RAW.csv file in the data directory and run
# the script. The Graph class handles visualization while the main function sets up
# the board connection with the playback file.

# You need to use the BrainFlow-RAW.csv file to play from the file
# Setting to record data:
# 1. System Control Panel -> Ganglion Live
# 2. Select the following settings:
# - Pick Transfer Protocol: BLED112 Dongle
# - BLE Device: select your ganglion board
# - Session Data: OpenBCI (I don't know if this actually matter, but it will cause an error the current GUI if you choose BDF+)
# - Brainflow Streamer: File
# 3. Start Session 
# 4. Start Data Stream
# 5. Stop Data Stream
# 6. Stop Session
# 7. The session should be saved as BrainFlow-RAW_<whatever you put in the session data name>.csv

### Playback Data

# 1. Move the BrainFlow CSV file to the same folder as this script. 
# 2. Write your .csv's path into `/data/playback_from_file.py`
# 3. Run the script, and you should see the data you previosly recorded show up in the plot.

import logging

import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import brainflow





class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

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
                # Apply filters
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(channel_data, self.sampling_rate, 3.0, 45.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 48.0, 52.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(channel_data, self.sampling_rate, 58.0, 62.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                self.curves[count].setData(channel_data.tolist())
            except brainflow.BrainFlowError as e:
                print(f"Error processing channel {channel}: {str(e)}")

        self.app.processEvents()


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
    # params.file = "/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/playback/BrainFlow-RAW.csv"


    board_shim = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

    try:
        board_shim.prepare_session()
        board_shim.start_stream()
        Graph(board_shim)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()