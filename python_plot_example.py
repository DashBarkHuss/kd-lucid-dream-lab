import argparse
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
    params.ip_port = 6677
    params.ip_address = "225.1.1.1"
    params.master_board = BoardIds.GANGLION_BOARD

    board_shim = BoardShim(BoardIds.STREAMING_BOARD, params)

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