# previously called DataAcquisition
import time
import pandas as pd
import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class BoardManager:
    """Manages BrainFlow board configuration.
    
    Primarily used for board setup and configuration via set_board_shim().
    
    Args:
        master_board_id (BoardIds): The BrainFlow board ID to use. Defaults to CYTON_DAISY_BOARD.
    """
    def __init__(self, master_board_id: BoardIds = BoardIds.CYTON_DAISY_BOARD):
        self.master_board_id = master_board_id  # Add master board ID
        self.board_timestamp_channel = None  # Will be set in set_board_shim
        self.board_shim = None
        self.sampling_rate = None
        


    def set_board_shim(self):
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

        params = BrainFlowInputParams()
        params.master_board = self.master_board_id  # Use class variable
        params.playback_file_max_count = 1
        params.playback_speed = 1
        params.playback_file_offset = 0

        self.board_shim = BoardShim(self.master_board_id, params)

        # Get sampling rate and timestamp channel from master board, not playback board
        self.sampling_rate = BoardShim.get_sampling_rate(self.master_board_id)
        self.board_timestamp_channel = BoardShim.get_timestamp_channel(self.master_board_id)
        
        return self.board_shim









