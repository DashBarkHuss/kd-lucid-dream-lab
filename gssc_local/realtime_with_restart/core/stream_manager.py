import multiprocessing
import logging
from typing import Optional, Tuple, Dict, Any
import time
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

logger = logging.getLogger(__name__)

class StreamManager:
    """Manages the lifecycle of the data streaming process and parent-child communication."""
    
    def __init__(self, playback_file: str, board_id: int = BoardIds.CYTON_DAISY_BOARD):
        self.playback_file = playback_file
        self.board_id = board_id
        self.parent_conn: Optional[multiprocessing.connection.Connection] = None
        self.child_conn: Optional[multiprocessing.connection.Connection] = None
        self.process: Optional[multiprocessing.Process] = None
        self.start_first_data_ts: Optional[float] = None
        
    def start_stream(self) -> None:
        """Start the data streaming process."""
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=self._run_board_stream,
            args=(self.playback_file, self.child_conn)
        )
        self.process.start()
        self.parent_conn.send(('start_ts', self.start_first_data_ts))
        
    def stop_stream(self) -> None:
        """Stop the data streaming process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.parent_conn:
            self.parent_conn.close()
        if self.child_conn:
            self.child_conn.close()
            
    def is_streaming(self) -> bool:
        """Check if the stream is currently active."""
        return self.process is not None and self.process.is_alive()
    
    def get_next_message(self) -> Optional[Tuple[str, Any]]:
        """Get the next message from the child process if available."""
        if self.parent_conn and self.parent_conn.poll():
            return self.parent_conn.recv()
        return None
    
    def _run_board_stream(self, playback_file: str, conn: multiprocessing.connection.Connection) -> None:
        """Child process that handles data acquisition from the board."""
        try:
            # Receive initial timestamp from parent process
            msg_type, start_first_data_ts = conn.recv()
            start_first_data_ts = float(start_first_data_ts) if start_first_data_ts is not None else None
            
            # Set up board configuration for playback
            params = BrainFlowInputParams()
            params.board_id = BoardIds.PLAYBACK_FILE_BOARD
            params.file = playback_file
            params.master_board = self.board_id

            # Initialize and start the board
            board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)
            board.prepare_session()
            board.config_board('old_timestamps')
            board.start_stream()

            time.sleep(0.1)  # Brief pause to ensure stream is started

            timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            last_valid_data_ts = None

            while True:
                # Get new data from the board
                board_data = board.get_board_data()

                # Check for gap in data (empty data array)
                if board_data.shape[1] == 0:
                    logger.info(f"Gap detected at {time.time()}, exiting and telling parent to restart. last_valid_data_ts: {last_valid_data_ts}")
                    conn.send(('last_ts', last_valid_data_ts))
                    logger.info(f"Closed connection at {time.time()}")
                    conn.close()
                    return

                timestamps = board_data[timestamp_channel]

                # If this is the first data chunk, set the start timestamp
                if start_first_data_ts is None:
                    start_first_data_ts = float(timestamps[0])
                    # Send the updated start_first_data_ts back to parent
                    conn.send(('start_ts', start_first_data_ts))

                last_valid_data_ts = float(timestamps[-1])
                
                # Send both data and metadata to parent process
                conn.send(('data', {
                    'board_data': board_data,
                }))

                time.sleep(1)  # Control the rate of data acquisition

        except Exception as e:
            logger.error(f"Error in board stream: {e}")
            conn.close() 