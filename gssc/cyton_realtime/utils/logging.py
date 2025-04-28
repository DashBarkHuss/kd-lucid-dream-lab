import logging
import threading
import sys

class ThreadSafeStreamHandler(logging.StreamHandler):
    """A thread-safe stream handler that ensures atomic writes to the output stream."""
    def __init__(self, stream=None):
        super().__init__(stream)
        self.lock = threading.Lock()

    def emit(self, record):
        """Emit a record in a thread-safe way."""
        with self.lock:
            try:
                msg = self.format(record)
                stream = self.stream or sys.stdout
                # Ensure the message ends with exactly one newline
                msg = msg.rstrip('\n') + '\n'
                stream.write(msg)
                stream.flush()
            except Exception:
                self.handleError(record)

def setup_logging():
    """Configure logging with thread-safe handler"""
    handler = ThreadSafeStreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Get the root logger and remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    # Create our module's logger
    logger = logging.getLogger(__name__)

    # Only suppress matplotlib font manager debug messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    # Keep brainflow board logger at INFO level for important messages
    board_logger = logging.getLogger('board_logger')
    board_logger.setLevel(logging.INFO)

    return logger, board_logger 