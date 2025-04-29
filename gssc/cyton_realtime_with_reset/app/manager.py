import os
import sys
import time
import select
import termios
import tty
import datetime
import threading
import multiprocessing
from typing import Optional

from pyqtgraph.Qt import QtWidgets
from montage import Montage

from ..config import DEBUG_VERBOSE
from ..acquisition.board import DataAcquisition
from ..signal.buffer import BufferManager
from ..processing.stream import DataStreamProcessor

class ApplicationState:
    """Manages the application state and lifecycle"""
    def __init__(self):
        self.running = False
        self.streaming = False
        self.processing = False
        self.error = None
        self.should_restart = False
        self.should_quit = False

class ApplicationManager:
    """Manages the application lifecycle and resources"""
    def __init__(self, input_file: str):
        self.state = ApplicationState()
        self.input_file = input_file
        self.data_acquisition = None
        self.buffer_manager = None
        self.stream_processor = None
        self.pyqt_app = None
        self.montage = Montage.minimal_sleep_montage()
        
        # Setup output directory
        self.output_dir = "../data/processed"
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"processed_{self.timestamp}.csv")
        
    def initialize(self):
        """Initialize the application"""
        try:
            # Initialize PyQt application
            self.pyqt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            
            # Initialize data acquisition
            self.data_acquisition = DataAcquisition(self.input_file)
            board_shim = self.data_acquisition.setup_board()
            
            # Initialize buffer manager
            self.buffer_manager = BufferManager(
                board_shim, 
                self.data_acquisition.sampling_rate, 
                self.montage
            )
            self.data_acquisition.set_buffer_manager(self.buffer_manager)
            
            # Initialize stream processor
            self.stream_processor = DataStreamProcessor(
                self.data_acquisition, 
                self.buffer_manager
            )
            
            # Add this line after creating stream_processor
            self.buffer_manager.stream_processor = self.stream_processor
            
            self.state.running = True
            return True
            
        except Exception as e:
            self.state.error = str(e)
            print(f"\rInitialization failed: {str(e)}")
            return False
            
    def start_stream(self):
        """Start the data stream and processing"""
        try:
            # Start data stream and processing
            self.data_acquisition.start_and_stream()
            self.state.streaming = True
            
            # Wait for initial data
            time.sleep(0.5)
            initial_data = self.data_acquisition.get_initial_data()
            
            if initial_data.size > 0:
                success = self.buffer_manager.add_data(initial_data, is_initial=True)
                if success:
                    self.buffer_manager.save_new_data(initial_data, is_initial=True)
                print(f"\rInitial data processing: {'Success' if success else 'Failed'}")
            
            # Start processing
            self.stream_processor.start_processing()
            self.state.processing = True
            return True
            
        except Exception as e:
            self.state.error = str(e)
            print(f"\rFailed to start stream: {str(e)}")
            return False
            
    def stop_stream(self):
        """Stop the data stream and processing"""
        try:
            if self.state.processing:
                if self.stream_processor:
                    self.stream_processor.stop_processing()
                    self.state.processing = False
                
            if self.state.streaming:
                self.data_acquisition.stop_stream()
                self.state.streaming = False
                
            return True
            
        except Exception as e:
            self.state.error = str(e)
            print(f"\rFailed to stop stream: {str(e)}")
            return False
            
    def cleanup(self):
        """Clean up all resources"""
        try:
            if self.state.processing:
                self.stop_stream()
            
            if self.buffer_manager and self.buffer_manager.visualizer:
                # Stop timers in the main thread
                if self.buffer_manager.visualizer.timer:
                    self.buffer_manager.visualizer.timer.stop()
                if self.buffer_manager.visualizer.countdown_timer:
                    self.buffer_manager.visualizer.countdown_timer.stop()
                self.buffer_manager.visualizer.close()
                
            if self.data_acquisition:
                self.data_acquisition.release()
                
            # Save processed data
            if self.buffer_manager:
                if self.buffer_manager.save_to_csv(self.output_file):
                    print(f"\rData saved to {self.output_file}")
                    if self.buffer_manager.validate_saved_csv(self.input_file):
                        print("\rCSV validation passed")
                    else:
                        print("\rCSV validation failed")
                        
        except Exception as e:
            print(f"\rCleanup failed: {str(e)}")
            
    def run(self):
        """Main application loop"""
        try:
            if not self.initialize():
                print("\rInitialization failed. Exiting...")
                return
                
            # Set terminal to raw mode for immediate key input
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while self.state.running:
                # Create communication pipes
                parent_conn, child_conn = multiprocessing.Pipe()
                self.data_acquisition.set_communication_pipes(parent_conn, child_conn)
                
                if not self.start_stream():
                    break
                    
                print("\rPress 'k' to restart the stream, 'q' to quit")
                
                while self.state.processing:
                    self.pyqt_app.processEvents()
                    
                    # Poll for messages from the board process
                    if parent_conn.poll():
                        msg_type, received = parent_conn.recv()
                        if msg_type == 'last_ts':
                            print(f"\n[Parent] Received last good timestamp: {received}")
                            # Stop current processing
                            if self.state.processing:
                                self.stream_processor.stop_processing()
                                self.state.processing = False
                                
                            if self.state.streaming:
                                self.data_acquisition.stop_stream()
                                self.state.streaming = False
                                
                            # Reinitialize the board
                            self.data_acquisition.setup_board()
                            
                            # Restart the stream
                            if not self.start_stream():
                                self.state.running = False
                                break
                            continue
                    
                    # Check for keyboard input
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
                        if key == 'k':
                            print("\rRestarting stream...")
                            # Properly cleanup before reinitializing
                            self.cleanup()
                            # Reset state
                            self.state = ApplicationState()
                            # Reinitialize everything
                            if not self.initialize():
                                print("\rFailed to reinitialize. Exiting...")
                                self.state.running = False
                                break
                            # Break out of the inner loop to restart the stream
                            break
                        elif key == 'q':
                            print("\rQuitting...")
                            self.state.running = False
                            break
                    
                    # Add a small delay to prevent CPU overuse
                    time.sleep(0.1)
                            
                if not self.state.running:
                    break
                    
        except Exception as e:
            print(f"\rApplication error: {str(e)}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.cleanup()

def main(csv_file: str):
    """Main entry point"""
    old_settings = None
    try:
        app_manager = ApplicationManager(csv_file)
        app_manager.run()
    except Exception as e:
        print(f"\rFatal error: {str(e)}")
        if old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sys.exit(1)

if __name__ == '__main__':
    main("data/test_data/consecutive_data.csv")  # Default file for direct execution 