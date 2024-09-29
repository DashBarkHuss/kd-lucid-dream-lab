# To connect the OpenBCI GUI to the computer running this script
# 1. start the openbci gui and connect to the board
# 2. Open the Networking Widget
# 3. Set the protocol to "Serial"
# 4. Set Data Type to "TimeSeriesFiltered" or whatever you want
# 5. Set the Serial Port to the port your board is connected to (Bluetooth)
# 6. Set the baud rate to 115200
# 7. Click "Start Data Stream""
# 8. Click "Start Serial Stream"
# 9. Run this script



# get brainflow streamserial port stream from openbci
import brainflow
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
import time
import numpy as np

# # here we can list the serial ports to get the names to use to enter as a param into brainflow
# import serial.tools.list_ports
# # List available serial ports
# ports = serial.tools.list_ports.comports()
# for port in ports:
#     print(f"Available port: {port.device}")

# # ---end of listing serial ports

# Set up parameters for Ganglion board
params = BrainFlowInputParams()
# use the blue tooth port
params.serial_port = '/dev/cu.Bluetooth-Incoming-Port'  # Use the port that OpenBCI GUI is forwarding to
params.timeout = 15

# Enable verbose logging
BoardShim.enable_dev_board_logger()

# Create board object
board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)  # Use SYNTHETIC_BOARD for networked mode

try:
    board.prepare_session()
    print("Session prepared successfully")
    board.start_stream()
    print("Stream started successfully")
    
    # Keep the stream running for a few seconds
    print("Streaming for 5 seconds...")
    time.sleep(5)
    
    # Get and print some data
    print("Attempting to get data...")
    data = board.get_current_board_data(10)
    print("Received data:")
    print(data)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
finally:
    if board.is_prepared():
        print("Releasing session...")
        board.release_session()
        print("Session released")

print("Script completed.")


