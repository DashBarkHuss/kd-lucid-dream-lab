import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

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



def plot_data(signal, markers=None):
    app = pg.mkQApp("EEG Data Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="EEG Data")
    win.resize(1000,600)
    
    plot = win.addPlot(title="EEG Signal")
    plot.plot(signal, pen='b')
    plot.setLabel('left', "Amplitude", units='μV')
    plot.setLabel('bottom', "Sample")
    
    # Add legend
    legend = pg.LegendItem((80,60), offset=(70,20))
    legend.setParentItem(plot)
    legend.addItem(plot.listDataItems()[0], 'EEG Signal')
    
    # Add markers if specified
    if markers:
        for marker in markers:
            line = pg.InfiniteLine(pos=marker.x, angle=90, pen=marker.color)
            plot.addItem(line)
            
            # Add text label for marker
            text = pg.TextItem(f"{marker.name}: {marker.x}", color=marker.color, anchor=(0.5, 1))
            text.setPos(marker.x, plot.getViewBox().viewRange()[1][1])
            plot.addItem(text)
        
            # Add custom legend item for each marker
            legend.addItem(MarkerLegendItem(marker.color), marker.name)
    
    # Start Qt event loop
    if __name__ == '__main__':
        pg.exec()

# Usage
csv_file = 'data/BrainFlow-RAW_L_R_L_RL.csv'  # Update this path
channel_index = 0  # Adjust this to the correct channel index (0-based)

signal, left_eye_movements = detect_left_eye_movements(csv_file, channel_index)
signal, right_eye_movements = detect_right_eye_movements(csv_file, channel_index)

# plot with marker
# markers = [
#     Marker(500, 'r', 'Peak'),
#     Marker(1000, 'g', 'Valley'),
#     Marker(1500, 'y', 'Artifact')
# ]
# make markers from left_eye_movements and right_eye_movements
markers = [Marker(x, 'r', 'Left Eye Movement') for x in left_eye_movements]
markers += [Marker(x, 'g', 'Right Eye Movement') for x in right_eye_movements]

plot_data(signal, markers)
print("done 4")

