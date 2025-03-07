# kd-lucid-dream-lab

## How to stream from OpenBCI GUI to Brainflow

1. Open OpenBCI GUI
2. System Control Panel -> Ganglion Live
2. Select the following settings:
   1. Pick Transfer Protocol: BLED112 Dongle
   2. BLE Device: select your ganglion board
   3. Session Data: BDF+
   4. Brainflow Streamer: Choose "Network", Set IP address to 225.1.1.1, Set port to 6677
3. Start Session 
4. Start Data Stream
5. If you want the value in the OpenBCI GUI to match the value in the script, you need to remove the filters.   
   1. Click on the filters icon in the top left corner of the OpenBCI GUI
   2. Click "All" to turn the channel icons black which means no filters are applied
6. In a new terminal, run the script `stream-from-openbci-GUI.py`

In the terminal you'll see the samples ploted out in a vertical stream.

<div>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <p>Stream to brainflow demo - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/7c4b133287134a08a924a850928adf90">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/7c4b133287134a08a924a850928adf90-866d7ca113c60d57-full-play.gif">
    </a>
  </div>

## How to plot data from Brainflow with python

1. Follow steps 1-5 in the "How to stream from OpenBCI GUI to Brainflow" section
2. Run the script `plot_python_example.py`

<div>
    <a href="https://www.loom.com/share/7783bd62fe1b4170b1a8eb21419f26f6">
      <p>python - BrainFlow Plot  - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/7783bd62fe1b4170b1a8eb21419f26f6">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/7783bd62fe1b4170b1a8eb21419f26f6-54ac665bd899918e-full-play.gif">
    </a>
  </div>

## How to filter and plot data from Brainflow

1. Follow steps 1-5 in the "How to stream from OpenBCI GUI to Brainflow" section
2. Run the script `filtered_plot.py`

<div>
    <a href="https://www.loom.com/share/ee904b05f8484db69ce555ad0e6a11c5">
      <p>filtered python plot brainflow - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/ee904b05f8484db69ce555ad0e6a11c5">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/ee904b05f8484db69ce555ad0e6a11c5-fd57ab856caa8c2d-full-play.gif">
    </a>
  </div>

## How to playback data from a file

### Recording Data
You need to use the BrainFlow-RAW.csv file to play from the file
1. Open OpenBCI GUI
2. System Control Panel -> Ganglion Live
3.  Setting to record data:
  1. Pick Transfer Protocol: BLED112 Dongle
  2.  BLE Device: select your ganglion board
  3. Session Data: OpenBCI (I don't know if this actually matter, but it will cause an error the current GUI if you choose BDF+)
  4. Brainflow Streamer: File
3. Start Session 
4. Start Data Stream
5. Record your EEG/ECG/etc data
5. When done recording your session: Stop Data Stream
6. Stop Session
7. OpenBCI should have saved the session recording as `BrainFlow-RAW_<whatever you put in the session data name>.csv`

### Playback Data

1. Move the BrainFlow CSV file to the same folder as this script. 
2. Write your .csv's path into `/data/playback_from_file.py`
3. Run the script, and you should see the data you previosly recorded show up in the plot.


